import random
import time
import os

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ReLU, MaxPool1d, ReLU, Dropout

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def pinn_loss_navier_stokes(pred, features):
    """Loss function for the Navier-Stokes equation

    Args:
        pred (torch.Tensor(N, 4)): Predicted values of the model denormalized
        pos (torch.Tensor(N, 2)): Position of the nodes

    Returns:
        torch.Tensor: Loss value
    """

    u = pred[:, 0]
    v = pred[:, 1]
    p = pred[:, 2]
    nut = pred[:, 3]
        
    x = features[:, 0]
    y = features[:, 1]
    

    rho_inv = 1.0/1.225

    mu = 1.55e-5

    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=False)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=False)[0]
    
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=False)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=False)[0]
    
    nut_x = torch.autograd.grad(nut, x, torch.ones_like(nut), create_graph=False)[0]
    nut_y = torch.autograd.grad(nut, y, torch.ones_like(nut), create_graph=False)[0]
    
    
    laplace_u = u_xx + v_yy
    
    r1 = u_x + v_y
    
    r2 = u*u_x + v*u_y + p_x*rho_inv - (mu + nut)*(u_xx + u_yy) - (nut_x * u_x + nut_y * u_y)
    r3 = u*v_x + v*v_y + p_y*rho_inv - (mu + nut)*(v_xx + v_yy) - (nut_x * v_x + nut_y * v_y)
    x.requires_grad_(False)
    y.requires_grad_(False)

    return torch.mean((r1**2 + r2**2 + r3**2 + laplace_u**2 + nut_x**2 + nut_y**2))


class SharedMLP(nn.Module):
    def __init__(self, space_variable: int, hidden_layers: list = [64, 64], out_channels: int = 32, dropout: float = 0.15):
        # FIXME: Remove the necessity of using transpose
        super().__init__()
        if len(hidden_layers) <= 0:
            hidden_layers = [out_channels]
        
        self.model = nn.Sequential()
        self.model.add_module("conv1", Conv1d(in_channels=space_variable, out_channels=hidden_layers[0], kernel_size=1))
        self.model.add_module("relu1", ReLU())
        self.model.add_module("bn1", nn.BatchNorm1d(hidden_layers[0]))
        
        for i in range(1, len(hidden_layers)):
            self.model.add_module(f"conv{i+1}", Conv1d(in_channels=hidden_layers[i-1], out_channels=hidden_layers[i], kernel_size=1))
            self.model.add_module(f"relu{i+1}", ReLU())
            self.model.add_module(f"bn{i+1}", nn.BatchNorm1d(hidden_layers[i]))
            self.model.add_module(f"dropout{i+1}", Dropout(dropout))
            
        if len(hidden_layers) > 0:
            self.model.add_module("conv_last", Conv1d(in_channels=hidden_layers[-1], out_channels=out_channels, kernel_size=1))
            self.model.add_module("relu_last", ReLU())
            self.model.add_module("bn_last", nn.BatchNorm1d(out_channels))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model

        Args:
            x (torch.Tensor): Input tensor of shape [N, space_variable]

        Returns:
            torch.Tensor: Output tensor of shape [N, out_channels]
        """
        return self.model(x.unsqueeze(0)).squeeze(0).T

            
class BasicSimulator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "AirfRANSSubmission"
        use_cuda = torch.cuda.is_available()
        self.device = 'cuda:0' if use_cuda else 'cpu'
        self.model = SharedMLP(7, [64, 64, 8, 64, 64, 64, 8, 64, 64], 4)
        self.scaler = StandardScaler(copy=False)
        self.target_scaler = MinMaxScaler(copy=False)
        self.hparams = kwargs
        if use_cuda:
            print('Using GPU')
        else:
            print('Using CPU')


    def process_dataset(self, dataset, training: bool) -> DataLoader:
        print("Processing dataset")
        coord_x=dataset.data['x-position']
        coord_y=dataset.data['y-position']
        surf_bool=dataset.extra_data['surface']
        position = np.stack([coord_x,coord_y],axis=1)

        nodes_features, node_labels = dataset.extract_data()
        fitted = False
        try:
            check_is_fitted(self.scaler)
            check_is_fitted(self.target_scaler)
            fitted = True
        except:
            pass
        if training or not fitted:
            print("Scale train data")
            nodes_features = self.scaler.fit_transform(nodes_features)
            node_labels = self.target_scaler.fit_transform(node_labels)
        else:
            print("Scale not train data")
            nodes_features = self.scaler.transform(nodes_features)
            node_labels = self.target_scaler.transform(node_labels)        

        torchDataset=[]
        nb_nodes_in_simulations = dataset.get_simulations_sizes()
        start_index = 0
        print("Start processing dataset")
        i = 0
        for nb_nodes_in_simulation in nb_nodes_in_simulations:
            i += 1
            end_index = start_index+nb_nodes_in_simulation
            simulation_positions = torch.tensor(position[start_index:end_index,:], dtype = torch.float) 
            simulation_features = torch.tensor(nodes_features[start_index:end_index,:], dtype = torch.float) 
            simulation_labels = torch.tensor(node_labels[start_index:end_index,:], dtype = torch.float) 
            simulation_surface = torch.tensor(surf_bool[start_index:end_index])
            sampleData = Data(pos=simulation_positions,
                            x=simulation_features, 
                            y=simulation_labels,
                            surf = simulation_surface.bool()) 
            torchDataset.append(sampleData)
            start_index += nb_nodes_in_simulation
        return DataLoader(dataset=torchDataset, batch_size=1)

    def train(self, train_dataset, save_path=None, local=False):
        print("Training")

        if local:
            # Check if file exists
            if os.path.exists("train_dataset_meshed.pth"):
                train_dataset = torch.load("train_dataset_meshed.pth")
            else:
                # Process the dataset
                train_dataset = self.process_dataset(dataset=train_dataset, training=True)
                # Save the dataset
                torch.save(train_dataset, "train_dataset_meshed.pth")
        else:
            train_dataset = self.process_dataset(dataset=train_dataset, training=True)
        model = global_train(self.device, train_dataset, self.model, self.hparams,criterion = 'MSE_weighted', local=local)
        # Save the model
        if local:
            torch.save(self, "model.pth")
        torch.save(self, "model.pth")

    def predict(self, dataset, **kwargs):
        test_dataset = self.process_dataset(dataset, training=False)
        self.model.eval()
        avg_loss_per_var = np.zeros(4)
        avg_loss = 0
        avg_loss_surf_var = np.zeros(4)
        avg_loss_vol_var = np.zeros(4)
        avg_loss_surf = 0
        avg_loss_vol = 0
        iterNum = 0

        print("Dataset length: ", len(test_dataset))
        predictions=[]
        with torch.no_grad():
            for data in test_dataset:        
                data_clone = data.clone()
                data_clone = data_clone.to(self.device)
                out = self.model(data_clone.x.T)

                targets = data_clone.y
                loss_criterion = nn.MSELoss(reduction = 'none')

                loss_per_var = loss_criterion(out, targets).mean(dim = 0)
                loss = loss_per_var.mean()
                loss_surf_var = loss_criterion(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim = 0)
                loss_vol_var = loss_criterion(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim = 0)
                loss_surf = loss_surf_var.mean()
                loss_vol = loss_vol_var.mean()  

                avg_loss_per_var += loss_per_var.cpu().numpy()
                avg_loss += loss.cpu().numpy()
                avg_loss_surf_var += loss_surf_var.cpu().numpy()
                avg_loss_vol_var += loss_vol_var.cpu().numpy()
                avg_loss_surf += loss_surf.cpu().numpy()
                avg_loss_vol += loss_vol.cpu().numpy()  
                iterNum += 1
                prediction = self._post_process(out)
                prediction = out.cpu().data.numpy()
                predictions.append(prediction)
        print("Results for test")
        print(avg_loss/iterNum, avg_loss_per_var/iterNum, avg_loss_surf_var/iterNum, avg_loss_vol_var/iterNum, avg_loss_surf/iterNum, avg_loss_vol/iterNum)
        predictions= np.vstack(predictions)
        predictions = dataset.reconstruct_output(predictions)
        return predictions
    
    def _post_process(self, data):
        try:
            processed = self.target_scaler.inverse_transform(data)
        except TypeError:
            processed = self.target_scaler.inverse_transform(data.cpu())
        return processed


def global_train(device, train_dataset: DataLoader, network, hparams: dict, criterion='MSE', reg=1, local=False):
    model = network.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
    )

    best_loss = float('inf')
    patience = hparams.get('patience', 5)  # Set default patience if not provided
    patience_counter = 0

    train_loss_list = []

    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    epoch_nb = 0

    for epoch in pbar_train:
        epoch_nb += 1
        print('Epoch: ', epoch_nb)
        train_dataset_sampled = []
        for data in train_dataset:
            data_sampled = data.clone()
            idx = random.sample(range(data_sampled.x.size(0)), hparams['subsampling'])
            idx = torch.tensor(idx)

            data_sampled.x = data_sampled.x[idx]
            data_sampled.y = data_sampled.y[idx]
            data_sampled.surf = data_sampled.surf[idx]
            train_dataset_sampled.append(data_sampled)
        train_loader = DataLoader(train_dataset_sampled, batch_size = hparams['batch_size'], shuffle = True)
        del(train_dataset_sampled)

        train_loss, _, _, _, _, _ = train_model(device, model, train_loader, optimizer, lr_scheduler, criterion, reg=reg)
        del(train_loader)
        train_loss_list.append(train_loss)
        # Check for early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0  # Reset counter if there's improvement
            # Save the model or any other actions
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1  # Increment counter if no improvement

        if patience_counter >= patience:
            print(f"Early stopping activated. No improvement for {patience} epochs.")
            break

    # Load the best model
    model.load_state_dict(torch.load("best_model.pth"))
    
    return model
    

def train_model(device, model, train_loader, optimizer, scheduler, criterion = 'MSE', reg = 1):
    model.to(device)
    model.train()
    avg_loss_per_var = torch.zeros(4, device = device)
    avg_loss = 0
    avg_loss_surf_var = torch.zeros(4, device = device)
    avg_loss_vol_var = torch.zeros(4, device = device)
    avg_loss_surf = 0
    avg_loss_vol = 0
    iterNum = 0
    start = time.time()
    for data in train_loader:
        data_clone = data.clone()
        data_clone = data_clone.to(device)   
        optimizer.zero_grad()  
        x = data_clone.x.T
        x.requires_grad_(True)
        out = model(x)
        pinn_loss = pinn_loss_navier_stokes(out, x.T)
        
        targets = data_clone.y.to(device)   
        if criterion == 'MSE' or criterion == 'MSE_weighted':
            loss_criterion = nn.MSELoss(reduction = 'none')
        elif criterion == 'MAE':
            loss_criterion = nn.L1Loss(reduction = 'none')
        loss_per_var = loss_criterion(out, targets).mean(dim = 0)
        total_loss = loss_per_var.mean() + 1.0*pinn_loss
        loss_surf_var = loss_criterion(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim = 0)
        loss_vol_var = loss_criterion(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim = 0)
        loss_surf = loss_surf_var.mean()
        loss_vol = loss_vol_var.mean()
        if criterion == 'MSE_weighted':            
            (loss_vol + reg*loss_surf).backward()      
        else:
            total_loss.backward()
        
        optimizer.step()
        scheduler.step()
        avg_loss_per_var += loss_per_var
        avg_loss += total_loss
        avg_loss_surf_var += loss_surf_var
        avg_loss_vol_var += loss_vol_var
        avg_loss_surf += loss_surf
        avg_loss_vol += loss_vol 
        iterNum += 1
    print("GPU memory allocated: ", torch.cuda.memory_allocated(device)/1e9, "GB, Max memory allocated: ", torch.cuda.max_memory_allocated(device)/1e9, "GB")
    print("Time for epoch: ", time.time()-start)
    print("Loss: ", avg_loss.cpu().data.numpy()/iterNum)

    return avg_loss.cpu().data.numpy()/iterNum, avg_loss_per_var.cpu().data.numpy()/iterNum, avg_loss_surf_var.cpu().data.numpy()/iterNum, avg_loss_vol_var.cpu().data.numpy()/iterNum, \
            avg_loss_surf.cpu().data.numpy()/iterNum, avg_loss_vol.cpu().data.numpy()/iterNum
