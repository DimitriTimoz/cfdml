import time
import os

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Conv1d, ReLU, MaxPool1d

import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout, LayerNorm
from torch_geometric.nn import MessagePassing, fps, radius, global_max_pool

from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

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
        # x has shape [N, space_variable]
        return self.model(x.unsqueeze(0)).squeeze(0).T

class TNet(nn.Module):
    """Transformation Network for regressing the transformation matrix."""
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        self.mlp = SharedMLP(space_variable=k, hidden_layers=[64, 128], out_channels=1024)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, k*k)
        )

        # Initialize as an identity matrix
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(torch.eye(k).view(-1))

    def forward(self, x):
        """Forward pass of the model

        Args:
            x (torch.Tensor)

        Returns:
            torch.Tensor: Shape (k, k)
        """
        x = self.mlp(x)  # (N, 1024)
        x = torch.max(x.T, 1)[0]  # Max pooling across points, shape (1024)

        x = self.fc(x)  # Shape (k*k)
        x = x.view(self.k, self.k)
        return x
    

class SharedMLP(nn.Module):
    def __init__(self, space_variable: int, hidden_layers: list = [64, 64], out_channels: int = 32, dropout: float = 0.15):
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
        # x has shape [N, space_variable]
        return self.model(x.unsqueeze(0)).squeeze(0).T
        

class FinitePoint(torch.nn.Module):
    def __init__(self, device, pos_dim: int, num_features: int, num_attributes: int):
        super().__init__()
        self.device = device
        self.num_attributes = num_attributes
        
        self.mlp1 = SharedMLP(space_variable=pos_dim, hidden_layers=[64], out_channels=64)
        self.tnet1 = TNet(k=pos_dim)

        self.mlp2 = SharedMLP(space_variable=64+num_features, hidden_layers=[128], out_channels=1024)
        self.tnet2 = TNet(k=64)
        
        self.mlp3 = SharedMLP(space_variable=1024+64, hidden_layers=[512, 256], out_channels=128)
        self.mlp4 = SharedMLP(space_variable=128, hidden_layers=[], out_channels=num_attributes)
        
    
    def forward(self, features: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model

        Args:
            features (torch.Tensor): Tensor of shape [N, num_features]
            pos (torch.Tensor): Tensor of shape [N, pos_dim]

        Returns:
            torch.Tensor: Tensor of shape [N, num_attributes]
        """
        N = features.shape[0]
         
        pos = pos.T # Shape [n_dim, N]
        features = features.T # Shape [num_features, N]
        
        pos_clone = pos.clone()
        pos = self.tnet1(pos) # Shape [2, 2]
        pos = pos_clone.T@pos # Shape [N, 2]
        # Positional encoding
        pos = self.mlp1(pos.T) # Shape [N, 64]
        
        pos_clone = pos.clone()
        print(pos.shape)
        pos = self.tnet2(pos.T) # Shape [64, 64]
        pos = pos_clone@pos  # Shape [N, 64]

        # Concatenate the positional encoding with the features
        x = torch.cat([features, pos.T], dim=0) # Shape [num_features+64, N]
        
        # Pass through the first MLP
        x = self.mlp2(x) # Shape [N, 1024]
        pooling = MaxPool1d(kernel_size=x.shape[0]) 
        x = pooling(x.T).squeeze() # [1024]

        x = x.repeat(N, 1) # [N, 1024]
        
        # Concatenate the positional encoding with the features
        x = torch.cat([x.T, pos.T], dim=0) 
        x = self.mlp3(x)
        x = self.mlp4(x.T)
        
        return x
            
class BasicSimulator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "AirfRANSSubmission"
        use_cuda = torch.cuda.is_available()
        self.device = 'cuda:0' if use_cuda else 'cpu'
        self.model = FinitePoint(self.device, pos_dim=2, num_features=5, num_attributes=4)
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
        print(dataset.data["x-inlet_velocity"])
        position = np.stack([coord_x,coord_y],axis=1)

        nodes_features, node_labels = dataset.extract_data()
        nodes_features = nodes_features[:, 2:]
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
                # Load the dataset
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
                out = self.model(data_clone.x, data_clone.pos)

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



def custom_collate_fn(batch):
    print("in Batch: ", batch)
    return Data.from_data_list(batch)

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

        train_loss, _, _, _, _, _ = train_model(device, model, train_dataset, optimizer, lr_scheduler, criterion, reg=reg)

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
        out = model(data_clone.x, data_clone.pos)
        targets = data_clone.y.to(device)

        if criterion == 'MSE' or criterion == 'MSE_weighted':
            loss_criterion = nn.MSELoss(reduction = 'none')
        elif criterion == 'MAE':
            loss_criterion = nn.L1Loss(reduction = 'none')
        loss_per_var = loss_criterion(out, targets).mean(dim = 0)
        total_loss = loss_per_var.mean()
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
    print("Time for epoch: ", time.time()-start)
    print("Loss: ", avg_loss.cpu().data.numpy()/iterNum)

    return avg_loss.cpu().data.numpy()/iterNum, avg_loss_per_var.cpu().data.numpy()/iterNum, avg_loss_surf_var.cpu().data.numpy()/iterNum, avg_loss_vol_var.cpu().data.numpy()/iterNum, \
            avg_loss_surf.cpu().data.numpy()/iterNum, avg_loss_vol.cpu().data.numpy()/iterNum
