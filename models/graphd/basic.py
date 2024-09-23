import time
import os

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Conv1d, ReLU, MaxPool1d, ReLU, Dropout
from torch_geometric.nn import SAGEConv, TransformerConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from parser import load_edges

class GraphD(torch.nn.Module):
    def __init__(self, device, in_dim, hidden_dims: list, out_dim, dropout=0.2):
        super().__init__()
        self.device = device
        self.dropout = dropout
        self.heads = 4
        hidden_dims.append(out_dim)
        self.transformers = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.transformers.append(TransformerConv(in_dim, hidden_dim, heads=self.heads, dropout=0.1).to(self.device))
            in_dim = hidden_dim * self.heads
    
    def forward(self, data) -> Tensor: 
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        for transformer in self.transformers[:-1]:
            x = F.relu(transformer(x, edge_index))
            
        x = self.transformers[-1](x, edge_index)
        return x # TODO: choose activation function according to the variable
            
class BasicSimulator(nn.Module):
    def __init__(self, benchmark, **kwargs):
        super().__init__()
        self.name = "AirfRANSSubmission"
        use_cuda = torch.cuda.is_available()
        self.device = 'cuda:0' if use_cuda else 'cpu'
        self.model = GraphD(self.device, 5, [128, 512, 256], 4)
        self.scaler = StandardScaler(copy=False)
        self.target_scaler = MinMaxScaler(copy=False)
        self.hparams = kwargs
        self.benchmark_path = benchmark.benchmark_path

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

        print("Extracting edges")
        edges = load_edges(dataset, self.benchmark_path)
        print("Edges extracted")
        torchDataset=[]
        nb_nodes_in_simulations = dataset.get_simulations_sizes()
        simulation_names = dataset.extra_data["simulation_names"]
        print("Number of simulations: ", simulation_names[:][1])
        start_index = 0
        print("Start processing dataset")
        i = 0
        for name, nb_nodes_in_simulation in zip(simulation_names, nb_nodes_in_simulations):
            end_index = start_index+nb_nodes_in_simulation
            simulation_positions = torch.tensor(position[start_index:end_index,:], dtype = torch.float) 
            simulation_features = torch.tensor(nodes_features[start_index:end_index,:], dtype = torch.float) 
            simulation_labels = torch.tensor(node_labels[start_index:end_index,:], dtype = torch.float) 
            simulation_surface = torch.tensor(surf_bool[start_index:end_index])
            print("Simulation ", i, " with ", nb_nodes_in_simulation, " nodes")
            sampleData = Data(pos=simulation_positions,
                            x=simulation_features, 
                            y=simulation_labels,
                            edge_index=Tensor(edges[name[0]]),
                            surf = simulation_surface.bool()) 
            torchDataset.append(sampleData)
            start_index += nb_nodes_in_simulation
            i += 1
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
                out = self.model(data_clone)

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
        out = model(data_clone)
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
    print("GPU memory allocated: ", torch.cuda.memory_allocated(device)/1e9, "GB, Max memory allocated: ", torch.cuda.max_memory_allocated(device)/1e9, "GB")
    print("Time for epoch: ", time.time()-start)
    print("Loss: ", avg_loss.cpu().data.numpy()/iterNum)

    return avg_loss.cpu().data.numpy()/iterNum, avg_loss_per_var.cpu().data.numpy()/iterNum, avg_loss_surf_var.cpu().data.numpy()/iterNum, avg_loss_vol_var.cpu().data.numpy()/iterNum, \
            avg_loss_surf.cpu().data.numpy()/iterNum, avg_loss_vol.cpu().data.numpy()/iterNum

