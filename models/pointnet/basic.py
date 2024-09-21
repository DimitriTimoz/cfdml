import time
import os

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear, ReLU, Dropout

import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d
from torch_geometric.nn import MessagePassing, fps, radius, global_max_pool

from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='max')
        
        # MLP avec BatchNorm et ReLU
        self.mlp = Sequential(
            Linear(in_channels + 3, out_channels),
            BatchNorm1d(out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            BatchNorm1d(out_channels),
            ReLU(),
        )
    
    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Propager les informations
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j: torch.Tensor, pos_j: torch.Tensor, pos_i: torch.Tensor) -> torch.Tensor:
        # Combiner les caractéristiques des nœuds avec les informations de position relative
        relative_pos = pos_j - pos_i  # Dimension 3
        edge_feat = torch.cat([h_j, relative_pos], dim=-1)  # in_channels + 3
        return self.mlp(edge_feat)

class SetAbstractionLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_samples, radius):
        super().__init__()
        self.num_samples = num_samples
        self.radius = radius
        self.conv = PointNetLayer(in_channels, out_channels)
    
    def forward(self, h, pos):
        # Échantillonnage par FPS
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)  # Supposons un seul lot
        idx = fps(pos, batch, ratio=self.num_samples / pos.size(0))
        pos_sampled = pos[idx]
        h_sampled = h[idx]
        
        # Regroupement par rayon
        edge_index = radius(pos, pos_sampled, self.radius, batch, batch[idx])
        
        # Appliquer le PointNetLayer
        h_new = self.conv(h_sampled, pos, edge_index)
        return h_new, pos_sampled

class ImprovedPointNetPlusPlus(torch.nn.Module):
    def __init__(self, device, in_channels: int, num_attributes: int):
        super().__init__()
        self.device = device
        self.num_attributes = num_attributes
        
        # Couche d'abstraction hiérarchiques
        self.sa1 = SetAbstractionLayer(in_channels + 2, 128, num_samples=512, radius=0.1).to(device)
        self.sa2 = SetAbstractionLayer(128, 256, num_samples=128, radius=0.2).to(device)
        self.sa3 = SetAbstractionLayer(256, 512, num_samples=32, radius=0.4).to(device) 
        
        # Global Feature Extraction
        self.conv_global = PointNetLayer(512, 1024).to(device)
        
        # MLP pour la prédiction des attributs
        self.mlp = Sequential(
            Dropout(0.3),
            Linear(1024, 1024),
            BatchNorm1d(1024),
            ReLU(),
            Dropout(0.3),
            Linear(1024, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(0.3),
            Linear(512, num_attributes)  # Nombre d'attributs en sortie
        ).to(device)
    
    def forward(self, h: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # Première couche d'abstraction
        h, pos = self.sa1(h, pos)
        h = F.relu(h)
        
        # Deuxième couche d'abstraction
        h, pos = self.sa2(h, pos)
        h = F.relu(h)
        
        # Troisième couche d'abstraction
        h, pos = self.sa3(h, pos)
        h = F.relu(h)
        
        # Extraction de caractéristiques globales
        edge_index = self.radius_graph(pos, pos, r=0.5)  # Exemple de rayon pour la globalisation
        h = self.conv_global(h, pos, edge_index)
        h = F.relu(h)
        
        # Agrégation globale (max pooling)
        h = global_max_pool(h, torch.zeros(h.size(0), dtype=torch.long, device=h.device))
        
        # Prédiction des attributs
        h = self.mlp(h)
        return h  # Forme de sortie : [num_nodes, num_attributes]
    
    def radius_graph(self, pos, pos_sampled, r):
        # Fonction utilitaire pour créer un graphe de rayon global
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        batch_sampled = batch[:pos_sampled.size(0)]
        return radius(pos, pos_sampled, r, batch, batch_sampled)
            
class BasicSimulator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "AirfRANSSubmission"
        use_cuda = torch.cuda.is_available()
        self.device = 'cuda:0' if use_cuda else 'cpu'
        self.model = ImprovedPointNetPlusPlus(self.device, 2, 4)
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
