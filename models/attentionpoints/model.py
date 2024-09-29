import os
import time
import random
import math 
import datetime as dt
from typing import List

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv1d, ReLU, MaxPool1d, ReLU, Dropout, LayerNorm, Linear
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint

from lips.dataset.scaler.standard_scaler_iterative import StandardScalerIterative

def smoothSoftmax(x):
    s = torch.nn.functional.softmax(x, dim=1)
    ss = torch.nn.functional.relu(x)
    num = ss * 0.1 + s
    denom = num.sum(1).unsqueeze(-1)
    return num / denom

class MLP(torch.nn.Module):
    def __init__(self, layers_sizes):
        super(MLP, self).__init__()
        layers = [torch.nn.Linear(layers_sizes[i], layers_sizes[i+1]) for i in range(len(layers_sizes)-1)]
        self.mlp = torch.nn.Sequential()
        for i, layer in enumerate(layers):
            self.mlp.append(layer)
            if i < len(layers) - 1:
                self.mlp.append(torch.nn.ReLU())
    def forward(self, x):
        return self.mlp(x)


class AttentionBlock(torch.nn.Module):
    def __init__(self, sIN, sOUT, yDIM=7, sPROJ=None):
        super(AttentionBlock, self).__init__()

        if sPROJ is None:
            sPROJ = sOUT

        self.k = torch.nn.Linear(yDIM, sPROJ, bias=False)
        self.q = torch.nn.Linear(sIN, sPROJ, bias=False)
        self.v = torch.nn.Linear(yDIM, sPROJ, bias=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        YK = self.k(y)
        XQ = self.q(x)
        YV = self.v(y)

        M = torch.matmul(XQ, YK.t())/np.sqrt(YK.shape[1])
        M = smoothSoftmax(M)
        output = torch.matmul(M, YV)

        return output

class TransformerBlock(torch.nn.Module):
    def __init__(self, sIN, sOUT, yDIM=7, sPROJ=None, layers: List[int] = [8, 64, 64, 8]):
        super(TransformerBlock, self).__init__()
        
        self.att = AttentionBlock(sIN, sOUT, yDIM, sPROJ)
        self.mlp = MLP(layers)
        self.layer_norm = LayerNorm(sOUT)

    def custom_forward(self, x, z1, w):
        with autocast():
            z1 = self.att(z1, w)
            z1 = z1 + x
            z2 = self.layer_norm(z1)
            z2 = self.mlp(z2)
            return z2 + z1

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z1 = self.layer_norm(x)
        w = self.layer_norm(y)
        # Passez x en argument
        z = checkpoint.checkpoint(self.custom_forward, x, z1, w) if self.training else self.custom_forward(x, z1, w)
        return z

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
            x (torch.Tensor) (N, k)

        Returns:
            torch.Tensor: Shape (k, k)
        """
        x = self.mlp(x)  # (N, 1024)
        x = torch.max(x.T, 1)[0]  # Max pooling across points, shape (1024)

        x = self.fc(x)  # Shape (k*k)
        x = x.view(self.k, self.k) # Shape (k, k)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, device, dim: int, output_channels: int = 64):
        super().__init__()
        self.device = device
        self.dim = dim
        self.output_channels = output_channels
        
        self.tnet1 = TNet(k=self.dim)
        self.mlp1 = SharedMLP(space_variable=self.dim, hidden_layers=[64], out_channels=64)

        self.tnet2 = TNet(k=64)
        self.mlp2 = SharedMLP(space_variable=64, hidden_layers=[128, 128], out_channels=self.output_channels)
                
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model

        Args:
            features (torch.Tensor): Tensor of shape [N, num_features]
            pos (torch.Tensor): Tensor of shape [N, pos_dim]

        Returns:
            torch.Tensor: Tensor of shape [N, num_attributes]
        """
         
        features = features.T # Shape [num_features, N]
        features_clone = features.clone()
        features = self.tnet1(features) # Shape [num_features, num_features]
        features = features.T@features_clone # Shape [N, num_features]
        # Positional encoding
        features = self.mlp1(features) # Shape [N, 64]
        
        features_clone = features.clone()

        features = self.tnet2(features.T) # Shape [64, 64]
        features = features_clone@features  # Shape [N, 64]

        # Concatenate the positional encoding with the features
        x = features.T # Shape [64, N]
        
        # Pass through the first MLP
        x = self.mlp2(x) # Shape [N, 1024]

        return x


class AttentionPoint(torch.nn.Module):
    def __init__(self, device, input_dim: int, num_attributes: int):
        super().__init__()
        self.device = device
        self.input_dim = input_dim 
        
        self.sample_size = 3000
        
        self.encoder = PointNetEncoder(device, dim=input_dim, output_channels=32)
        self.transf1 = TransformerBlock(32, 32, yDIM=32, layers=[32, 64, 64, 64, 32])
        self.transf2 = TransformerBlock(32, 32, yDIM=32, layers=[32, 64, 64, 64, 32])
        self.transf3 = TransformerBlock(32, 32, yDIM=32, layers=[32, 64, 64, 64, 32])

        self.decoder = PointNetEncoder(device, dim=32, output_channels=num_attributes)
        

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model

        Args:
            features (torch.Tensor): Tensor of shape [N, num_features]
            pos (torch.Tensor): Tensor of shape [N, pos_dim]

        Returns:
            torch.Tensor: Tensor of shape [N, num_attributes]
        """
        x = self.encoder(features) # Shape [N, 32]
        
        # Sampling
        sample = x[torch.randint(0, x.shape[0], (self.sample_size,))] # Shape [sample_size, num_features]
        
        x = self.transf1(x, sample) # Shape [N, 32]
        sample = self.transf1(sample, sample) # Shape [sample_size, 32]
        
        x = self.transf2(x, sample) # Shape [N, 32]
        sample = self.transf2(sample, sample) # Shape [sample_size, 32]
        
        out = self.transf3(x, sample) # Shape [N, 32]
            
        out = self.decoder(out) # Shape [N, 4]
        return out

class AugmentedSimulator():
    def __init__(self,benchmark,**kwargs):
        self.name = "AirfRANSSubmission"
        chunk_sizes=benchmark.train_dataset.get_simulations_sizes()
        scalerParams={"chunk_sizes":chunk_sizes}
        self.scaler = StandardScalerIterative(**scalerParams)

        self.model = None
        self.hparams = kwargs
        use_cuda = torch.cuda.is_available()
        self.device = 'cuda:0' if use_cuda else 'cpu'
        if use_cuda:
            print('Using GPU')
        else:
            print('Using CPU')

        self.model = AttentionPoint(self.device, input_dim=7, num_attributes=4).to(self.device)

    def process_dataset(self, dataset, training: bool) -> DataLoader:
        coord_x=dataset.data['x-position']
        coord_y=dataset.data['y-position']
        surf_bool=dataset.extra_data['surface']
        position = np.stack([coord_x,coord_y],axis=1)

        nodes_features,node_labels=dataset.extract_data()
        if training:
            print("Normalize train data")
            nodes_features, node_labels = self.scaler.fit_transform(nodes_features, node_labels)
            print("Transform done")
        else:
            print("Normalize not train data")
            nodes_features, node_labels = self.scaler.transform(nodes_features, node_labels)
            print("Transform done")

        torchDataset=[]
        nb_nodes_in_simulations = dataset.get_simulations_sizes()
        start_index = 0
        # check alive
        t = dt.datetime.now()

        for nb_nodes_in_simulation in nb_nodes_in_simulations:
            #still alive?
            if dt.datetime.now() - t > dt.timedelta(seconds=60):
                print("Still alive - index : ", end_index)
                t = dt.datetime.now()
            end_index = start_index+nb_nodes_in_simulation
            simulation_positions = torch.tensor(position[start_index:end_index,:], dtype = torch.float) 
            simulation_features = torch.tensor(nodes_features[start_index:end_index,:], dtype = torch.float) 
            simulation_labels = torch.tensor(node_labels[start_index:end_index,:], dtype = torch.float) 
            simulation_surface = torch.tensor(surf_bool[start_index:end_index])

            sampleData=Data(pos=simulation_positions,
                            x=simulation_features, 
                            y=simulation_labels,
                            surf = simulation_surface.bool()) 
            torchDataset.append(sampleData)
            start_index += nb_nodes_in_simulation
        
        return DataLoader(dataset=torchDataset,batch_size=1)

    def train(self, train_dataset, save_path=None):
        train_dataset = self.process_dataset(dataset=train_dataset,training=True)
        print("Start training")
        model = global_train(self.device, train_dataset, self.model, self.hparams,criterion = 'MSE_weighted')
        print("Training done")

    def predict(self,dataset,**kwargs):
        print(dataset)
        test_dataset = self.process_dataset(dataset=dataset,training=False)
        self.model.eval()
        avg_loss_per_var = np.zeros(4)
        avg_loss = 0
        avg_loss_surf_var = np.zeros(4)
        avg_loss_vol_var = np.zeros(4)
        avg_loss_surf = 0
        avg_loss_vol = 0
        iterNum = 0

        predictions=[]
        with torch.no_grad():
            for data in test_dataset:        
                data_clone = data.clone()
                data_clone = data_clone.to(self.device)
                out = self.model(data_clone.x)

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

                out = out.cpu().data.numpy()
                prediction = self._post_process(out)
                predictions.append(prediction)
        print("Results for test")
        print(avg_loss/iterNum, avg_loss_per_var/iterNum, avg_loss_surf_var/iterNum, avg_loss_vol_var/iterNum, avg_loss_surf/iterNum, avg_loss_vol/iterNum)
        predictions= np.vstack(predictions)
        predictions = dataset.reconstruct_output(predictions)
        return predictions

    def _post_process(self, data):
        try:
            processed = self.scaler.inverse_transform(data)
        except TypeError:
            processed = self.scaler.inverse_transform(data.cpu())
        return processed


def global_train(device, train_dataset, network, hparams, criterion = 'MSE', reg = 1):
    model = network.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = hparams['lr'],
            total_steps = (len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
        )
    start = time.time()

    train_loss_surf_list = []
    train_loss_vol_list = []
    loss_surf_var_list = []
    loss_vol_var_list = []

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

            data_sampled.pos = data_sampled.pos[idx]
            data_sampled.x = data_sampled.x[idx]
            data_sampled.y = data_sampled.y[idx]
            data_sampled.surf = data_sampled.surf[idx]
            train_dataset_sampled.append(data_sampled)
        train_loader = DataLoader(train_dataset_sampled, batch_size = hparams['batch_size'], shuffle = True)
        del(train_dataset_sampled)

        train_loss, _, loss_surf_var, loss_vol_var, loss_surf, loss_vol = train_model(device, model, train_loader, optimizer, lr_scheduler, criterion, reg = reg)        
        print('Train loss: ', train_loss)
        if criterion == 'MSE_weighted':
            train_loss = reg*loss_surf + loss_vol
        del(train_loader)

        train_loss_surf_list.append(loss_surf)
        train_loss_vol_list.append(loss_vol)
        loss_surf_var_list.append(loss_surf_var)
        loss_vol_var_list.append(loss_vol_var)

    loss_surf_var_list = np.array(loss_surf_var_list)
    loss_vol_var_list = np.array(loss_vol_var_list)

    return model

def train_model(device, model, train_loader, optimizer, scheduler, criterion = 'MSE', reg = 1):
    model.train()
    avg_loss_per_var = torch.zeros(4, device = device)
    avg_loss = 0
    avg_loss_surf_var = torch.zeros(4, device = device)
    avg_loss_vol_var = torch.zeros(4, device = device)
    avg_loss_surf = 0
    avg_loss_vol = 0
    iterNum = 0
    
    for data in train_loader:
        data_clone = data.clone()
        data_clone = data_clone.to(device)   
        optimizer.zero_grad()  
        out = model(data_clone.x)
        targets = data_clone.y

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

    return avg_loss.cpu().data.numpy()/iterNum, avg_loss_per_var.cpu().data.numpy()/iterNum, avg_loss_surf_var.cpu().data.numpy()/iterNum, avg_loss_vol_var.cpu().data.numpy()/iterNum, \
            avg_loss_surf.cpu().data.numpy()/iterNum, avg_loss_vol.cpu().data.numpy()/iterNum
