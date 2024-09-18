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

from torch_geometric.transforms import BaseTransform
from scipy.spatial import Delaunay

from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

import matplotlib.pyplot as plt

class DelaunayTransform(BaseTransform):
    def __init__(self, dim=2):
        """
        Initialize the DelaunayTransform.

        Args:
            dim (int): Dimensionality of the points (default is 2 for 2D).
        """
        self.dim = dim

    def __call__(self, data: Data) -> Data:
        """
        Apply Delaunay triangulation to the node coordinates to construct edge_index.

        Args:
            data (Data): PyTorch Geometric Data object with 'x' attribute.

        Returns:
            Data: Updated Data object with 'edge_index' constructed via Delaunay triangulation.
        """            
        # Convert node features to NumPy array
        points = data.pos

        # Perform Delaunay triangulation
        tri = Delaunay(points)
        # Extract edges from the simplices
        edges = set()
        for simplex in tri.simplices:
            # Each simplex is a triangle represented by three vertex indices
            edges.add(tuple(sorted([simplex[0], simplex[1]])))
            edges.add(tuple(sorted([simplex[0], simplex[2]])))
            edges.add(tuple(sorted([simplex[1], simplex[2]])))

        # Convert set of edges to a list
        edge_index = np.array(list(edges)).T  # Shape: (2, num_edges)

        # Convert edge_index to torch tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Optionally, you can compute edge attributes here (e.g., Euclidean distances)
        # For example:
        # edge_attr = torch.norm(data.x[edge_index[0]] - data.x[edge_index[1]], dim=1, keepdim=True)
        # data.edge_attr = edge_attr

        # Update the Data object
        data.edge_index = edge_index
        data.edge_attr = np.zeros((edge_index.shape[1], 1))

        return data


class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='mean')

        # MLP initialization
        # Now, the input to the MLP is the node feature dimension (in_channels)
        # plus the positional information (3D positions).
        self.mlp = Sequential(
            Linear(in_channels + 2, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self,
                h: Tensor,
                pos: Tensor,
                edge_index: Tensor) -> Tensor:
        # Propagate both node features and positions
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self,
                h_j: Tensor,
                pos_j: Tensor,
                pos_i: Tensor) -> Tensor:
        # Combine node features with relative positional information
        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)

class PointNet(torch.nn.Module):
    def __init__(self, device, in_channels: int, num_attributes: int):
        super().__init__()

        self.device = device
        self.num_attributes = num_attributes

        # PointNet layers (in_channels is now the dimension of node features)
        self.conv1 = PointNetLayer(in_channels, 128).to(self.device)
        self.conv2 = PointNetLayer(128, 256).to(self.device)
        self.conv3 = PointNetLayer(256, 1024).to(self.device)
        self.conv4 = PointNetLayer(1024, 1024).to(self.device)

        # MLP for per-node attribute prediction
        self.mlp = Sequential(
            Dropout(0.2),
            Linear(1024, 1024),
            Dropout(0.2),
            ReLU(),
            Linear(1024, 512),
            Dropout(0.2),
            ReLU(),
            Dropout(0.2),
            Linear(512, num_attributes)  # Output number of attributes per node
        ).to(self.device)

    def forward(self,
                h: Tensor,  # Node features
                pos: Tensor,  # Node positions
                edge_index: Tensor) -> Tensor:

        # Message passing over two layers (now with node features)
        h = self.conv1(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv3(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv4(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        # Predict attributes for each node
        h = self.mlp(h)
        return h  # Output shape: [num_nodes, num_attributes]
            

class BasicSimulator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "AirfRANSSubmission"
        use_cuda = torch.cuda.is_available()
        self.device = 'cuda:0' if use_cuda else 'cpu'
        self.model = PointNet(self.device, 5, 4)
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
        print("Transform done")
        
        transform = DelaunayTransform() 

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
            if i <= 103 or not training:
                sampleData = transform(sampleData)
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
                out = self.model(data_clone.x, data_clone.pos, data_clone.edge_index)

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

def global_train(device, train_dataset: DataLoader, network: PointNet, hparams: dict, criterion='MSE', reg=1, local=False):
    model = network.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
    )

    train_loss_surf_list = []
    train_loss_vol_list = []
    loss_surf_var_list = []
    loss_vol_var_list = []
    train_loss_list = []
    
    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    epoch_nb = 0

    for epoch in pbar_train:
        epoch_nb += 1
        print('Epoch: ', epoch_nb)

        train_loss, _, loss_surf_var, loss_vol_var, loss_surf, loss_vol = train_model(device, model, train_dataset, optimizer, lr_scheduler, criterion, reg=reg)
        if criterion == 'MSE_weighted':
            train_loss = reg * loss_surf + loss_vol
        train_loss_list.append(train_loss)
        train_loss_surf_list.append(loss_surf)
        train_loss_vol_list.append(loss_vol)
        loss_surf_var_list.append(loss_surf_var)
        loss_vol_var_list.append(loss_vol_var)

        f = open("losses.txt", "a")
        f.write("Epoch: " + str(epoch) + " loss: " + str(train_loss) + " loss_surf: " + str(loss_surf) + " loss_vol: " + str(loss_vol) + "\n")
        f.close()
        
    loss_surf_var_list = np.array(loss_surf_var_list)
    loss_vol_var_list = np.array(loss_vol_var_list)

    # Plotting the loss
    plt.plot(train_loss_list, label='Total Loss')
    plt.savefig('total_loss.png')
    
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
        out = model(data_clone.x, data_clone.pos, data_clone.edge_index)
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
