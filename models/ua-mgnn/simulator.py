import time
import datetime as dt

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from utils import *
from my_model import *

from lips.dataset.scaler.standard_scaler_iterative import StandardScalerIterative


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

        self.model = UaMgnn(5, 4, R=self.hparams["R"], K=self.hparams["K"], device=self.device)

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
        transform = DelaunayTransform()

        for nb_nodes_in_simulation in nb_nodes_in_simulations:
            #still alive?
            if dt.datetime.now() - t > dt.timedelta(seconds=60):
                print("Processing still alive - index : ", end_index)
                t = dt.datetime.now()
            end_index = start_index+nb_nodes_in_simulation
            simulation_positions = torch.tensor(position[start_index:end_index,:], dtype = torch.float) 
            simulation_features = torch.tensor(nodes_features[start_index:end_index, 2:], dtype = torch.float) # Ignore positions
            simulation_labels = torch.tensor(node_labels[start_index:end_index,:], dtype = torch.float) 
            simulation_surface = torch.tensor(surf_bool[start_index:end_index])

            sampleData=transform(Data(pos=simulation_positions,
                            x=simulation_features, 
                            y=simulation_labels,
                            surf = simulation_surface.bool())) 
            sampleData = generate_coarse_graphs(sampleData, self.hparams["R"], self.hparams["K"], factor=self.hparams["factor"], range_=self.hparams["range"])
            torchDataset.append(sampleData)
            start_index += nb_nodes_in_simulation
        
        return DataLoader(dataset=torchDataset,batch_size=1)

    def train(self,train_dataset, save_path=None):
        train_dataset = self.process_dataset(dataset=train_dataset,training=True)
        print("Start training")
        model = global_train(self.device, train_dataset, self.model, self.hparams, criterion = 'MSE')
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
                data = data.to(self.device)
                out = self.model(data)

                targets = data.y
                loss_criterion = nn.MSELoss(reduction = 'none')

                loss_per_var = loss_criterion(out, targets).mean(dim = 0)
                loss = loss_per_var.mean()
                loss_surf_var = loss_criterion(out[data.surf, :], targets[data.surf, :]).mean(dim = 0)
                loss_vol_var = loss_criterion(out[~data.surf, :], targets[~data.surf, :]).mean(dim = 0)
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


def global_train(device, train_dataset, network, hparams, criterion = 'MSE', reg = 0.5):
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
        train_loss, _, loss_surf_var, loss_vol_var, loss_surf, loss_vol = train_model(device, model, train_dataset, optimizer, lr_scheduler, criterion, reg = reg)        
        if criterion == 'MSE_weighted':
            train_loss = reg*loss_surf + loss_vol

        train_loss_surf_list.append(loss_surf)
        train_loss_vol_list.append(loss_vol)
        loss_surf_var_list.append(loss_surf_var)
        loss_vol_var_list.append(loss_vol_var)
        print('Train loss: ', train_loss)
    loss_surf_var_list = np.array(loss_surf_var_list)
    loss_vol_var_list = np.array(loss_vol_var_list)

    return model

def train_model(device, model, train_loader, optimizer, scheduler, criterion = 'MSE', reg = 0.5):
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
        out = model(data_clone)
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
        if iterNum % 50 == 0:
            print('Loss per var: ', loss_per_var,'loss: ', total_loss, "loss_surf: ", loss_surf, "loss_vol: ", loss_vol)

    print("Loss per var: ", avg_loss_per_var.cpu().data.numpy()/iterNum, iterNum)
    return avg_loss.cpu().data.numpy()/iterNum, avg_loss_per_var.cpu().data.numpy()/iterNum, avg_loss_surf_var.cpu().data.numpy()/iterNum, avg_loss_vol_var.cpu().data.numpy()/iterNum, \
            avg_loss_surf.cpu().data.numpy()/iterNum, avg_loss_vol.cpu().data.numpy()/iterNum
            