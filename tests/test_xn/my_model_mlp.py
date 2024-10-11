import torch
import torch.nn as nn
from torch_geometric.nn import  MessagePassing
from torch_geometric.data import Data, Batch
from torch import Tensor
from torch_geometric.utils import add_self_loops
from torch.nn import Conv1d, ReLU, MaxPool1d, ReLU, Dropout

class MyMlp(nn.Module):
    def __init__(self, in_dim, out_dim, device, hidden_layers: list = [64, 64], dropout: float = 0.15):
        super().__init__()
        self.in_dim = in_dim
        self.device = device
        self.out_dim = out_dim
        self.model = nn.Sequential()
        self.model.add_module("conv1", Conv1d(in_channels=in_dim, out_channels=hidden_layers[0], kernel_size=1))
        self.model.add_module("relu1", ReLU())
        self.model.add_module("bn1", nn.BatchNorm1d(hidden_layers[0]))
        
        for i in range(1, len(hidden_layers)):
            self.model.add_module(f"conv{i+1}", Conv1d(in_channels=hidden_layers[i-1], out_channels=hidden_layers[i], kernel_size=1))
            self.model.add_module(f"relu{i+1}", ReLU())
            self.model.add_module(f"bn{i+1}", nn.BatchNorm1d(hidden_layers[i]))
            self.model.add_module(f"dropout{i+1}", Dropout(dropout))
            
        if len(hidden_layers) > 0:
            self.model.add_module("conv_last", Conv1d(in_channels=hidden_layers[-1], out_channels=out_dim, kernel_size=1))
            self.model.add_module("relu_last", ReLU())
            self.model.add_module("bn_last", nn.BatchNorm1d(out_dim))
        

    def forward(self, x) -> Tensor:
        """Process the input graph

        Args:
            data (Data): The input graph

        Returns:
            Tensor: The output features
        """
         # Extract the features (data.x) from the input Data object
        # Reshape x to [batch_size=1, in_channels=4, seq_length=2678]
        x = x.unsqueeze(0).permute(0, 2, 1)
        
        return self.model(x).squeeze(0).T
        
        