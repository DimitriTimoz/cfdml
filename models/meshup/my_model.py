import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, Batch
from torch import Tensor
import torch.nn.functional as F

# 
# ENCODER ::::
# Pour chaque edge on encode une représentation de l'edge à partir de son vecteur directeur et sa norme dans un espace de taille 128, à vérifier ?
# Pour chaque node on encode une représentation du note à partir de ses features initiales et limites vers un espace de taille 128
# PROCESSOR ::::
# Pour chaque graphe Gr:
#   Pour chaque Gr,k dans Gr:
#      On propage Lk fois les features des nodes et des edges
#
# # Une fois ?
# Pour chaque Graphe Gr de 2 à R 
#   On sélection Gr-1
#   On propage Gr-1 vers Gr
#
#
# DECODER ::::
# Pour chaque node on décode une représentation du node à partir de ses features et des features des edges dans un espace de la taille des features à prédire
#


class GraphSAGE(torch.nn.Module):
    def __init__(self, device, in_dim, hidden_dims: list, out_dim, dropout=0.2):
        super().__init__()
        self.device = device
        self.dropout = dropout
        hidden_dims.append(out_dim)
        self.convs = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.convs.append(SAGEConv(in_dim, hidden_dim).to(self.device))
            in_dim = hidden_dim
    
    def forward(self, data: Data) -> Tensor: 
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            
        x = self.convs[-1](x, edge_index)
        return x

class JSP(torch.nn.Module):
    
    def __init__(self, in_dim, hidden_dims: list, out_dim=128, device = 'cpu'):
        super().__init__()
        self.in_dim = in_dim
        self.device = device
        self.out_dim = out_dim
        self.model = GraphSAGE(self.device, self.in_dim, hidden_dims, self.out_dim)

    def forward(self, data: Data) -> Tensor:
        edge_ranges_to_exclude = data.up_scale_edge_ranges
        mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)

        # Exclure les slices définies dans `slices`
        for start, end in edge_ranges_to_exclude:
            mask[start:end] = False  # Mettre à False les éléments à exclure

        edges = data.edge_index[:, mask]
        d = Data(x=data.x, edge_index=edges, device=self.device)
        return self.model(d)

class EdgeEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, device = 'cpu'):
        super().__init__()
        self.in_dim = in_dim
        self.device = device
        self.out_dim = out_dim
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        
    def forward(self, data: Data) -> Tensor:
        return self.model(data)
    
class NodeEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, device = 'cpu'):
        super().__init__()
        self.in_dim = in_dim
        self.device = device
        self.out_dim = out_dim
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        
    def forward(self, data: Data) -> Tensor:
        return self.model(data)
    

class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super().__init__()
        self.in_dim = in_dim
        self.device = device
        self.out_dim = out_dim
        self.edge_encoder = EdgeEncoder(3, out_dim, self.device)
        self.node_encoder = NodeEncoder(in_dim, out_dim, self.device)
        
    def forward(self, data: Data) -> Tensor:
        """Encode the nodes and edges of the input

        Args:
            data (Data): The input graph 

        Returns:
            (Tensor, Tensor): The node and edge embeddings
        """
        edge_ranges_to_exclude = data.up_scale_edge_ranges
        mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)
  
        # Exclude the slices of edge connections between layers
        for start, end in edge_ranges_to_exclude:
            mask[start:end] = False 

        edges = data.edge_index[:, mask].to(self.device)

        edges_direction = data.pos[edges[0]][:, :2] - data.pos[edges[1]][:, :2]
        norm = torch.norm(edges_direction, dim=1, keepdim=True)
        edges_attr = torch.cat([edges_direction, norm], dim=1)

        # Run the encoders in parallel using torch.jit.fork
        edge_embedding_fut = torch.jit.fork(self.edge_encoder, edges_attr)
        node_embedding_fut = torch.jit.fork(self.node_encoder, data.x)
        
        # Wait for both futures to complete
        edge_embedding = torch.jit.wait(edge_embedding_fut)
        node_embedding = torch.jit.wait(node_embedding_fut)
        
        return node_embedding, edge_embedding 
    