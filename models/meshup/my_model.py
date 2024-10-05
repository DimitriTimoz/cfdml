import torch
import torch.nn as nn
from torch_geometric.nn import  MessagePassing
from torch_geometric.data import Data, Batch
from torch import Tensor
from torch_geometric.utils import add_self_loops
import pyvista as pv
import numpy as np
def plot_graph(data, l=1, plotter=None, node_colors=None):
    
    c = ['r', 'g', 'b', 'm']
    
    p = pv.Plotter() if plotter is None else plotter
    
    # Création d'un PolyData pour les points (nœuds)
    mesh = pv.PolyData()
    
    # Gestion des dimensions des positions
    if data.pos.shape[1] != 3:
        # Ajouter une troisième dimension si nécessaire
        mesh.points = np.concatenate([data.pos.cpu().numpy(), np.full((data.pos.shape[0], 1), l)], axis=1)
    else:
        mesh.points = data.pos.cpu().numpy()

    # Création des lignes pour les arêtes
    edges = data.edge_index.t().cpu().numpy()
    lines = np.hstack([np.full((edges.shape[0], 1), 2), edges]).ravel()
    mesh.lines = lines
    
    
    # Ajout des couleurs au PolyData
    mesh.point_data['values'] = node_colors if node_colors is not None else np.random.randint(0, 255, size=(data.pos.shape[0], 3))
    
    # Ajouter le mesh avec les couleurs des nœuds
    p.add_mesh(mesh, scalars='values', line_width=0.5, point_size=0.3, render_points_as_spheres=True)

    # Si aucun plotter n'a été fourni, on montre la figure
    if plotter is None:
        p.show()
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
        ).to(device)
        
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
        ).to(device)
        
    def forward(self, data: Data) -> Tensor:
        return self.model(data.x)

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super().__init__()
        self.in_dim = in_dim
        self.device = device
        self.out_dim = out_dim
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        ).to(device)
        
    def forward(self, node_embedding: Tensor) -> Tensor:
        """Decode the nodes and edges embeddings into the output features

        Args:
            node_embedding (Tensor): The node embeddings
             (Tensor): The edge embeddings

        Returns:
            Tensor: The output features
        """
        return self.model(node_embedding)
    
    
#class Processor(nn.Module):
#    def __init__(self, in_dim, out_dim, device):
#        super().__init__()
#        self.in_dim = in_dim
#        self.device = device
#        self.out_dim = out_dim
#        self.edge_model = nn.Sequential(
#            nn.Linear(128*3, 512),
#            nn.ReLU(),
#            nn.Linear(512, 128)
#        ).to(device)    
#        
#        self.node_model = nn.Sequential(
#            nn.Linear(128*2, 512),
#            nn.ReLU(),
#            nn.Linear(512, 128)
#        ).to(device)
#            
#    
#    def edge_processor(self, edge_embedding: Tensor, nodes_from_edge_embedding: Tensor) -> Tensor:
#        """
#        e_ij^{r,k,l+} = f_e(e_ij^{r,k,l}, v_i^{r,k,l}, v_j^{r,k,l})
#        Args:
#            edge_embedding (Tensor(N, 128)): Edge embeddings of the subgraph
#            nodes_from_edge_embedding (Tensor(N, 128*2)): The nodes embeddings from the edge embeddings
#
#        Returns:
#            Tensor(N, 128): The processed edge embeddings of the subgraph 
#        """
#        return self.edge_model(torch.cat((edge_embedding, nodes_from_edge_embedding), dim=1))
#    
#    def forward(self, node_embeddings: Tensor, edge_embeddings_sg: Tensor, edge_indices: Tensor, node_indices_sg: Tensor) -> Tensor:
#        """Process the input graph
#
#        Args:
#            node_embedding (Tensor(N, 128)): The node embeddings of whole graph
#            edge_embeddings_sg (Tensor(N, 128)): The edge embeddings of the subgraph
#            edge_indices (Tensor(N, 2)): The edge indices of the subgraph indexed by the whole graph
#            node_indices_sg (Tensor(N)): The node indices of the subgraph indexed by the whole graph
#
#        Returns:
#            Tensor: The processed node embeddings
#        """
#        edge_embedding = self.edge_processor(edge_embeddings_sg, node_embeddings[edge_indices].reshape(-1, 256))
#        # Sum of the edge embeddings from the same node 
#        return self.node_model(torch.stack((node_embeddings[node_indices_sg], edge_embedding[], ), dim=1))
    

class Processor(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')  # or 'add', 'max', etc.
        self.edge_mlp = nn.Sequential(
            nn.Linear((2 * in_dim) + out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: Node features [N, in_dim]
        # edge_attr: Edge features [E, out_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    # Computes messages sent from source nodes to target nodes along edges
    # Updates edge features based on the current edge features and the features of connected nodes.
    def message(self, x_i, x_j, edge_attr):
        """Computes messages sent from source nodes to target nodes along edges

        Args:
            x_i (_type_): Features of target nodes, shape [E, in_dim]
            x_j (_type_): Features of source nodes, shape [E, in_dim]
            edge_attr (_type_): Current edge features, shape [E, out_dim]

        Returns:
            Tensor(E, out_dim): Returns the updated edge features of shape [E,out_dim], which are the messages to be aggregated at the target nodes.
        """
        # x_i: Features of target nodes [E, in_dim]
        # x_j: Features of source nodes [E, in_dim]
        # edge_attr: Edge features [E, out_dim]
        msg = torch.cat([x_i, x_j, edge_attr], dim=1)  # [E, 2*in_dim + out_dim]
        return self.edge_mlp(msg)  # Output: Updated edge features [E, out_dim]

    # Updates node features based on the original node features and the aggregated messages
    def update(self, aggr_out, x):
        """Updates node features in the graph

        Args:
            aggr_out (_type_): Aggregated messages at each node, shape [N,out_dim]
            x (_type_): Original node features, shape [N, in_dim]

        Returns:
            _type_: Returns the updated node features, shape [N, out_dim]
        """
        # aggr_out: Aggregated messages at each node [N, out_dim]
        # x: Original node features [N, in_dim]
        out = torch.cat([x, aggr_out], dim=1)  # [N, in_dim + out_dim]
        return self.node_mlp(out)  # Output: Updated node features [N, out_dim]


class UaMgnn(nn.Module):
    def __init__(self, in_dim, out_dim, R, K, device):
        super().__init__()
        self.in_dim = in_dim
        self.device = device
        self.out_dim = out_dim
        self.R = R
        self.K = K
        self.node_encoder = NodeEncoder(in_dim, 128, self.device)
        
        self.up_sampling_edge_encoder = torch.nn.ModuleList([EdgeEncoder(3, 128, self.device) for _ in range(R)])
        self.subgraph_edge_encoders =  torch.nn.ModuleList([torch.nn.ModuleList([EdgeEncoder(3, 128, self.device) for _ in range(K)]) for _ in range(R)])

        self.node_decoder = Decoder(128, out_dim, device)
        self.processors = torch.nn.ModuleList([torch.nn.ModuleList([Processor(128, 128) for _ in range(K)]) for _ in range(R)]).to(device)
        
        
    def forward(self, data: Data) -> Tensor:
        """Process the input graph

        Args:
            data (Data): The input graph

        Returns:
            Tensor: The output features
        """
        
        # Compute the default edge attributes TODO: post_processing
        edge_directions = data.pos[data.edge_index[1]][:, :2] - data.pos[data.edge_index[0]][:, :2]
        edge_norms = torch.norm(edge_directions, dim=1, keepdim=True)
        edge_directions = edge_directions / edge_norms
        edges_attr = torch.cat([edge_directions, edge_norms], dim=1)

        
        # Initiate {v^r}_i by node encoder for 1 ≤ 𝑟 ≤ 𝑅;
        node_embedding = self.node_encoder(data) # (N, 128)
        edge_embedding = torch.zeros((data.edge_index.shape[1], 128), device=self.device) # (E, 128)
    
        for r in range(self.R): # the 𝑟-th mesh graph
            ir = self.R - r - 1 
            # We get the range of the edges in the layer r
            #r_edge_range = data.layer_ranges[ir]
            r_clusters_edge_indicies = data.clusters[ir*self.K:(ir+1)*self.K]
            nodes_of_r = torch.cat(data.node_clusters[ir*self.K:(ir+1)*self.K])
            new_node_embedding_r = []
            for k in range(self.K): # the 𝑘-th subgraph
                ik = self.K - k - 1
                rk_cluster_edge_indices = r_clusters_edge_indicies[ik]
                edges_attr_rk_cluster = edges_attr[rk_cluster_edge_indices]
                # Initiate e_{i,j}^{r,k} by edge encoder; 
                edge_embedding[rk_cluster_edge_indices] = self.subgraph_edge_encoders[ir][ik](edges_attr_rk_cluster)
                # v_{i}^{r,k,0} <- v_{i}^{r};
                n_mp_lk = data.edge_frequencies[ir][ik] # the 𝑙-th MP step
                
                # Get the nodes indices of the subgraph indexed by the whole graph
                nodes_of_rk = data.node_clusters[ir*self.K + ik] # (N_rk)
                # Get the nodes embeddings of the subgraph from the whole list of nodes embeddings
                node_embedding_rk = node_embedding[nodes_of_rk] # (N_rk, 128)
                # Get the edge embeddings of the subgraph from the whole list of edge embeddings
                edge_embedding_rk = edge_embedding[rk_cluster_edge_indices] # (E_rk, 128)
                
                # Get the edge index of the subgraph from the whole list of edge index
                edge_index_rk = data.edge_index[:, rk_cluster_edge_indices] # (2, E_k) mais avec des indexs dans le graph complet
                # Remove the minimum value from the edge index to have the correct index
                edge_index_rk -= torch.min(edge_index_rk)
                max_ = torch.max(edge_index_rk)
                max_ = max_.cpu()
                print(max_, flush=True)
                mask = torch.full((torch.max(edge_index_rk)+1,), 1, dtype=torch.int32, device=edge_index_rk.device)
                mask[nodes_of_rk-torch.min(nodes_of_r)] = 0    
                cumsum = torch.cumsum(mask, dim=0)
                edge_index_rk -= cumsum[edge_index_rk]-1 # TODO: fix -1 should not be here
                
                for l in range(n_mp_lk): # the 𝑙-th MP step
                    # Sep l of message passing between nodes and edges of the same k,𝑟-th mesh graph
                    #node_embedding_rk = (node_embedding, edge_embedding[rk_cluster_edge_indices], data.edge_index[:, rk_cluster_edge_indices])
                    print(node_embedding_rk.shape, edge_index_rk.shape, edge_embedding_rk.shape, flush=True)
                    print(edge_index_rk.min(), edge_index_rk.max(), flush=True)
                    plot_graph(Data(pos=data.pos[nodes_of_rk], edge_index=edge_index_rk), l, node_colors=node_embedding_rk[:, 0].detach().cpu().numpy())
                    node_embedding_rk = self.processors[ir][ik](node_embedding_rk, edge_index_rk, edge_embedding_rk)
                    print("one done")
            new_node_embedding_r.append(node_embedding_rk)
            # Aggregate the node node_embedding_r
            if n_mp_lk > 0:
                new_node_embedding_r = torch.stack(new_node_embedding_r)
                print(node_embedding.shape)
                node_embedding[nodes_of_r] = torch.mean(new_node_embedding_r, dim=0)
            print("one k done")
        return self.node_decoder(node_embedding)
        