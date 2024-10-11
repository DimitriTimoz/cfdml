import torch
import torch.nn as nn
from torch_geometric.nn import  MessagePassing
from torch_geometric.data import Data, Batch
from torch import Tensor
from torch_geometric.utils import add_self_loops

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

REDUCE_F = 1

class EdgeEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, device = 'cpu'):
        super().__init__()
        self.in_dim = in_dim
        self.device = device
        self.out_dim = out_dim
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128//REDUCE_F),
            nn.ReLU(),
            nn.Linear(128//REDUCE_F, out_dim)
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
            nn.Linear(in_dim, 128//REDUCE_F),
            nn.ReLU(),
            nn.Linear(128//REDUCE_F, out_dim)
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
            nn.Linear(in_dim, 128//REDUCE_F),
            nn.ReLU(),
            nn.Linear(128//REDUCE_F, out_dim)
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
            nn.Linear((2 * in_dim) + out_dim, 512//REDUCE_F),
            nn.ReLU(),
            nn.Linear(512//REDUCE_F, out_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, 512//REDUCE_F),
            nn.ReLU(),
            nn.Linear(512//REDUCE_F, out_dim)
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
        
        self.up_sampling_edge_encoder = torch.nn.ModuleList([EdgeEncoder(2, 128, self.device) for _ in range(R-1)])
        self.subgraph_edge_encoders =  torch.nn.ModuleList([torch.nn.ModuleList([EdgeEncoder(3, 128, self.device) for _ in range(K)]) for _ in range(R)])

        self.node_decoder = Decoder(128, out_dim, device)
        self.processors = torch.nn.ModuleList([torch.nn.ModuleList([Processor(128, 128) for _ in range(K)]) for _ in range(R)]).to(device)
        self.up_sampling_processors = torch.nn.ModuleList([Processor(128, 128) for _ in range(R-1)]).to(device)

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
        edge_norms[edge_norms == 0] = 1.0
        edges_attr = torch.cat([edge_directions / edge_norms, edge_norms], dim=1)

        # Initiate {v^r}_i by node encoder for 1 ≤ 𝑟 ≤ 𝑅;
        node_embedding = self.node_encoder(data) # (N, 128)
        edge_embedding = torch.zeros((data.edge_index.shape[1], 128), device=self.device) # (E, 128)
        for r in range(self.R): # the 𝑟-th mesh graph
            ir = self.R - r - 1 
            # We get the range of the edges in the layer r
            r_node_indices_range = data.layer_ranges[ir]
            nodes_embedding_r = node_embedding[r_node_indices_range[0]:r_node_indices_range[1]]
            new_node_embedding_r = [torch.zeros((nodes_embedding_r.shape[0], 128), device=self.device) for _ in range(self.K)]
            for k in range(self.K): # the 𝑘-th subgraph
                edge_indices_of_k = data.clusters[ir*self.K + k]
                edges_attr_rk_cluster = edges_attr[edge_indices_of_k]
                # Initiate e_{i,j}^{r,k} by edge encoder; 
                edge_embedding[edge_indices_of_k] = self.subgraph_edge_encoders[ir][k](edges_attr_rk_cluster)
                # v_{i}^{r,k,0} <- v_{i}^{r};
                n_mp_lk = data.edge_frequencies[ir][k] # the 𝑙-th MP step
                 
                nodes_of_k_in_r = data.node_clusters[ir*self.K + k] 
                node_embeddings_rk = nodes_embedding_r[nodes_of_k_in_r]
                
                edge_index_rk_in_r = data.edge_index[:, edge_indices_of_k] - r_node_indices_range[0]
                mask = torch.full((nodes_embedding_r.shape[0],), 1, dtype=torch.int32, device=self.device)
                mask[nodes_of_k_in_r] = 0
                
                mask = torch.cumsum(mask, dim=0)

                edge_index_of_k_in_k = edge_index_rk_in_r - mask[edge_index_rk_in_r]
                edge_embedding_rk = edge_embedding[edge_indices_of_k]
                for l in range(n_mp_lk): # the 𝑙-th MP step
                    # Sep l of message passing between nodes and edges of the same k,𝑟-th mesh graph
                    node_embeddings_rk = self.processors[ir][k](node_embeddings_rk, edge_index_of_k_in_k, edge_embedding_rk)
                new_node_embedding_r[k][nodes_of_k_in_r] = node_embeddings_rk
            # Aggregate the node node_embedding_r
            if n_mp_lk > 0:
                new_node_embedding_r = torch.stack(new_node_embedding_r)
                node_embedding[r_node_indices_range[0]:r_node_indices_range[1]] = torch.sum(new_node_embedding_r, dim=0)
            
            if r < self.R - 1:
                # We initiate e_{i,j}^{r,r+1} by edge encoder;
                up_scale_edge_range = data.up_scale_edge_ranges[ir-1]
                up_scale_edge_attributes = edge_directions[up_scale_edge_range[0]:up_scale_edge_range[1]] 
                up_scale_edge_embeddings = self.up_sampling_edge_encoder[r](up_scale_edge_attributes)

                # Update v_[j]^{r+1} by up-sampling on e_{i,j}^{r,r+1} and v_{i}^{r}
                pf_node_indices_range = data.layer_ranges[ir-1]
                mf_node_indices_range = data.layer_ranges[ir]
                nodes_embedding_up_sampling = node_embedding[pf_node_indices_range[0]:mf_node_indices_range[1]]
               
                up_scale_edge_index = data.edge_index[:, up_scale_edge_range[0]:up_scale_edge_range[1]] - pf_node_indices_range[0]
                mp_output = self.up_sampling_processors[r](nodes_embedding_up_sampling, up_scale_edge_index, up_scale_edge_embeddings)[:pf_node_indices_range[1]-pf_node_indices_range[0]] # Useless computation to fix
                node_embedding[pf_node_indices_range[0]:pf_node_indices_range[1]] = mp_output
        return self.node_decoder(node_embedding[:data.layer_ranges[0][1]])
        