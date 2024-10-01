# %%
#!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
#!pip install torch_geometric
#!pip install pyvista
#%pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.1+cpu.html

# %%
import torch
import matplotlib.pyplot as plt
from utils import DelaunayTransform
from torch_geometric.data import Data
import networkx as nx
import pyvista as pv
import numpy as np
torch.__version__

# %%
import random
N = 50_000
pos = torch.rand((N, 2))
data = Data(pos=pos, surf=torch.full((N, 1), False))
transform = DelaunayTransform()
data = transform(data)
data.pos = pos

#data = torch.load('./sampleData.pth')

def plot_graph(data, l=1, plotter=None):
    c = ['r', 'g', 'b', 'm']
    
    p = pv.Plotter() if plotter is None else plotter
    
    mesh = pv.PolyData()
    if data.pos.shape[1] != 3:
        mesh.points = np.concatenate([data.pos.numpy(), np.full((data.pos.shape[0], 1), l)], axis=1) 
    else:
        mesh.points = data.pos.numpy()
    edges = data.edge_index.t().numpy()
    lines = np.hstack([np.full((edges.shape[0], 1), 2), edges]).ravel()
    mesh.lines = lines
    p.add_mesh(mesh, line_width=1, color=random.choice(c))
    
    if plotter is None:
        p.show()

# %%
#plot_graph(data)

# %%
import time

def divide_mesh(v, e, k):
    clusters = [Data(edge_ids=set()) for _ in range(k)]
    
    # Randomly initialize centroids (2D points)
    centroids = torch.rand((k, 2), device=v.device)

    # Precompute edge directions and norms
    edges_directions = v[e[:, 1]] - v[e[:, 0]]
    edges_norms = torch.norm(edges_directions, dim=1, keepdim=True)  # Shape: [num_edges, 1]
    edges_directions /= edges_norms  # Normalize edge directions
    start_all = time.time()
    norm_changes = float('inf')
    while norm_changes > 1e-3:
        # Vectorized clustering step
        centroids_norms = torch.norm(centroids, dim=1, keepdim=True)  # Shape: [num_centroids, 1]
        cosine_angles = torch.matmul(edges_directions, centroids.T) / (centroids_norms.T)  # Shape: [num_edges, num_centroids]
        angles = torch.acos(cosine_angles)  # Ensure values are in valid range for acos
        min_edge_idxs = torch.argmin(angles, dim=1)  # Shape: [num_edges]
        # Efficient assignment to clusters using torch
        cluster_masks = [(min_edge_idxs == i) for i in range(k)]
        for i in range(k):
            clusters[i].edge_ids.update(torch.nonzero(cluster_masks[i]).squeeze(1).tolist())

        # Efficient centroid update
        n_m = 0.0
        for i in range(k):
            if clusters[i].edge_ids:  # Check if the cluster has assigned edges
                cluster_edges = edges_directions[torch.tensor(list(clusters[i].edge_ids), device=v.device)]
                last_centroid = centroids[i].clone()
                centroids[i] = torch.mean(cluster_edges, dim=0)
                n_m = max(torch.norm(centroids[i] - last_centroid), n_m)
        norm_changes = n_m

    print("All:", time.time() - start_all)

    # Post-process clusters to finalize edge indices
    for cluster in clusters:
        cluster.edge_index = e[list(cluster.edge_ids)]
        del cluster.edge_ids

    return clusters


device = torch.device('cpu')
data.pos = data.pos[:, :2].to(device) 
clusters = divide_mesh(data.pos, data.edge_index.T, 8)
clusters

# %%
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.cluster import MiniBatchKMeans

def generate_coarse_graph_with_clustering(g, num_clusters):
    """
    Generates a coarser graph from a given graph using MiniBatchKMeans clustering
    to reduce the number of nodes. Returns the coarse graph and the mapping from
    original nodes to clusters.

    :param g: Original graph (of type torch_geometric Data).
    :param num_clusters: The number of clusters or "super-nodes" to create.
    :return: A tuple (coarse_graph, labels) where labels map original nodes to clusters.
    """
    pos = g.pos.cpu().numpy()  
    edges = g.edge_index.cpu().numpy() 

    # Use MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=10000, random_state=0).fit(pos)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Create new edges by connecting clusters if a link exists between two nodes in the original graph
    new_edges = set()
    for edge in edges.T:
        cluster_u = labels[edge[0]]
        cluster_v = labels[edge[1]]
        if cluster_u != cluster_v:
            new_edges.add((min(cluster_u, cluster_v), max(cluster_u, cluster_v)))

    # Convert new edges to torch tensor
    if new_edges:
        new_edges = np.array(list(new_edges)).T
        new_edges = torch.tensor(new_edges, dtype=torch.long)
    else:
        new_edges = torch.empty((2, 0), dtype=torch.long)
    
    # Convert cluster centers to tensor
    new_pos = torch.tensor(cluster_centers, dtype=torch.float)

    # Create new coarse graph
    coarse_graph = Data(pos=new_pos, edge_index=new_edges)
    
    return coarse_graph, labels  # Return the mapping

# Now, generate the hierarchical mesh
graphs = [data]  # List of graphs, starting with the original graph
mappings = []    # List of mappings from nodes to clusters at each level

current_graph = data
num_levels = 3  # Define the number of levels you want
for level in range(num_levels):
    num_clusters = int(len(current_graph.pos) // 10)
    print(f"Generating coarse graph at level {level+1} with {num_clusters} clusters...")
    coarse_graph, mapping = generate_coarse_graph_with_clustering(current_graph, num_clusters)
    graphs.append(coarse_graph)
    mappings.append(mapping)
    current_graph = coarse_graph

# Now, construct the final multi-layer graph
all_positions = []
all_edges = []
node_offsets = []  # To keep track of node index offsets for each layer

offset = 0
z_offset = 1.0  # Vertical offset between layers
for layer_index, graph in enumerate(graphs):
    num_nodes = graph.pos.shape[0]
    pos = graph.pos

    # Adjust positions to add vertical dimension
    if pos.shape[1] == 2:
        # If positions are 2D, add a z-coordinate
        z = torch.full((num_nodes, 1), layer_index * z_offset)
        pos = torch.cat((pos, z), dim=1)
    elif pos.shape[1] == 3:
        # If positions are 3D, adjust the z-coordinate
        pos[:, 2] += layer_index * z_offset
    else:
        raise ValueError("Position tensor has unsupported number of dimensions.")

    all_positions.append(pos)
    node_offsets.append(offset)
    # Adjust edges
    if graph.edge_index.numel() > 0:
        adjusted_edges = graph.edge_index + offset
        all_edges.append(adjusted_edges)
    offset += num_nodes

# Create inter-layer edges based on the mappings and deleted edges
for level, mapping in enumerate(mappings):
    # Nodes in layer n-1
    start_idx = node_offsets[level]
    end_idx = node_offsets[level] + len(mapping)
    # Nodes in layer n
    next_layer_start_idx = node_offsets[level+1]
    # Map nodes from layer n-1 to layer n
    node_indices = np.arange(start_idx, end_idx)
    cluster_indices = mapping + next_layer_start_idx
    inter_layer_edges = np.vstack((node_indices, cluster_indices))

    # Add inter-layer edges to all_edges
    all_edges.append(torch.tensor(inter_layer_edges, dtype=torch.long))

    # Include deleted edges (edges between nodes merged into the same cluster)
    # Identify edges in layer n-1 that were merged in layer n
    edge_indices = graphs[level].edge_index.numpy()
    for edge in edge_indices.T:
        # edge[0] and edge[1] are local node indices (starting from 0)
        if mapping[edge[0]] == mapping[edge[1]]:
            # These nodes were merged; create an edge to the cluster center in layer n
            cluster_idx = mapping[edge[0]] + next_layer_start_idx
            # Adjust node indices by adding start_idx to align with global indices
            all_edges.append(torch.tensor([[edge[0] + start_idx, cluster_idx],
                                           [edge[1] + start_idx, cluster_idx]], dtype=torch.long))

# Combine all positions and edges
all_positions = torch.cat(all_positions, dim=0)
all_edges = torch.cat(all_edges, dim=1)

# Create final graph
final_graph = Data(pos=all_positions, edge_index=all_edges)

# Now you can plot the final graph
p = pv.Plotter()
print(final_graph)
plot_graph(final_graph, 1, p)
p.show()



