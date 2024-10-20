import torch
from torch_cluster import grid_cluster
from torch_scatter import scatter, scatter_mean
import numpy as np

from scipy.spatial import Delaunay
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

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
        # Convert node features to NumPy arrays
        points = data.pos.cpu().numpy()
        surf = data.surf.cpu().numpy()

        # Perform Delaunay triangulation
        tri = Delaunay(points)

        # Check if all nodes in simplex are on the surface
        simplices_surf = surf[tri.simplices]
        all_on_surf = simplices_surf.all(axis=1)
        simplices_to_use = ~all_on_surf
        valid_simplices = tri.simplices[simplices_to_use]

        # Extract edges from valid simplices
        edges = np.concatenate([
            valid_simplices[:, [0, 1]],
            valid_simplices[:, [1, 2]],
            valid_simplices[:, [2, 0]],
            valid_simplices[:, [1, 0]],
            valid_simplices[:, [2, 1]],
            valid_simplices[:, [0, 2]]
        ], axis=0)

        edge_index = np.unique(edges, axis=0)

        # Convert edge_index to torch tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=data.pos.device)

        # Update the Data object
        data.edge_index = edge_index.t()
        return data


def divide_mesh(v: torch.Tensor, e: torch.Tensor, k: int, verbose=False):
    """Divide a mesh into k clusters of edges according to their direction.

    Args:
        v (Tensor(N, 2)): Positions of the vertices in the mesh.
        e (Tensor(N, 2)): Edge indices of the mesh.
        k (int): Number of clusters to divide the mesh into.

    Returns:
        clusters_edge_indices, nodes
    """
    # Initialize centroids as unit vectors in different directions
    angles = torch.linspace(0, 2 * np.pi, k, device=v.device)
    centroids = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)  # Shape: (k, 2)

    # Compute edge directions and normalize
    edges_directions = v[e[:, 1]] - v[e[:, 0]]
    edges_directions = edges_directions / edges_directions.norm(dim=1, keepdim=True)

    norm_changes = float('inf')
    while norm_changes > 1e-3:
        # Compute cosine similarity between edge directions and centroids
        cosine_sim = torch.matmul(edges_directions, centroids.T)

        # Assign edges to the centroid with the highest cosine similarity
        max_edge_idxs = torch.argmax(cosine_sim, dim=1)

        # Update centroids
        centroids_old = centroids.clone()
        centroids = scatter_mean(edges_directions, max_edge_idxs, dim=0, dim_size=k)
        # Handle clusters with no edges assigned
        zero_norms = centroids.norm(dim=1) == 0
        centroids[~zero_norms] = centroids[~zero_norms] / centroids[~zero_norms].norm(dim=1, keepdim=True)
        centroids[zero_norms] = centroids_old[zero_norms]

        # Compute maximum change in centroids
        norm_changes = torch.max((centroids - centroids_old).norm(dim=1))

    # Collect clusters
    clusters_edge_indices = [torch.where(max_edge_idxs == i)[0] for i in range(k)]
    nodes = [torch.unique(e[clusters_edge_indices[i]].view(-1)) for i in range(k)]

    if verbose:
        angles = torch.rad2deg(torch.atan2(centroids[:, 1], centroids[:, 0]))
        return clusters_edge_indices, nodes, centroids, angles
    else:
        return clusters_edge_indices, nodes


def generate_coarse_graph(data, r, clusters_per_layer):
    """Generate a coarse graph from a fine graph.

    Args:
        data (Data): The fine graph to coarsen.
        r (int): The coarsening factor.

    Returns:
        (Data, Tensor(N, 2), list(Tensor(C), Tensor(C)): The coarsened graph, the connection edge index, the clusters and the number of edges per coarse element of a subgraph.
    """
    #FIXME: use square grid 
    size = torch.max(data.pos, dim=0)[0] - torch.min(data.pos, dim=0)[0]

    size /= r
    
    # Assign each node to a grid cell id
    cluster = grid_cluster(data.pos, size) 
    
    # Get the indices of the unique clusters
    _, new_index, counts = torch.unique(cluster, return_inverse=True, return_counts=True)
    # Gather each node to its cluster and compute the mean for position features
    out_positions = scatter(data.pos, new_index, dim=0, reduce='mean')
    
    out_x = scatter(data.x, new_index, dim=0, reduce='mean')
    
    # Interpolate the other features accordingly to the position    
    surf = scatter(data.surf.to(torch.int), new_index, reduce='max')
    connection_edge_index = torch.stack([new_index+data.pos.shape[0], torch.arange(0, new_index.shape[0], device=new_index.device)], dim=0)
    
    transform = DelaunayTransform()
    data = transform(Data(pos=out_positions[:, :2].to(data.pos.device), x=out_x.to(data.pos.device), surf=surf, device=data.pos.device))
    new_clusters, new_cluster_nodes = divide_mesh(data.pos, data.edge_index.T, clusters_per_layer)
    s = torch.Tensor([c.shape[0] for c in new_clusters])
    
    # Average edges per coarse element in a subgraph
    m = torch.round(((torch.sum(counts)//(2*counts.shape[0]))*6)*(s/torch.sum(s))).int()
    return data, connection_edge_index, new_clusters, m, new_cluster_nodes
    
    
def generate_coarse_graphs(data, R: int, K: int, factor=7, range_=5000, mp=9, visualize=False):
    data = data.cpu() # Quicker to compute on CPU
    edge_clusters, new_cluster_nodes = divide_mesh(data.pos, data.edge_index.T, K)
    data.clusters = edge_clusters
    data.node_clusters = new_cluster_nodes
    base = data.clone()
    base.R = R
    base.clusters_per_layer = K
    base.edge_frequencies = []
    base.layer_ranges = torch.zeros((R, 2), device=base.pos.device, dtype=torch.int)
    base.layer_ranges[0] = torch.tensor([0, base.pos.shape[0]], device=base.pos.device)
    base.up_scale_edge_ranges = torch.zeros((R-1, 2), device=base.pos.device, dtype=torch.int)
    if visualize:
        base.pos = torch.concatenate([base.pos, torch.full((base.pos.shape[0], 1), 1, device=base.pos.device)], axis=1)
    s = [base.pos.shape[0]]
    for i in range(2, R+1):
        subgraph, connection_index, new_clusters_edges, edge_frequencies, new_cluster_nodes = generate_coarse_graph(data, range_//(factor**i), base.clusters_per_layer) # TODO: choose the right scale factor
        base.edge_frequencies.append(edge_frequencies)
        # We add the new node clusters indexed in the subgraph
        base.node_clusters.extend(new_cluster_nodes)
        # We add the new clusters edge indices indexed in the subgraph 
        base.clusters.extend([c + base.edge_index.shape[1] for c in new_clusters_edges])

        data = subgraph.clone()
        s.append(subgraph.pos.shape[0])        
        
        
        # We need to add the new dimension to the positions to visualize them
        if visualize:
            subgraph.pos = torch.concatenate([subgraph.pos, torch.full((subgraph.pos.shape[0], 1), i, device=subgraph.pos.device)], axis=1) # TODO: remove it

        # We need to add the new edges to the base graph so the new nodes ids have to be shifted by the number of nodes in the base graph
        subgraph.edge_index = torch.add(subgraph.edge_index, base.pos.shape[0])
        
        # We need to connect the new nodes to the base graph nodes
        connection_index = torch.add(connection_index, sum(s[:-2]))

        base.pos = torch.cat([base.pos, subgraph.pos], dim=0) # TODO: use barycentric interpolation
        base.surf = torch.cat([base.surf, subgraph.surf], dim=0) 
        base.x = torch.cat([base.x, subgraph.x], dim=0)
        base.edge_index = torch.cat([base.edge_index, subgraph.edge_index, connection_index], dim=1)
        
        base.up_scale_edge_ranges[i-2] = torch.tensor([base.edge_index.shape[1]-connection_index.shape[1], base.edge_index.shape[1]], device=base.pos.device)
        base.layer_ranges[i-1] = torch.tensor([base.pos.shape[0]-subgraph.pos.shape[0]-1, base.pos.shape[0]], device=base.pos.device)
        if i >= R:
            last_one_frequencies = torch.full((K,), mp, device=base.pos.device, dtype=torch.int) 
            base.edge_frequencies.append(last_one_frequencies)
    return base
