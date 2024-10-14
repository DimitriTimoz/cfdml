import torch
from torch_cluster import grid_cluster
from torch_scatter import scatter
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
        e (Tensor(2, N)): Edge indices of the mesh.
        k (int): Number of clusters to divide the mesh into.

    Returns:
        
    """
    clusters = [set() for _ in range(k)]
    
    # Randomly initialize centroids (2D points)
    centroids = torch.randn(k, 2, device=v.device)

    # Precompute edge directions and norms
    edges_directions = v[e[:, 1]] - v[e[:, 0]]
    edges_norms = torch.norm(edges_directions, dim=1, keepdim=True)  # Shape: [num_edges, 1]
    edges_directions /= edges_norms  # Normalize edge directions
    norm_changes = float('inf')
    while norm_changes > 1e-3:
        # Vectorized clustering step
        centroids_norms = torch.norm(centroids, dim=1, keepdim=True)  # Shape: [num_centroids, 1]
        cosine_angles = torch.matmul(edges_directions, centroids.T) / (centroids_norms.T)  # Shape: [num_edges, num_centroids]
        # We use directly the cosine because they have opposite variations
        max_edge_idxs = torch.argmax(cosine_angles, dim=1)  # Shape: [num_edges]
        # Efficient assignment to clusters using torch
        cluster_masks = [(max_edge_idxs == i) for i in range(k)]
        for i in range(k):
            clusters[i] = set((torch.nonzero(cluster_masks[i]).squeeze(1).tolist()))

        # Efficient centroid update
        n_m = 0.0
        for i in range(k):
            if clusters[i]:  # Check if the cluster has assigned edges
                cluster_edges = edges_directions[torch.tensor(list(clusters[i]), device=v.device)]
                last_centroid = centroids[i].clone()
                centroids[i] = torch.mean(cluster_edges, dim=0)
                n_m = max(torch.norm(centroids[i] - last_centroid), n_m)
        norm_changes = n_m

    # Post-process clusters to finalize edge indices
    clusters_edge_indices = [torch.tensor(list(cluster), device=v.device) for cluster in clusters]
    nodes = [torch.unique(e[cluster].view(-1)) for cluster in clusters_edge_indices]
    if verbose:
        angles = torch.rad2deg(torch.atan(centroids[:, 1] / centroids[:, 0]))
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
    out_positions = scatter(data.pos.t(), new_index, reduce='mean')
    
    out_x = scatter(data.x.t(), new_index, reduce='mean')
    
    # Interpolate the other features accordingly to the position    
    surf = scatter(data.surf.to(torch.int), new_index, reduce='max')
    connection_edge_index = torch.stack([new_index+data.pos.shape[0], torch.arange(0, new_index.shape[0], device=new_index.device)], dim=0)
    
    transform = DelaunayTransform()
    data = transform(Data(pos=out_positions.t()[:, :2].to(data.pos.device), x=out_x.t().to(data.pos.device), surf=surf, device=data.pos.device))
    new_clusters, new_cluster_nodes = divide_mesh(data.pos, data.edge_index.T, clusters_per_layer)
    s = torch.Tensor([c.shape[0] for c in new_clusters])
    
    # Average edges per coarse element in a subgraph
    m = torch.round(((torch.sum(counts)//(2*counts.shape[0]))*6)*(s/torch.sum(s))).int()
    return data, connection_edge_index, new_clusters, m, new_cluster_nodes
    
def generate_coarse_graphs(data, R: int, K: int, visualize=False):
    data = data.cpu() # Quicker to compute on CPU
    range_ = 750
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
        subgraph, connection_index, new_clusters_edges, edge_frequencies, new_cluster_nodes = generate_coarse_graph(data, range_//(5**i), base.clusters_per_layer) # TODO: choose the right scale factor
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
            last_one_frequencies = torch.full((K,), 1, device=base.pos.device, dtype=torch.int) 
            base.edge_frequencies.append(last_one_frequencies)
            # TODO
    return base
