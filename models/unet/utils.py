from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

import torch

import numpy as np
from scipy.spatial import Delaunay

class Delaunay(BaseTransform):
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
