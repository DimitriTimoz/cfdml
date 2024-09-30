

# %%
import torch
import matplotlib.pyplot as plt
from utils import  DelaunayTransform
from torch_geometric.data import Data
import torch_geometric
import networkx as nx
import pyvista as pv
import numpy as np

# %%
pos = torch.rand((100, 3))
pos[:, 2] = 0
data = Data(pos=pos[:, :2], surf=torch.full((100, 1), False))
transform = DelaunayTransform()
data = transform(data)
data.pos = pos

mesh = pv.PolyData()
mesh.points = data.pos.numpy()
edges = data.edge_index.t().numpy()
lines = np.hstack([np.full((edges.shape[0], 1), 2), edges]).ravel()
mesh.lines = lines

# %%
mesh.plot(show_edges=True, line_width=1)



