import torch
import pyvista as pv
import numpy as np
import json
import os
import concurrent.futures

# Read the PyVista UnstructuredGrid

def get_dataset_name(dataset, train):
    taskk = 'full' if dataset._task == 'scarce' and not train else dataset._task
    split = 'train' if train else 'test'
    return f"{taskk}_{split}"


def load_manifest(dataset_path) -> dict:
    with open(os.path.join(dataset_path, "manifest.json"), 'r') as file:
        return json.load(file)

def load_edges(dataset, path, train):
    dataset_name = get_dataset_name(dataset, train)
    manifest = load_manifest(path)

    # Generate paths to simulation files
    simulation_list = [
        os.path.join(path, simulation, f"{simulation}_internal.vtu")
        for simulation in manifest[dataset_name]
    ]

    # Use efficient parallel processing
    edges = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(_extract_edges, sim_path): sim_path for sim_path in simulation_list}

        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            edges.append(future.result())
    
    return edges

def _extract_edges(path) -> torch.Tensor:
    internal = pv.read(path)
    node_length = len(internal.celltypes)
    # Extract the cell array directly
    cell_array = internal.cells.copy()

    # First pass: Count the number of edges
    num_edges = 0
    i = 0
    while i < len(cell_array):
        n_points = cell_array[i]
        if n_points == 2:  # Only count cells with 2 points (i.e., edges)
            num_edges += 1
        i += n_points + 1

    # Preallocate the edges array
    edges = np.empty((2, num_edges), dtype=np.int64)

    # Second pass: Fill the edges array
    i = 0
    edge_index = 0
    while i < len(cell_array):
        n_points = cell_array[i]
        if n_points == 2:  # Process cells with 2 points (edges)
            edges[0, edge_index] = cell_array[i + 1]
            edges[1, edge_index] = cell_array[i + 2]
            edge_index += 1
        i += n_points + 1

    # Convert the preallocated array to a torch tensor
    return torch.tensor(edges, dtype=torch.long), node_length
