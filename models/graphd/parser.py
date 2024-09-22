import os
import concurrent.futures
import numpy as np
import pyvista as pv

def load_edges(dataset, path):
    raws = dataset.extra_data["simulation_names"]
    # Generate paths to simulation files
    simulation_list = [
        (os.path.join(path, simulation, f"{simulation}_internal.vtu"), simulation)
        for simulation, _ in raws
    ]

    # Use efficient parallel processing
    edges = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_name = {
            executor.submit(_extract_edges, sim_path): name for sim_path, name in simulation_list
        }

        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
                edges[name] = result
                print(f"Simulation {name} has been processed")
            except Exception as exc:
                print(f"Simulation {name} generated an exception: {exc}")

    return edges

def _extract_edges(path):
    internal = pv.read(path)
    edges = internal.extract_all_edges()
    lines = edges.lines

    # Optimize edge extraction by directly accessing point indices
    edges_array = np.column_stack((lines[1::3], lines[2::3]))
    return edges_array
