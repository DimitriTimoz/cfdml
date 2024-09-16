# %%
import os
from lips import get_root_path

# %%
LIPS_PATH = get_root_path()
DIRECTORY_NAME = 'Dataset'
BENCHMARK_NAME = "Case1"
LOG_PATH = LIPS_PATH + "lips_logs.log"
BENCH_CONFIG_PATH = os.path.join("airfoilConfigurations","benchmarks","confAirfoil.ini") #Configuration file related to the benchmark
SIM_CONFIG_PATH = os.path.join("airfoilConfigurations","simulators","torch_fc.ini") #Configuration file re

# %%
from lips.dataset.airfransDataSet import download_data
if not os.path.isdir(DIRECTORY_NAME):
    download_data(root_path=".", directory_name=DIRECTORY_NAME)

# %%
from lips.benchmark.airfransBenchmark import AirfRANSBenchmark

benchmark=AirfRANSBenchmark(benchmark_path = DIRECTORY_NAME,
                            config_path = BENCH_CONFIG_PATH,
                            benchmark_name = BENCHMARK_NAME,
                            log_path=LOG_PATH)
                            
benchmark.load(path=DIRECTORY_NAME)

# %%
import torch
import json
import importlib
import solution
import solution.models
import solution.models.basic
f = open("solution/parameters.json")
parameters = json.load(f)
hparams = parameters["simulator_extra_parameters"]

device = torch.device("cuda:0")
module = importlib.import_module("solution." + parameters["simulator_config"]["simulator_file"])
importlib.reload(module)
Network = getattr(module, parameters["simulator_config"]["model"])
global_train = getattr(module, "global_train")
network = Network(**hparams)
network.train(benchmark.train_dataset,
             local=True)


