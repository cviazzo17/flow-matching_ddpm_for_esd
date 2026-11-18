# EXP_BGFlowMatching_Simulate.py
#
# Simulate with the model trained in EXP_BGFlowMatching.py
# - one Slurm array task = one simulation member
# - optional multi-GPU parallel simulation through DistributedSampler (per node/task)




hash_cfg = True
folder_simulations = "final_models"
max_number_of_simulations = 15  # number of the ensamble size  
number_of_simulations_targets = 50 # 1000
batch_size_simulation = 512
simulate_specific_days = False


solver = "euler"   # none to use default (CFG*)
n_steps_simulation = 50    # none to use default (CFG*)

import sys

DEFAULT_MODEL_CONFIG_FILE = "CFG_FlowMatching"
DEFAULT_DATA_TRANSFORMS_FILE = "CFGD_sqrTransforms"

MODEL_CONFIG_FILE = (
    sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_CONFIG_FILE
)
DATA_TRANSFORMS_FILE = (
    sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DATA_TRANSFORMS_FILE
)


import os
import importlib
import hashlib
from pathlib import Path
import time

from utils.load_netcdf_data import load_tensor_data_from_netcdf_filepaths as load_data

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from models.DDPModel import Model
import gc

# --------------------------
# DDP setup for simulation
# --------------------------
use_ddp_simulation = int(os.environ.get("WORLD_SIZE", "1")) > 1

if use_ddp_simulation:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Keep same spirit as your previous simulation worker
    # because simulateFlowMatching may communicate CPU tensors
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )

    torch.cuda.set_device(local_rank)
    device = local_rank
else:
    local_rank = 0
    rank = 0
    world_size = 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

is_main_process = (rank == 0)

print(f"[simulate] rank={rank}, local_rank={local_rank}, world_size={world_size}", flush=True)


# --------------------------
# Slurm array index = simulation member
# --------------------------
slurm_step_i = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
if is_main_process:
    print(f"SLURM_ARRAY_TASK_ID = {slurm_step_i}", flush=True)


# --------------------------
# Import config modules
# --------------------------
module_model_config = importlib.import_module(MODEL_CONFIG_FILE)
module_data = importlib.import_module(DATA_TRANSFORMS_FILE)
ema_decay = getattr(module_model_config, "ema_decay", None)

# Flow matching specific -- no noise_scheduler here
time_scale = module_model_config.time_scale
if not solver:
    solver = module_model_config.solver

if not n_steps_simulation:
    n_steps_simulation = module_model_config.n_steps_simulation

print("Solver:", solver)
print("Number of simulation steps:", n_steps_simulation)


# --------------------------
# Hash logic copied from training
# --------------------------
def file_hash(path, n=6):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


if hash_cfg:
    base = Path().absolute()
    config_path = base / f"{MODEL_CONFIG_FILE}.py"
    config_path_data = base / f"{DATA_TRANSFORMS_FILE}.py"

    hash_model = file_hash(config_path)
    hash_data = file_hash(config_path_data)
    unique_identifier_hash = f"{hash_model}X{hash_data}"


# --------------------------
# Load training data to fit transforms exactly as in training
# --------------------------

predictand_fname = module_model_config.predictand_fname
predictors_fname = module_model_config.predictors_fname
predictand_var_name = module_model_config.varname_dataset

train_period = module_model_config.train_period
threshold = module_model_config.threshold


_, _, yy_pr = load_data(
    path_predictors=predictors_fname,
    path_predictand=predictand_fname,
    predictand_var_name=predictand_var_name,
    period=train_period,
    add_seasonal_predictors=True,
    input_padding=None,
)

xx_pr = torch.Tensor(np.load(module_model_config.folder_background_predictors  + "/prediction_train.npy"))

yy_pr = yy_pr[:, None, :, :]  # ensure yy_pr has shape (N, 1, H, W) for the transforms (and later for the model)


xx_pr[:, [1], ...] = torch.exp(xx_pr[:, [1], ...]) # from log_shape to shape
xx_pr[:, [2], ...] = torch.exp(xx_pr[:, [2], ...]) # from log_scale to scale


# -------------------------------------------------
# Split BEFORE fitting normalization statistics, identically to EXP_BGFlowMatching.py,
# so the transforms here match exactly the ones used during training.
# -------------------------------------------------
seed = module_model_config.seed
fractionLeft4earlystop = module_model_config.fractionLeft4earlystop(xx_pr.shape[0])

torch.manual_seed(seed)

subsample_size = int(fractionLeft4earlystop * yy_pr.shape[0])
indices = torch.randperm(yy_pr.shape[0])[:subsample_size]

mask = torch.ones(yy_pr.shape[0], dtype=torch.bool)
mask[indices] = False

# Initialize transform classes, fit using TRAIN ONLY (same split as training)
xxPrTransforms = module_data.XPrTransforms(xx_pr=xx_pr[mask], threshold=threshold)
yy_pr_Transform = module_data.YPrTransforms(yy_pr=yy_pr[mask])

del(indices)
del(mask)
del(xx_pr)
del(yy_pr)

gc.collect()
torch.cuda.empty_cache()

# --------------------------
# Rebuild trained model name exactly as in training
# --------------------------
model_name = (
    module_model_config.model_name
    + MODEL_CONFIG_FILE.split("_", 1)[1]
    + "_"
    + DATA_TRANSFORMS_FILE.split("_", 1)[1]
    + "_s"
    + str(module_model_config.seed)
)

if hash_cfg:
    model_name += "_" + unique_identifier_hash


variable = module_model_config.variable

folder_final = "trained_model_parameters/" + model_name
folder_simulations = folder_simulations + "/" + model_name
final_model_name = os.path.join(folder_final, variable) 

if is_main_process:
    print(f"Loading model from {final_model_name}", flush=True)


# --------------------------
# Load model
# --------------------------
model_module = module_model_config.model_module_pr

# Keep ddp=False here.
# Parallelism for simulation is handled by DistributedSampler + one process per GPU.
model_pr = Model(
    model_module=model_module,
    ddp=False,
    device=device,
    world_size=world_size
)
model_pr.load_state_dict(final_model_name)

# Prefer EMA weights for sampling if they were trained: this is what you'd deploy for actual downscaling.
if ema_decay is not None:
    ema_checkpoint = final_model_name + "_ema"
    if os.path.exists(ema_checkpoint):
        model_pr.load_state_dict(ema_checkpoint)
        if is_main_process:
            print(f"Loaded EMA weights from {ema_checkpoint} for simulation.", flush=True)
    else:
        if is_main_process:
            print(f"ema_decay was set but {ema_checkpoint} was not found. Falling back to the non-EMA weights.", flush=True)

if is_main_process:
    print("Model loaded.", flush=True)



# --------------------------
# Load / prepare test set
# --------------------------
# Replace this loader if needed.
xx_test_pr_raw = torch.Tensor(np.load(module_model_config.folder_background_predictors  + "/prediction_test.npy"))

xx_test_pr_raw[:, [1], ...] = torch.exp(xx_test_pr_raw[:, [1], ...]) # from log_shape to shape
xx_test_pr_raw[:, [2], ...] = torch.exp(xx_test_pr_raw[:, [2], ...]) # from log_scale to scale


xx_test_pr = xxPrTransforms.transform(xx_test_pr_raw)
xx_test_pr = xx_test_pr.to(dtype=torch.float32)

del(xx_test_pr_raw)
gc.collect()
torch.cuda.empty_cache()

# --------------------------
# Simulate
# --------------------------
folder_simulations = os.path.join(folder_simulations, "simulations")
os.makedirs(folder_simulations, exist_ok=True)


# The complete set of simulations:

n_test = xx_test_pr.shape[0]
h = xx_test_pr.shape[2]
w = xx_test_pr.shape[3]

if is_main_process:
    print(f"Loaded test set with shape {tuple(xx_test_pr.shape)}", flush=True)



keep_simulating = (len(os.listdir(folder_simulations)) <= max_number_of_simulations)

while keep_simulating:
    if is_main_process:
        print(f"Current number of simulations: {len(os.listdir(folder_simulations))}.", flush=True)

    # Generate random string based on current time in nanosecond
    # I use this for the name!
    seed_simulation = time.time_ns() % 10_000_000
    np.random.seed(seed_simulation) 
    torch.manual_seed(seed_simulation)  


    # Build distributed test dataloader
    sample = torch.randn(n_test, 1, h, w)
    index = torch.arange(n_test, dtype=torch.long)

    normalized_dataset_test = TensorDataset(sample, xx_test_pr, index)

    if use_ddp_simulation:
        test_sampler = DistributedSampler(
            normalized_dataset_test,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
    else:
        test_sampler = None

    test_dataloader = DataLoader(
        normalized_dataset_test,
        batch_size=int(np.ceil(n_test / world_size)),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        sampler=test_sampler
    )


    simname = os.path.join(
        folder_simulations,
        "simulation_" + str(seed_simulation) + str(slurm_step_i) + ".npy"
    )

    if is_main_process:
        print("\nProducing simulation " + simname + "...\n", flush=True)



    model_pr.simulateFlowMatching(
        test_dataloader,
        n_steps=n_steps_simulation,
        solver=solver,
        sub_batch_size = batch_size_simulation,
        time_scale=time_scale,
        transformSimulation=yy_pr_Transform.inverse,
        file_name=simname
    )

    if is_main_process:
        print(f"Simulation saved to {simname}", flush=True)
    
    del test_dataloader
    gc.collect()
    torch.cuda.empty_cache()


    keep_simulating = (len(os.listdir(folder_simulations)) < max_number_of_simulations)



if simulate_specific_days:
    # For specific days:
    dateS_ids = np.arange(9) # np.load("data" + "/target_dates.npy")


    random_noise = torch.randn(
        (dateS_ids.shape[0] * number_of_simulations_targets, 1, xx_test_pr.shape[2], xx_test_pr.shape[3])
    )
    conditionings = xx_test_pr[dateS_ids, ...]
    conditionings = conditionings.repeat_interleave(number_of_simulations_targets, dim=0)
    index = torch.arange(conditionings.shape[0], dtype=torch.long)


    testDataLoader_days = DataLoader(
        TensorDataset(random_noise, conditionings, index),
        batch_size=batch_size_simulation,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    if is_main_process:
        print(f"Simulating for {dateS_ids.shape[0]} days with {number_of_simulations_targets} simulations each "
            f"({solver}, {n_steps_simulation} steps)...", flush=True)

    simulations_targets = model_pr.simulateFlowMatching(
        testDataLoader_days,
        n_steps=n_steps_simulation,
        solver=solver,
        time_scale=time_scale,
        transformSimulation=yy_pr_Transform.inverse
    )

    if is_main_process:

        simulations_targets = torch.tensor(
            simulations_targets.reshape(
                dateS_ids.shape[0],
                number_of_simulations_targets,
                xx_test_pr.shape[2],
                xx_test_pr.shape[3]
            )
        )


    np.save(folder_simulations + "/simulations_days.npy", simulations_targets)


    del(random_noise)
    del(conditionings)
    del(testDataLoader_days)
    del(simulations_targets)

    gc.collect()
    torch.cuda.empty_cache()



if use_ddp_simulation:
    dist.barrier()
    dist.destroy_process_group()
    