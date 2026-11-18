# EXP_Simulate_ExtremeDays_MultiNFE.py
#
# Simulates the 100 most-precipitating test days for several models
# (DDPM baseline vs Flow Matching), sweeping several NFE (number of
# integration/denoising steps) values per model, all sharing the SAME
# initial noise across models and NFE values -- so results are directly
# comparable (same idea your colleague uses for his NFE sweep).

solver = "euler"

import xarray as xr

hash_cfg = True
number_of_simulations_targets = 80
batch_size_simulation = 512

# ------------------------------------------------------------
# NFE values to sweep, same list used for every model below.
# ------------------------------------------------------------
NFE_LIST = [5, 10, 25, 50, 100, 250, 500, 1000]
 
SIM_NAMES = [
    "DEFprDef_sqrTransforms_s0_d5b998X1bf0b7",
    "FlowDEFprFlowMatching_sqrTransforms_s0_b68803X1bf0b7",
]

# Config used ONLY to fit the shared transforms (both experiments use the
# same folder_background_predictors / seed / train_period / CFGD file --
# verified: CFG_Def.py and CFG_FlowMatching.py agree on all four).
DATA_TRANSFORMS_FILE = "CFGD_sqrTransforms"


import os
import sys
import importlib
import gc
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from utils.load_netcdf_data import load_tensor_data_from_netcdf_filepaths as load_data
from utils.load_netcdf_data import standardize_spatial_orientation
from models.DDPModel import Model

ROOT_MODEL_FOLDER = Path("./trained_model_parameters")
ROOT_SIM_FOLDER = Path("./final_models")


# --------------------------
# DDP setup for simulation
# --------------------------
use_ddp_simulation = int(os.environ.get("WORLD_SIZE", "1")) > 1

if use_ddp_simulation:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

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
# Slurm array index = simulation member (kept for parity with the other scripts;
# unused here since this script produces the full ensemble in one go)
# --------------------------
slurm_step_i = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
if is_main_process:
    print(f"SLURM_ARRAY_TASK_ID = {slurm_step_i}", flush=True)


# ----------------------------------------------------------------
# Load the config of the FIRST experiment only, to fit the shared
# transforms (both experiments use the exact same background predictor,
# seed, and train/test period -- this is NOT per-experiment).
# ----------------------------------------------------------------
shared_experiment = SIM_NAMES[0]
shared_model_folder = ROOT_MODEL_FOLDER / shared_experiment
shared_config_file = list(shared_model_folder.glob("CFG_*.py"))[0].resolve()

spec = importlib.util.spec_from_file_location("module_config_shared", shared_config_file)
module_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module_config)

module_data = importlib.import_module(DATA_TRANSFORMS_FILE)

threshold = module_config.threshold

_, _, yy_pr = load_data(
    path_predictors=module_config.predictors_fname,
    path_predictand=module_config.predictand_fname,
    predictand_var_name=module_config.varname_dataset,
    period=module_config.train_period,
    add_seasonal_predictors=True,
    input_padding=None,
)
yy_pr = yy_pr[:, None, :, :] 

xx_pr = torch.Tensor(np.load(module_config.folder_background_predictors + "/prediction_train.npy"))

xx_pr[:, [1], ...] = torch.exp(xx_pr[:, [1], ...])  # from log_shape to shape
xx_pr[:, [2], ...] = torch.exp(xx_pr[:, [2], ...])  # from log_scale to scale


# -------------------------------------------------
# Split BEFORE fitting normalization statistics, identically to training.
# -------------------------------------------------
seed = module_config.seed
fractionLeft4earlystop = module_config.fractionLeft4earlystop(xx_pr.shape[0])

torch.manual_seed(seed)

subsample_size = int(fractionLeft4earlystop * yy_pr.shape[0])
indices = torch.randperm(yy_pr.shape[0])[:subsample_size]

mask = torch.ones(yy_pr.shape[0], dtype=torch.bool)
mask[indices] = False

# Fit using TRAIN ONLY (same split as training)
xxPrTransforms = module_data.XPrTransforms(xx_pr=xx_pr[mask], threshold=threshold)
yy_pr_Transform = module_data.YPrTransforms(yy_pr=yy_pr[mask])

del indices, mask, xx_pr, yy_pr
gc.collect()
torch.cuda.empty_cache()


# -----------------------------------------------
# Load the 100 pre-selected extreme test days
# -----------------------------------------------
dates = np.load("./data/100_more_precipitating_days.npy")
date_indices = np.load("./data/100_more_precipitating_days_indices.npy")

xx_test_pr_raw = torch.Tensor(np.load(module_config.folder_background_predictors + "/prediction_test.npy"))

assert xx_test_pr_raw.shape[0] > date_indices.max(), (
    "date_indices out of range for prediction_test.npy -- check that it was "
    "generated against the same test_period/background predictor folder."
)

xx_test_pr_raw = xx_test_pr_raw[date_indices, ...]

xx_test_pr_raw[:, [1], ...] = torch.exp(xx_test_pr_raw[:, [1], ...])  # from log_shape to shape
xx_test_pr_raw[:, [2], ...] = torch.exp(xx_test_pr_raw[:, [2], ...])  # from log_scale to scale

xx_test_pr = xxPrTransforms.transform(xx_test_pr_raw)
xx_test_pr = xx_test_pr.to(dtype=torch.float32)
xx_test_pr_raw = xx_test_pr_raw.to(dtype=torch.float32)

assert xx_test_pr.shape[0] == dates.shape[0] == date_indices.shape[0]

n_test = xx_test_pr.shape[0]
h = xx_test_pr.shape[2]
w = xx_test_pr.shape[3]

if is_main_process:
    print(f"Loaded test set with shape {tuple(xx_test_pr.shape)}", flush=True)


# -----------------------------------------------------------
# Build the test dataloader ONCE -- noise and conditioning are
# shared across BOTH models and ALL NFE values on purpose.
# -----------------------------------------------------------
random_noise = torch.randn(
    (n_test * number_of_simulations_targets, 1, h, w)
)

conditionings = xx_test_pr.repeat_interleave(number_of_simulations_targets, dim=0)

index = torch.arange(conditionings.shape[0], dtype=torch.long)

testDataLoader = DataLoader(
    TensorDataset(random_noise, conditionings, index),
    batch_size=batch_size_simulation,
    shuffle=False,
    drop_last=False,
    pin_memory=True
)

if is_main_process:
    print(
        f"Simulating for {dates.shape[0]} days with "
        f"{number_of_simulations_targets} simulations each, "
        f"NFE in {NFE_LIST}...",
        flush=True
    )

_ds_coords = xr.open_dataset(module_config.predictand_fname)
_ds_coords = standardize_spatial_orientation(_ds_coords)
lat = _ds_coords["lat"].values
lon = _ds_coords["lon"].values
_ds_coords.close()
del _ds_coords
gc.collect()
torch.cuda.empty_cache()


# --------------------------
# Simulate: one model load per experiment, looped over all NFE values
# inside -- this avoids reloading model weights per NFE.
# --------------------------
for experiment in SIM_NAMES:

    model_folder = ROOT_MODEL_FOLDER / experiment
    sim_folder = ROOT_SIM_FOLDER / experiment / "simulations"
    sim_folder.mkdir(parents=True, exist_ok=True)

    config_file = list(model_folder.glob("CFG_*.py"))[0].resolve()

    spec = importlib.util.spec_from_file_location("module_config", config_file)
    module_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_config)

    model_module = module_config.model_module_pr
    variable = module_config.variable

    folder_final = ROOT_MODEL_FOLDER / experiment
    final_model_name = os.path.join(folder_final, variable)

    if is_main_process:
        print(f"Loading model from {final_model_name}", flush=True)

    model_pr = Model(
        model_module=model_module,
        ddp=False,
        device=device,
        world_size=world_size
    )

    # Prefer EMA weights for sampling if they were trained.
    ema_decay = getattr(module_config, "ema_decay", None)

    if ema_decay is not None:
        ema_checkpoint = final_model_name + "_ema"
        if os.path.exists(ema_checkpoint):
            model_pr.load_state_dict(ema_checkpoint)
            if is_main_process:
                print(f"Loaded EMA weights from {ema_checkpoint}.", flush=True)
        else:
            if is_main_process:
                print(f"ema_decay was set but {ema_checkpoint} was not found. Falling back to non-EMA weights.", flush=True)
            model_pr.load_state_dict(final_model_name)
    else:
        model_pr.load_state_dict(final_model_name)

    if is_main_process:
        print(f"Model loaded for {experiment}.", flush=True)

    is_flow_matching = "FlowDEFprFlowMatching_" in experiment
    is_ddpm = "DEFprDef_" in experiment

    if is_ddpm:
        # Scheduler built in the CFG with the TRAINING num_train_timesteps
        # (e.g. 500 for CFG_Def.py) -- NEVER rebuild it per NFE, just
        # restride it with set_timesteps() below.
        noise_scheduler_pr = module_config.noise_scheduler_pr

    for n_steps_simulation in NFE_LIST:

        if is_main_process:
            print(f"\n=== {experiment} | NFE = {n_steps_simulation} ===", flush=True)

        if is_flow_matching:
            time_scale = module_config.time_scale
            if not solver:
                solver = module_config.solver
                print("Solver", solver)
            simulations_targets = model_pr.simulateFlowMatching(
                testDataLoader,
                n_steps=n_steps_simulation,
                solver=solver,
                sub_batch_size=batch_size_simulation,
                time_scale=time_scale,
                transformSimulation=yy_pr_Transform.inverse
            )

        elif is_ddpm:
            noise_scheduler_pr.set_timesteps(n_steps_simulation, device=model_pr.device)

            simulations_targets = model_pr.simulateDiffusion(
                testDataLoader,
                noise_scheduler_pr,
                sub_batch_size=batch_size_simulation,
                transformSimulation=yy_pr_Transform.inverse
            )

        else:
            raise ValueError(f"Don't know how to simulate experiment: {experiment}")

        if is_main_process:
            simulations_targets = simulations_targets.reshape( dates.shape[0], number_of_simulations_targets, h, w )

            simulations_targets_ds = xr.Dataset(
                {
                    module_config.varname_dataset: ( ("time", "member", "lat", "lon"), simulations_targets )
                },
                coords={
                    "time": dates,
                    "member": np.arange(number_of_simulations_targets),
                    "lat": lat,
                    "lon": lon,
                }
            )

            encoding = {
                module_config.varname_dataset: {"zlib": True, "complevel": 4}
            }

            fname_out = sim_folder / f"simulations_extreme_days_nfe{n_steps_simulation}.nc"
            simulations_targets_ds.to_netcdf(fname_out, encoding=encoding)
            print(f"Saved {fname_out}", flush=True)

            gc.collect()
            torch.cuda.empty_cache()

    del model_pr
    gc.collect()
    torch.cuda.empty_cache()


if use_ddp_simulation:
    dist.barrier()
    dist.destroy_process_group()
