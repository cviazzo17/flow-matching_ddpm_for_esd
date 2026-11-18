MODEL_CONFIG_FILE = "CFG_BernoulliGammaUNET"  # Common to all experiments and domains.
DATA_TRANSFORMS_FILE = "CFGD_StandardTransforms4Prediction_incX1D" 
hash_cfg = True
folder_simulations = "final_models"
max_number_of_simulations = 50  
datafolder = "data/"
batch_size_simulation = 256
save_some_stats = True
save_prediction_as_diffusion_background = True

import os
import importlib
from functions.simulateBernoulliGamma import simulate_bernoulli_gammaV2 as simulate_bernoulli_gamma
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from utils.load_netcdf_data import load_tensor_data_from_netcdf_filepaths as load_data

from models.DDPModel import Model


# --------------------------
# Set number of threads
# --------------------------
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
print("Using threads:", torch.get_num_threads())


local_rank = 0
rank = 0
world_size = 1
device = "cuda:0" if torch.cuda.is_available() else "cpu"

is_main_process = (rank == 0)

print(f"rank={rank}, local_rank={local_rank}, world_size={world_size}", flush=True)



module_model_config = importlib.import_module(MODEL_CONFIG_FILE)
module_data = importlib.import_module(DATA_TRANSFORMS_FILE)



if hash_cfg:
    from pathlib import Path
    import hashlib

    def file_hash(path, n=6):
        h = hashlib.sha256()
        with open(path, "rb") as f:  # binary mode (important)
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
test_period = module_model_config.test_period

threshold = module_model_config.threshold

xx_pr, xx1d_pr, yy_pr, target_dates = load_data(
    path_predictors=predictors_fname,
    path_predictand=predictand_fname,
    predictand_var_name=predictand_var_name,
    period=train_period,
    add_seasonal_predictors=True,
    input_padding=None,
    return_target_dates=True
)


pr_shape = yy_pr.shape


# Transforms to y (precip):
yy_pr = yy_pr[:, None, :, :]  # add channel dimension

# -------------------------------------------------
# Split BEFORE fitting normalization statistics, identically to EXP_BernoulliGamma.py,
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
xxPrTransforms = module_data.XPrTransforms(xx_pr=xx_pr[mask])
xx1DPrTransforms = module_data.XPrTransforms1D(xx_pr=xx1d_pr[mask])
yy_pr_Transform = module_data.YPrTransforms(yy_pr=yy_pr[mask])


# --------------------------
# Rebuild trained model name exactly as in training
# --------------------------
variable = module_model_config.variable

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

folder_final = "trained_model_parameters/" + model_name
folder_simulations = folder_simulations + "/" + model_name
final_model_name = os.path.join(folder_final, variable)

if is_main_process:
    print(f"Loading model from {final_model_name}", flush=True)


# --------------------------
# Load model
# --------------------------
model_module = module_model_config.model_module

# Keep ddp=False here.
# Parallelism for simulation is handled by DistributedSampler + one process per GPU.
model_pr = Model(
    model_module=model_module,
    ddp=False,
    device=device,
    world_size=world_size
)
model_pr.load_state_dict(final_model_name)




if is_main_process:
    print("Model loaded.", flush=True)
    folder_to_save_predictions = folder_simulations
    os.makedirs(folder_to_save_predictions, exist_ok=True)
    folder_simulations = os.path.join(folder_simulations, "simulations")
    os.makedirs(folder_simulations, exist_ok=True)




if save_prediction_as_diffusion_background:
    xx_pr = xxPrTransforms.transform(xx_pr)
    xx1d_pr = xx1DPrTransforms.transform(xx1d_pr)

    TensorDataset_train = TensorDataset(xx_pr, xx1d_pr)
    train_dataloader = DataLoader(TensorDataset_train, batch_size = batch_size_simulation,
                                shuffle=False, drop_last=False)

    prediction_train = torch.empty((0, 3, pr_shape[-2], pr_shape[-1]), dtype=torch.float32)
    for batch in train_dataloader:
        prediction_train = torch.cat([prediction_train, model_pr.predict(batch[0], x_1D = batch[1])], dim=0)
                
    np.save(folder_to_save_predictions + "/prediction_train.npy", prediction_train.cpu().numpy())
    
    del(TensorDataset_train)
    del(train_dataloader)
    del(prediction_train)



del(xx_pr)
del(xx1d_pr)


xx2d_test, xx1d_test, _, target_dates = load_data(
    path_predictors=predictors_fname,
    path_predictand=predictand_fname,
    predictand_var_name=predictand_var_name,
    period=test_period,
    add_seasonal_predictors=True,
    input_padding=None,
    return_target_dates=True
)



xx2d_test = xxPrTransforms.transform(xx2d_test)
xx1d_test = xx1DPrTransforms.transform(xx1d_test)



TensorDataset_test = TensorDataset(xx2d_test, xx1d_test)
test_dataloader = DataLoader(TensorDataset_test, batch_size = batch_size_simulation,
                             shuffle=False, drop_last=False)

prediction = torch.empty((0, 3, pr_shape[-2], pr_shape[-1]), dtype=torch.float32)
for batch in test_dataloader:
    prediction = torch.cat([prediction, model_pr.predict(batch[0], x_1D = batch[1])], dim=0)
            



if save_prediction_as_diffusion_background:
    np.save(folder_to_save_predictions + "/prediction_test.npy", prediction.cpu().numpy())



# --------------------------
# Simulate
# --------------------------
folder_simulations = os.path.join(folder_simulations, "simulations")
os.makedirs(folder_simulations, exist_ok=True)





for sim_name in range(50):
    simulation = simulate_bernoulli_gamma(p = prediction[:, 0, ...],
                                          log_shape = prediction[:,1, ...], log_scale = prediction[:,2, ...],
                                          localize = threshold).unsqueeze(1)

    simulation = simulation.numpy()
    np.save(folder_simulations + "/simulation_" + str(sim_name) + ".npy", simulation)




dateS_ids = np.arange(len(target_dates))[:9]

simulations_targets = np.zeros((9, 1000, 64, 64))

print(prediction.shape)
print(dateS_ids.min(), dateS_ids.max())
print(len(dateS_ids))

for i in range(1000):
    simulations_targets[:,i,...] = simulate_bernoulli_gamma(p = prediction[dateS_ids, 0, ...],
                                                            log_shape = prediction[dateS_ids,1, ...],
                                                            log_scale = prediction[dateS_ids,2, ...],
                                          localize = threshold)


np.save(folder_simulations + "/simulations_days.npy", simulations_targets)




