# CFG_FlowMatchingUNET.py
import torch

from diffusers import UNet2DModel

import os
from utils.constants import (
    MAP_VARNAME_WITH_DATASETS_VARS,
    THRESHOLDS_PER_VARIABLES,
)

variable = "pr"  # pr   sfcWind

# Distinct prefix from CFG_DiffUNET.py's "DEF{variable}" -> different model_name -> different
# folder_final/folder_temp, so flow matching runs never collide with DDPM runs of the same variable.
model_name = f"FlowDEF{variable}"
folder_background_predictors = "final_models/pr_nllbergammaBernoulliGammaUNET_StandardTransforms4Prediction_incX1D_s0_b02a1fX40b171"
seed = 0
validation_seed = 0
max_epochs = 50
patience = None # set to None to disable early stopping
batch_size = 32
learningRate_pr = 4e-04
save_best_checkpoint = False # set to True to save the best checkpoint based on validation loss

saveModelEvery = 20
dropout = 0
write_losses = True
my_optimizer = torch.optim.Adam
my_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau # can be set to None

ema_decay = None  # set to None to disable EMA

# --- Flow matching specific (no noise_scheduler object here, unlike CFG_DiffUNET.py) ---
# time_scale: rescales the continuous t in [0,1] before it's fed to the model's sinusoidal timestep
# embedding, which is shaped for a ~[0, N) integer-like range (same embedding as the DDPM UNet uses).
# Must be identical between trainFlowMatchingModel (training) and simulateFlowMatching (sampling).
time_scale = 1000.0

# ODE integration settings, used only at sampling time (see EXP_BGFlowMatching_Simulate.py).
# Flow matching typically needs far fewer steps than DDPM's discrete schedule -- start low
# (e.g. 30-50) and only raise n_steps_simulation, or switch solver to "heun", if the distributional
# diagnostics (stats_pr.txt) show it's actually needed.
n_steps_simulation = 30
solver = "heun"  # "euler" or "heun" (heun: 2x cost per step, usually better quality per step count)

varname_dataset = MAP_VARNAME_WITH_DATASETS_VARS.get(variable, "")
threshold = THRESHOLDS_PER_VARIABLES.get(variable, "")

db = "ERA5"
domain = "SESA"
train_period = ("1981", "2012")
test_period = ("2013", "2024")

datafolder = "data/"
predictand_fname = os.path.join(datafolder, "predictand", f"{variable}_{domain}_{db}.nc")
predictors_fname = os.path.join(datafolder, "predictors", "predictors_ERA5_MD2_2-0deg.nc")

use_double_precision = False
fractionLeft4earlystop = lambda dataset_size: (256*round((dataset_size*0.1)/256))/dataset_size


time_embedding_dim = 256


class wrapped_huggingUNET(UNet2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # You can add custom logic here if needed
        return super().forward(return_dict = False, *args, **kwargs)[0]


# Same architecture as CFG_DiffUNET.py: the velocity field v_phi(x_t, t, X_t) and the noise-prediction
# epsilon_phi(x_t, t, X_t) of DDPM are both just "a UNet that takes a noisy/interpolated image, a
# timestep, and conditioning, and outputs one channel" -- the network doesn't need to change, only
# what target it's trained to predict (handled in DDPModel.trainFlowMatchingModel).
model_module_pr = wrapped_huggingUNET(
                                sample_size=torch.Size([64, 64]),
                                in_channels = 5,
                                out_channels = 1,

                                layers_per_block = 2,
                                block_out_channels = (64, 128, 256),

                                down_block_types = (
                                        "DownBlock2D",
                                        "DownBlock2D",
                                        "AttnDownBlock2D"   # attention at 16x16 (no downsample on the final block)
                                ),
                                up_block_types = (
                                        "AttnUpBlock2D",
                                        "AttnUpBlock2D",
                                        "UpBlock2D"
                                ),

                                attention_head_dim = 8,
                                norm_num_groups = 32,
                                time_embedding_dim = time_embedding_dim
)
