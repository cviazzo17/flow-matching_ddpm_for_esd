# CFG_BG_UNET.py

import torch
import torch.nn as nn
import numpy as np
import os

from models.UNET import UNET
from models.bernoulliGammaLogLikelihood import negativeBernoulliGammaLogLikelihoodV2 as negativeBernoulliGammaLogLikelihood
from models.aux import ConvolutionalLayer

from utils.constants import (
    MAP_VARNAME_WITH_DATASETS_VARS,
    THRESHOLDS_PER_VARIABLES,
)

variable = "sfcWind"  # pr   sfcWind

model_name = f"{variable}_nllbergamma"
seed = 0
max_epochs = 100000
patience = 50
batch_size = 32
fractionLeft4earlystop = lambda dataset_size: (256*round((dataset_size*0.1)/256))/dataset_size
learningRate = 1e-3
saveModelEvery = 100
dropout = 0
write_losses = True
my_optimizer = torch.optim.AdamW
weight_decay = 0.0001
my_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau # can be set to None

varname_dataset = MAP_VARNAME_WITH_DATASETS_VARS.get(variable, "")
threshold = THRESHOLDS_PER_VARIABLES.get(variable, "")

db = "ERA5"
domain = "SESA"
train_period = ("1981", "2012")
test_period = ("2013", "2024")


datafolder = "data/"
predictand_fname = os.path.join(datafolder, "predictand", f"{variable}_{domain}_{db}.nc")
predictors_fname = os.path.join(datafolder, "predictors", "predictors_ERA5_FPS-SSA_1-5deg_conservative.nc")


my_loss = negativeBernoulliGammaLogLikelihood(wet_day_threshold = threshold,
                                              epsilon_if_zero = 1e-5, 
                                              localize = True)


class input_processing_block_2D(nn.Sequential):
    def __init__(self, input_shape) -> None:
        super().__init__()
        self.out_channels = int(32)
        in_channels = input_shape[1]
        kernel_size_1 = (2, 7)
        kernel_size_2 = (1, 8)
        padding_1 = (0, 0)
        padding_2 = (0, 0)

        self.block = nn.Sequential(
            ConvolutionalLayer(in_channels=in_channels, out_channels=self.out_channels,
                                kernel_size=kernel_size_1, padding=padding_1),
            ConvolutionalLayer(in_channels=self.out_channels, out_channels=self.out_channels,
                                kernel_size=kernel_size_2, padding=padding_2)
        )

        def conv_out(size, k, p):  # stride=1, dilation=1
            return size + 2 * p - (k - 1) - 1 + 1

        h1 = conv_out(input_shape[2], kernel_size_1[0], padding_1[0])
        w1 = conv_out(input_shape[3], kernel_size_1[1], padding_1[1])
        h2 = conv_out(h1, kernel_size_2[0], padding_2[0])
        w2 = conv_out(w1, kernel_size_2[1], padding_2[1])

        self.output_shape = (input_shape[0], self.out_channels, h2, w2)


model_module = UNET(input_shape_2D = torch.Size([10, 25, 17, 29]), 
                    input_shape_1D = torch.Size([10, 52]),
                    # output_channels = [nn.Sigmoid, nn.Softplus,  nn.Softplus], 
                    output_channels = [nn.Sigmoid, nn.Identity, nn.Identity], 
                    double_resolution_times = 2, 
                    kernel_size = 2, 
                    input_processing_block_2D = input_processing_block_2D(torch.Size([10, 25, 17, 29])),
                    input_number_kernels = 32,
                    number_of_blocks = 2,
                    inputNN_neurons_list = [16, 32],
                    dropout_rate_1Dinput=dropout,
                    dropout_rate_2D=dropout)
