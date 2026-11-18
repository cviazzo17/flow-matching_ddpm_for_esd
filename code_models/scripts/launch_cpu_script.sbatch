#!/bin/bash
##---------------Script SBATCH - NLHPC ----------------
#SBATCH -J PrepareNetCDFs
#SBATCH -p largemem
#SBATCH -n 1
#SBATCH -c 1
##SBATCH --mem-per-cpu=2200
#SBATCH --mem=9000M
#SBATCH -t 24:00:00
#SBATCH -o PrepareNetCDFs_%j.out
#SBATCH -e PrepareNetCDFs_%j.out
#SBATCH --mail-user=carlaviazzo17@gmail.com
#SBATCH --mail-type=ALL

#-----------------Toolchain---------------------------
ml purge

# ----------------Modulos----------------------------
# ml python/3.11.7-zen4-b

# ----------------Conda------------------------------
eval "$(conda shell.bash hook)"
conda activate downscaling

echo "Verify the conda environment is correctly activated"
#which python
#python --version
python -c "
import torch
import numpy as np
print('torch version:', torch.__version__)
print('numpy version:', np.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CPU count:', torch.get_num_threads())
print('Environment OK')
"

# ----------------Comando--------------------------

echo "Starting at: $(date)"
START=$(date +%s)

path_script="."
cd "${path_script}"

echo "Script de python starts..."
srun python prepare_netcdf.py --SIM FlowRESIDprFlowMatching_Residual_ResidualSqrTransforms_s0_25c44fX13ce29
echo "Script de python ends."


END=$(date +%s)
echo "Finished at: $(date)"

DURATION=$(( ($END - $START) / 60 ))
echo "It takes $DURATION mins to complete the task."
