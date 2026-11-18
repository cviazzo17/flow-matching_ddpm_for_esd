#!/bin/bash
##---------------Script SBATCH - NLHPC ----------------
#SBATCH -J DL_Def
##SBATCH -p mi210
#SBATCH -p v100
#SBATCH --gpus=1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=15G
#SBATCH -t 10:00:00
#SBATCH -o DL_Def_%j.out
#SBATCH -e DL_Def_%j.out
#SBATCH --mail-user=carlaviazzo17@gmail.com
#SBATCH --mail-type=ALL

#-----------------Toolchain---------------------------
ml purge

# ----------------Modulos----------------------------
# ml python/3.11.7-zen4-b

# ----------------Conda------------------------------
eval "$(conda shell.bash hook)"
conda activate downscaling_v100
# conda activate downscaling_rocm7
#conda activate downscaling

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
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
else:
    raise Exception('No GPU was found.')
"

# ----------------Comando--------------------------

echo "Starting at: $(date)"
START=$(date +%s)

path_script="../.."
cd "${path_script}"

echo "Script de python inicia ejecución"
# srun python EXP_BGDiffusion.py
srun python EXP_BGDiffusion_DenseLoss.py
echo "Script de python finaliza ejecución"


END=$(date +%s)
echo "Finished at: $(date)"
