#!/bin/bash
#SBATCH --job-name=NXS3DBB
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -x

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load SciPy-bundle/2023.11-gfbf-2023b
module load OpenCV/4.8.1-foss-2023a-CUDA-12.1.1-contrib

source "$HOME/venvs/BaIPNXS/bin/activate"
export PYTHONPATH=$PWD:$PYTHONPATH

python3 Train.py
