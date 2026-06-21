#!/bin/bash -l
#SBATCH --job-name=convnext_imagenet_centralized
#SBATCH -N 1
#SBATCH --time=18:00:00
#SBATCH --account=p201222
#SBATCH -p gpu
#SBATCH -q default
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

module load Python
module load CUDA/12.8.0
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0-whl
module load torchvision/0.24.1-foss-2025a-CUDA-12.8.0-whl

source my_python-env/bin/activate

torchrun --nproc_per_node=4 imagenet_convnext_centralized.py