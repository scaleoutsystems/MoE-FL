#!/bin/bash -l
#SBATCH --job-name=convnext_moe
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

#Create the virtual environment
#python -m venv my_python-env

#Source to activate the virtual envronment
source my_python-env/bin/activate

#Install the dependencies (listed in requirement.txt) with pip in the virtual environment
#python -m pip install -r requirements.txt

#Execute the program
python imagenet_convnext_moe.py
