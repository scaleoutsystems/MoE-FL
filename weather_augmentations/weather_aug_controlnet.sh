#!/bin/bash -l
#SBATCH --job-name=weather_aug
#SBATCH -N 1
#SBATCH --time=48:00:00
#SBATCH --account=p201222
#SBATCH -p gpu
#SBATCH -q default
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32

export HF_HOME=/project/home/p201222/.cache/huggingface

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

#Start 4 parallel workers
srun -n 1 --exact --gres=gpu:1  python weather_aug_worker.py --task_id 0 &
srun -n 1 --exact --gres=gpu:1  python weather_aug_worker.py --task_id 1 &
srun -n 1 --exact --gres=gpu:1  python weather_aug_worker.py --task_id 2 &
srun -n 1 --exact --gres=gpu:1  python weather_aug_worker.py --task_id 3 &
wait
echo "All tasks finished"
