#!/bin/bash -l
#SBATCH --job-name=zod_experiment
#SBATCH -N 1
#SBATCH --time=18:00:00
#SBATCH --account=p201222
#SBATCH -p gpu
#SBATCH -q default
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

#Usage, arguments are passed through to zod_experiment.py:
#  sbatch zod_experiment.sh --setting fl --method context_distributed
#  sbatch zod_experiment.sh --setting centralized --method context_aware
#  sbatch zod_experiment.sh --setting fl --method moe --benchmark
#Resume after the time limit: add --ckpt runs/<run dir>

module load Python
module load CUDA/12.8.0
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0-whl
module load torchvision/0.24.1-foss-2025a-CUDA-12.8.0-whl

source my_python-env/bin/activate

if [[ " $* " == *" --setting centralized "* && " $* " != *" --benchmark "* ]]; then
    torchrun --nproc_per_node=4 zod_experiment.py "$@"
else
    python zod_experiment.py "$@"
fi
