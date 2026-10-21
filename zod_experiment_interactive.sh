#!/bin/bash
set -eo pipefail

# --- 1. Configuration ---
NETWORK_DIR="/MoE-FL-ai-sweden"
LOCAL_ROOT="/workspace"
LOCAL_PROJECT_DIR="$LOCAL_ROOT/MoE-FL-ai-sweden"
IMAGE_DIR="$LOCAL_PROJECT_DIR/data/images_1333"    # the JSONs point to ../images_1333/{train,val}/<id>.jpg
ZIP_DIR="$IMAGE_DIR"                               # train.zip and val.zip (the images) sit in images_1333 and are extracted in place
NETWORK_RUNS_DIR="$NETWORK_DIR/runs"               # results and checkpoints live on the network drive
LOG_DIR="$NETWORK_DIR/logs"                        # one .out file per job, like slurm-<jobid>.out

# All output of the job goes to the log file, as under Slurm (the terminal only shows its path)
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/zod-$(date +%Y%m%d_%H%M%S)-$$.out"
echo "Logging to $LOG_FILE"
exec >> "$LOG_FILE" 2>&1
export PYTHONUNBUFFERED=1   # print to the log as it happens, not in blocks
trap 'echo "=== Job exited with status $? ==="' EXIT

echo "=== Starting ZOD Experiment Job ==="

# --- 2. Smart Sync & Extraction ---
# Sync the project to the local SSD, except the unzipped image folders and runs/ and logs/ on the network drive
# (rsync only copies what changed, so this is cheap when the data is already there)
echo "[1/3] Syncing project from network to local SSD..."
rsync -ah --exclude='train' --exclude='val' --exclude='runs' --exclude='logs' "$NETWORK_DIR" "$LOCAL_ROOT/"

# Extract each zip into images_1333/. The marker file is only written when the extraction finished,
# so a job killed halfway extracts again instead of training on a partial dataset
mkdir -p "$IMAGE_DIR"
for split in train val; do
    if [ ! -f "$IMAGE_DIR/.$split.done" ]; then
        if [ ! -f "$ZIP_DIR/$split.zip" ]; then
            echo "ERROR: $ZIP_DIR/$split.zip not found. Contents of $ZIP_DIR:"; ls -la "$ZIP_DIR"; exit 1
        fi
        echo "[2/3] Extracting $split.zip into $IMAGE_DIR ..."
        rm -rf "${IMAGE_DIR:?}/$split"
        # Quietly, ignoring Mac junk files
        unzip -q "$ZIP_DIR/$split.zip" -d "$IMAGE_DIR" -x "__MACOSX/*"
        [ -d "$IMAGE_DIR/$split" ] || { echo "ERROR: $split.zip did not create $IMAGE_DIR/$split"; exit 1; }
        touch "$IMAGE_DIR/.$split.done"
    else
        echo "[2/3] $split images already extracted."
    fi
done
echo "[3/3] Data ready."

# --- 3. Environment Setup & Requirements ---
# Move into the fast local project directory
cd "$LOCAL_PROJECT_DIR"

# runs/ (metrics, checkpoints) is written to the network drive through a symlink
mkdir -p "$NETWORK_RUNS_DIR"
if [ -e runs ] && [ ! -L runs ]; then
    echo "ERROR: $LOCAL_PROJECT_DIR/runs exists and is not a symlink, move it first"; exit 1
fi
ln -sfn "$NETWORK_RUNS_DIR" runs

# Install/update required packages directly to the VM's global Python environment
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt --quiet
else
    echo "Warning: requirements.txt not found. Skipping package installation."
fi

# Stop early, with a clear message in the log, if PyTorch can't use the GPU (e.g. a build newer than the driver)
python -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| CUDA', torch.version.cuda, '| GPUs', torch.cuda.device_count()); assert torch.cuda.is_available(), 'PyTorch cannot see a GPU: check nvidia-smi and the installed torch build'"

# --- 4. Execution ---
echo "Starting experiment with arguments: $@"

# Execute the run using the global python/torchrun installation
if [[ " $* " == *" --setting centralized "* && " $* " != *" --benchmark "* ]]; then
    torchrun --nproc_per_node=2 zod_experiment.py "$@"
else
    python zod_experiment.py "$@"
fi

echo "=== Job Completed ==="
