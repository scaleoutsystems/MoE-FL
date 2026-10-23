#!/bin/bash
# One-time setup of a fresh Ubuntu VM: tmux, nano, unzip, rsync, and build-essential (gcc, g++, make, libc headers:
# torch.compile needs a C compiler for Triton and a C++ compiler for Inductor).
# Run it before zod_experiment_ai_sweden.sh:
#   bash setup_vm.sh
set -eo pipefail

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

$SUDO apt-get update
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tmux nano unzip rsync build-essential

for cmd in tmux nano unzip rsync gcc g++; do
    command -v "$cmd" > /dev/null || { echo "ERROR: $cmd is still missing"; exit 1; }
done
echo "Done: tmux nano unzip rsync gcc g++ installed"
