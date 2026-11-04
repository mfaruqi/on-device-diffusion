#!/bin/bash
#SBATCH --job-name=klein-a100-counters
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --constraint=G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/a100-counters-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the isolated capture directory}"
# Same pinned software environment and offline model cache as the saved reference.
export HF_HOME=/scratch/gilbreth/mfaruqi/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PYTHON=/scratch/gilbreth/mfaruqi/envs/torch28-trt/bin/python
# Slurm reserves one PCIe A100 compute node; no inference runs on the login node.
exec "$PYTHON" scripts/profile_a100_counters.py --plan configs/a100-flux-klein-torch28-ncu.json
