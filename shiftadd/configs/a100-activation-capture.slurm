#!/bin/bash
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --job-name=shiftadd-activations
#SBATCH --output=/scratch/gilbreth/mfaruqi/shiftadd-initial/activations-%j.log
set -euo pipefail
# Run offline on an allocated GPU node, never on the login node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/scratch/gilbreth/mfaruqi/shiftadd-initial/src-activations
/home/mfaruqi/.conda/envs/2025.06-py313/flux_env/bin/python -m shiftadd_flux.capture_pipeline --config /scratch/gilbreth/mfaruqi/shiftadd-initial/configs/a100-activation-capture-001.json
