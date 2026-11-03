#!/bin/bash
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gilbreth-n012
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --job-name=upstream-fused-check
#SBATCH --output=/scratch/gilbreth/mfaruqi/shiftadd-initial/upstream-fused-%j.log
set -euo pipefail
# Run offline on an allocated GPU node, never on the login node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/scratch/gilbreth/mfaruqi/shiftadd-initial/upstream-image-stage-001/src
/home/mfaruqi/.conda/envs/2025.06-py313/flux_env/bin/python -m shiftadd_flux.pipeline_smoke --config /scratch/gilbreth/mfaruqi/shiftadd-initial/configs/a100-upstream-fused-001.json
