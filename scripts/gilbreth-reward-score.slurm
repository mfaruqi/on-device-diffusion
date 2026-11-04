#!/bin/bash
# Score GenEval image folders with CLIP, PickScore, HPS v2.1 and ImageReward (scripts/reward_score.py).
#   IMAGE_DIRS="<run>/images <run>/images" sbatch scripts/gilbreth-reward-score.slurm
# Writes <run>/reward-scores.jsonl and <run>/reward-summary.json next to each image folder; already scored runs are skipped.
# Run `python scripts/reward_score.py --prepare-only` on the login node first (it downloads and pins the models).
#SBATCH --job-name=reward-score
#SBATCH --account=you139
# standby: preemptible, max 4 h, any of these GPU types (scoring does not depend on the GPU model)
#SBATCH --partition=a100-80gb,a100-40gb,a30,a10
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/reward-score-%j.out
set -euo pipefail
# HF_HUB_OFFLINE: the models were downloaded by --prepare-only; the job never touches the network
export HF_HOME=/scratch/gilbreth/$USER/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=${PY:-/scratch/gilbreth/$USER/envs/quality-scores/bin/python}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
$PY scripts/reward_score.py --images ${IMAGE_DIRS:?set IMAGE_DIRS}
