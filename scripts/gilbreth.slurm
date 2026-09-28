#!/bin/bash
# FLUX.2 [klein] baseline on one Gilbreth A100. Submit from the repository root:
#
#   mkdir -p logs
#   sbatch scripts/gilbreth.slurm                         # baseline
#   sbatch --export=ALL,PROFILE=1 scripts/gilbreth.slurm  # baseline + profiler run
#
# Account/partition below are the defaults for this project (see `slist`);
# command-line flags override them, e.g. --partition=a100-80gb. Do not mix
# 40 GB and 80 GB results under one device label.
#
#SBATCH --job-name=flux-baseline
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
# The a100-40gb partition mixes A100-PCIE-40GB nodes (feature G, gilbreth-g*) and A100-SXM4-40GB
# nodes (feature N, gilbreth-n*, higher power limit). Pin to G so results stay comparable.
#SBATCH --constraint=G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/flux-baseline-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Python environment: call its interpreter directly so no activation is needed.
FLUX_ENV="${FLUX_ENV:-$HOME/.conda/envs/2025.06-py313/flux_env}"
PYTHON="$FLUX_ENV/bin/python"

# Model cache on scratch (populated earlier by --prepare-only). Offline: never download mid-benchmark.
export HF_HOME="${HF_HOME:-/scratch/gilbreth/$USER/huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CONFIG="${CONFIG:-configs/a100-flux-klein-bf16.resolved.json}"
EXTRA_ARGS=()
if [[ "${PROFILE:-0}" == "1" ]]; then EXTRA_ARGS+=(--profile); fi

echo "job=$SLURM_JOB_ID node=$(hostname) partition=${SLURM_JOB_PARTITION:-?} start=$(date -Is)"
echo "python=$PYTHON HF_HOME=$HF_HOME config=$CONFIG args=${EXTRA_ARGS[*]:-}"
nvidia-smi
"$PYTHON" -c 'import torch, diffusers; from diffusers import Flux2KleinPipeline; print("torch", torch.__version__, "cuda", torch.version.cuda, "diffusers", diffusers.__version__, "gpu", torch.cuda.get_device_name(0))'

"$PYTHON" scripts/run_flux.py --config "$CONFIG" "${EXTRA_ARGS[@]}"

echo "end=$(date -Is)"
