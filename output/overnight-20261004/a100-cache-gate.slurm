#!/bin/bash
# FLUX.2 [klein] with stable-diffusion.cpp on one Gilbreth A100: same protocol as gilbreth.slurm.
# Submit from the repository root:
#
#   mkdir -p logs
#   sbatch scripts/gilbreth-sdcpp.slurm                         # baseline
#   sbatch --export=ALL,PROFILE=1 scripts/gilbreth-sdcpp.slurm  # separate Nsight Systems run
#
# Requires the sd.cpp build and the harness build (wiki/methods/sdcpp-howto.md) and the resolved config
# from `run_sdcpp.py --prepare-only`.
#
#SBATCH --job-name=base-cache-control
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
#SBATCH --output=logs/base-cache-gate-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# The wrapper only needs the standard library, huggingface_hub (for the cache
# path) and Pillow (to save two PNGs); the flux env has all of them. It never imports torch.
FLUX_ENV="${FLUX_ENV:-$HOME/.conda/envs/2025.06-py313/flux_env}"
PYTHON="$FLUX_ENV/bin/python"
# nvcc / nsys from the same CUDA toolkit the harness was built with.
SDCPP_CUDA_ROOT="${SDCPP_CUDA_ROOT:-/apps/spack/gilbreth-r9/apps/cuda/12.6.0-gcc-11.5.0-a7cv7sp}"
export PATH="$SDCPP_CUDA_ROOT/bin:$PATH"
# PATH finds executables; the dynamic loader separately needs the toolkit libraries.
export LD_LIBRARY_PATH="$SDCPP_CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HOME="${HF_HOME:-/scratch/gilbreth/$USER/huggingface}"
export HF_HUB_OFFLINE=1
unset GGML_CUDA_CUBLAS_COMPUTE_TYPE

CONFIG="${CONFIG:-configs/a100-sdcpp-flux-klein-base-cache-control-smoke.resolved.json}"
EXTRA_ARGS=()
if [[ "${PROFILE:-0}" == "1" ]]; then EXTRA_ARGS+=(--nsys); fi

echo "job=$SLURM_JOB_ID node=$(hostname) partition=${SLURM_JOB_PARTITION:-?} start=$(date -Is)"
echo "python=$PYTHON HF_HOME=$HF_HOME config=$CONFIG args=${EXTRA_ARGS[*]:-}"
nvidia-smi

"$PYTHON" scripts/run_sdcpp.py --config "$CONFIG" --kind attempt "${EXTRA_ARGS[@]}"

echo "end=$(date -Is)"
