#!/bin/bash
#SBATCH --job-name=edge-sample-smoke
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --constraint=G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=logs/edge-sample-smoke-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
EDGE_CUDA=/apps/spack/gilbreth-r9/apps/cuda/12.6.0-gcc-11.5.0-a7cv7sp
EDGE_NVIDIA=$HOME/.conda/envs/2025.06-py313/flux_env/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$EDGE_NVIDIA/cudnn/lib:$EDGE_NVIDIA/cublas/lib:$EDGE_NVIDIA/cuda_nvrtc/lib:$EDGE_CUDA/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
unset GGML_CUDA_CUBLAS_COMPUTE_TYPE
# This gate emits functional evidence only; no startup-inclusive time is called generation latency.
"$HOME/.conda/envs/2025.06-py313/flux_env/bin/python" edge-sample-smoke.py
