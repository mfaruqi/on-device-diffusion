#!/bin/bash
# One-image functional check of stable-diffusion.cpp (sd-cli) with FLUX.2 [klein] 4B on one A100.
# Not a benchmark: it verifies that the pinned weights load, prints sd.cpp's verbose log
# (tensor types, placement, sampler, attention) for the settings audit, and saves one image.
#
#   mkdir -p logs && sbatch scripts/gilbreth-sdcpp-smoke.slurm
#
#SBATCH --job-name=sdcpp-smoke
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
#SBATCH --time=00:20:00
#SBATCH --output=logs/sdcpp-smoke-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SD_CLI="${SD_CLI:-$HOME/tools/stable-diffusion.cpp/build/bin/sd-cli}"
# Pinned FLUX.2-klein-4B snapshot (same commit as configs/a100-flux-klein-bf16.resolved.json).
SNAP="${SNAP:-/scratch/gilbreth/$USER/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294}"
OUT="${OUT:-logs/sdcpp-smoke-$SLURM_JOB_ID.png}"

echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
nvidia-smi
git -C "$(dirname "$SD_CLI")/../.." log -1 --format='sd.cpp commit %H'

# Reference settings: everything on cuda0, no auto-fit, no graph segmentation, eager weight
# load, no conditioning cache, no offload/tiling/step cache. Flash attention in the diffusion
# model only, matching the PyTorch run's SDPA-flash transformer.
"$SD_CLI" \
  --diffusion-model "$SNAP/flux-2-klein-4b.safetensors" \
  --llm "$SNAP/text_encoder" \
  --vae "$SNAP/vae/diffusion_pytorch_model.safetensors" \
  -p "A cat holding a sign that says hello world" \
  -W 1024 -H 1024 --steps 4 --cfg-scale 1.0 --seed 0 \
  --backend cuda0 --params-backend cuda0 --auto-fit off \
  --disable-segmented-compute --eager-load --conditioning-cache-size 0 \
  --diffusion-fa \
  -v -o "$OUT"

echo "image=$OUT end=$(date -Is)"
