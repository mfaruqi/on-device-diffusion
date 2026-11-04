#!/bin/bash
# Nunchaku single-layer W4A4 check on an A100 (scripts/nunchaku_layer_check.py run). Its outputs.safetensors is the
# reference for the same check on other devices. Submit from the repository root:
#   sbatch --export=ALL,BUNDLE=/scratch/.../layer-bundle.safetensors scripts/nunchaku_layer_check.slurm
#SBATCH --job-name=nunchaku-layer
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/nunchaku-layer-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${BUNDLE:?Set BUNDLE to the layer bundle written by nunchaku_layer_check.py extract}"
PY=${NUNCHAKU_PYTHON:-/scratch/gilbreth/mfaruqi/envs/nunchaku-klein-py311-cu128/bin/python}
OUT=results/runs/a100-nunchaku-layer-check-001__layer__$(date -u +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu.txt"
"$PY" scripts/nunchaku_layer_check.py run --bundle "$BUNDLE" --out "$OUT" 2>&1 | tee "$OUT/stdout.txt"
