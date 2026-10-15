#!/bin/bash
# Candidate Nunchaku timing: submit from repo root after preparing pinned inputs.
# NUNCHAKU_PYTHON must name the separate Python 3.11/cu128/Torch 2.11 environment.
#SBATCH --job-name=klein-nunchaku
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
#SBATCH --constraint=G
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/klein-nunchaku-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${NUNCHAKU_PYTHON:?Set the interpreter for the dedicated Nunchaku environment}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/$USER/huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
CONFIG="${CONFIG:-configs/a100-nunchaku-flux-klein-int4-r32-smoke.json}"
EXTRA=()
if [[ -n "${SMOKE_RUN:-}" ]]; then EXTRA+=(--smoke-run "$SMOKE_RUN"); fi
if [[ "${RUN_FULL_AFTER_SMOKE:-0}" == "1" ]]; then
    RECEIPT="logs/nunchaku-smoke-${SLURM_JOB_ID}.path"
    EXTRA+=(--run-dir-file "$RECEIPT")
fi
"$NUNCHAKU_PYTHON" scripts/run_nunchaku.py --device a100 --config "$CONFIG" "${EXTRA[@]}"
if [[ "${RUN_FULL_AFTER_SMOKE:-0}" == "1" ]]; then
    # set -e stops here on any smoke failure. The runner checks the matching smoke
    # status, quantization audit and nonflat images before starting the full load.
    "$NUNCHAKU_PYTHON" scripts/run_nunchaku.py --device a100 \
        --config configs/a100-nunchaku-flux-klein-int4-r32.json \
        --smoke-run "$(cat "$RECEIPT")"
fi
