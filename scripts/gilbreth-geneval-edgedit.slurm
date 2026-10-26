#!/bin/bash
# GenEval images for A100 edge-dit.cpp Base (GGUF Q4_0 / Q4_K_M / BF16 VAE, 1024², 50 steps, CFG 4): reference and
# step caches, each from its completed benchmark run's exact engine command (scripts/geneval_generate_edgedit.py).
# 96-prompt calibration subset x 4 images; a 2-prompt smoke first per configuration.
#   mkdir -p logs && sbatch scripts/gilbreth-geneval-edgedit.slurm
#SBATCH --job-name=geneval-edgedit
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --constraint=G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/geneval-edgedit-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/scratch/gilbreth/$USER/huggingface HF_HUB_OFFLINE=1
EDGE_PY=$HOME/.conda/envs/2025.06-py313/flux_env/bin/python
EDGE_CUDA=/apps/spack/gilbreth-r9/apps/cuda/12.6.0-gcc-11.5.0-a7cv7sp
EDGE_NVIDIA=$HOME/.conda/envs/2025.06-py313/flux_env/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$EDGE_NVIDIA/cudnn/lib:$EDGE_NVIDIA/cublas/lib:$EDGE_NVIDIA/cuda_nvrtc/lib:$EDGE_CUDA/lib64"
OUT=/scratch/gilbreth/$USER/geneval/images
R=results/runs
log() { echo "$(date -Is) $*" | tee -a logs/geneval-edgedit-queue.log; }
log "BEGIN job=$SLURM_JOB_ID node=$(hostname)"
for run in ${TEMPLATES:-a100-edgedit-flux-klein-base-001__baseline__20261015-165052 a100-edgedit-flux-klein-base-005__baseline__20261015-173406 a100-edgedit-flux-klein-base-008__baseline__20261015-175727 a100-edgedit-flux-klein-base-002__baseline__20261015-170356 a100-edgedit-flux-klein-base-007__baseline__20261015-175238}; do
  id=${run%%__*}
  if "$EDGE_PY" scripts/geneval_generate_edgedit.py --template-run $R/$run --limit 2 --out-root "$OUT" > logs/geneval-$id-smoke.log 2>&1; then
    log "SMOKE ok $id"
    "$EDGE_PY" scripts/geneval_generate_edgedit.py --template-run $R/$run --out-root "$OUT" > logs/geneval-$id.log 2>&1
    log "FULL $id rc=$? $(tail -1 logs/geneval-$id.log)"
  else
    log "SMOKE FAILED $id: $(tail -2 logs/geneval-$id-smoke.log | tr '\n' ' ')"
  fi
done
log "END"
