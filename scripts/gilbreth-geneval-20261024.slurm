#!/bin/bash
# GenEval calibration images (frozen split configs/geneval/split-20261024.json, 96 prompts x 4 seeds) for the
# A100 PyTorch torch 2.8 distilled configurations: BF16 reference, torch.compile, TaylorSeer. Each config runs a
# 2-prompt smoke first and continues only if it completed. Scoring runs separately (scripts/geneval_score.slurm).
#
#   mkdir -p logs && sbatch scripts/gilbreth-geneval-20261024.slurm
#
#SBATCH --job-name=geneval-gen
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --constraint=G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/geneval-gen-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/scratch/gilbreth/$USER/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/scratch/gilbreth/$USER/envs/torch28-trt/bin/python
OUT=/scratch/gilbreth/$USER/geneval/images
log() { echo "$(date -Is) $*" | tee -a logs/geneval-gen-queue.log; }
log "BEGIN job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
for stem in ${CONFIGS:-a100-flux-klein-torch28-bf16 a100-flux-klein-torch28-bf16-compile a100-flux-klein-torch28-bf16-taylorseer}; do
  if "$PY" scripts/geneval_generate.py --config configs/$stem.json --limit 2 --out-root "$OUT" > logs/geneval-$stem-smoke.log 2>&1; then
    log "SMOKE ok $stem"
    "$PY" scripts/geneval_generate.py --config configs/$stem.json --out-root "$OUT" > logs/geneval-$stem.log 2>&1
    log "FULL $stem rc=$? $(tail -1 logs/geneval-$stem.log)"
  else
    log "SMOKE FAILED $stem: $(tail -2 logs/geneval-$stem-smoke.log | tr '\n' ' ')"
  fi
done
log "END"
