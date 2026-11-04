#!/bin/bash
# GenEval images for the Nunchaku SVDQuant INT4/r32 configurations on the A100 (pinned community wheel environment):
# the A100 benchmark config (1024², BF16 encoder) and the GenEval-only copy of the Jetson config (512², NF4 encoder).
# bitsandbytes (NF4) is not in the pinned wheel environment; it is added from a --no-deps overlay on PYTHONPATH so the
# benchmark environment itself is unchanged:
#   /scratch/.../nunchaku-klein-py311-cu128/bin/python -m pip install --no-deps --target $EXTRA bitsandbytes==0.48.2
#   mkdir -p logs && sbatch scripts/gilbreth-geneval-nunchaku.slurm
#SBATCH --job-name=geneval-nunchaku
#SBATCH --account=you139
# standby: preemptible, max 4 h; 40 GB A100 only because the pinned runner asserts that GPU class
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/geneval-nunchaku-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/scratch/gilbreth/$USER/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=${NUNCHAKU_PYTHON:-/scratch/gilbreth/$USER/envs/nunchaku-klein-py311-cu128/bin/python}
export PYTHONPATH=${EXTRA:-/scratch/gilbreth/$USER/envs/nunchaku-geneval-extra}
OUT=/scratch/gilbreth/$USER/geneval/images
log() { echo "$(date -Is) $*" | tee -a logs/geneval-nunchaku-queue.log; }
log "BEGIN job=$SLURM_JOB_ID node=$(hostname)"
for cfg in ${CONFIGS:-configs/a100-nunchaku-flux-klein-int4-r32.json configs/geneval/a100-geneval-nunchaku-flux-klein-int4-r32-512.json}; do
  stem=$(basename "$cfg" .json)
  if "$PY" scripts/geneval_generate_nunchaku.py --config "$cfg" --limit 2 --out-root "$OUT" > logs/geneval-$stem-smoke.log 2>&1; then
    log "SMOKE ok $stem"
    "$PY" scripts/geneval_generate_nunchaku.py --config "$cfg" --out-root "$OUT" > logs/geneval-$stem.log 2>&1
    log "FULL $stem rc=$? $(tail -1 logs/geneval-$stem.log)"
  else
    log "SMOKE FAILED $stem: $(tail -2 logs/geneval-$stem-smoke.log | tr '\n' ' ')"
  fi
done
log "END"
