#!/bin/bash
# GenEval for the Jetson PyTorch configurations, evaluated on the A100: Q4_0 GGUF transformer + NF4 Qwen3 encoder, 512x512.
# 1) quantize the encoder once (scripts/save_quantized_text_encoder.py, receipt beside it); 2) derive cuda configs with its
# hash (scripts/geneval_matched_configs.py); 3) 96-prompt calibration subset per config, 2-prompt smoke first.
#   mkdir -p logs && sbatch scripts/gilbreth-geneval-matched.slurm
#SBATCH --job-name=geneval-matched
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --constraint=G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/geneval-matched-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/scratch/gilbreth/$USER/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/scratch/gilbreth/$USER/envs/torch28-geneval/bin/python
OUT=/scratch/gilbreth/$USER/geneval/images
NF4=/scratch/gilbreth/$USER/models/flux2-klein-torch/text_encoder-nf4
SNAP=$HF_HOME/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294
log() { echo "$(date -Is) $*" | tee -a logs/geneval-matched-queue.log; }
log "BEGIN job=$SLURM_JOB_ID node=$(hostname)"
if [ ! -s "$NF4/model.safetensors" ]; then
  mkdir -p "$(dirname "$NF4")"
  "$PY" scripts/save_quantized_text_encoder.py "$SNAP" "$NF4" --method bnb_nf4 > logs/geneval-matched-nf4.log 2>&1 || { log "NF4 quantization FAILED"; exit 1; }
fi
SHA=$(sha256sum "$NF4/model.safetensors" | cut -d" " -f1)
log "NF4 sha256 $SHA"
[ -s configs/geneval/a100-geneval-torch-flux-klein-q4-nf4-512.json ] || \
  python3 scripts/geneval_matched_configs.py --nf4-sha256 "$SHA" --distilled-gguf-revision 3b1f5a9dc3abb32238b053aeb3d823c30afdacbd >> logs/geneval-matched-queue.log
for stem in ${CONFIGS:-a100-geneval-torch-flux-klein-q4-nf4-512 a100-geneval-torch-flux-klein-q4-nf4-512-condreuse-compile a100-geneval-torch-flux-klein-q4-nf4-512-taylorseer a100-geneval-torch-flux-klein-q4-nf4-512-vaetile a100-geneval-torch-flux-klein-q4-nf4-512-sdpa-math a100-geneval-torch-flux-klein-base-q4-nf4-512 a100-geneval-torch-flux-klein-base-q4-nf4-512-condreuse-compile}; do
  if "$PY" scripts/geneval_generate.py --config configs/geneval/$stem.json --limit 2 --out-root "$OUT" > logs/geneval-$stem-smoke.log 2>&1; then
    log "SMOKE ok $stem"
    "$PY" scripts/geneval_generate.py --config configs/geneval/$stem.json --out-root "$OUT" > logs/geneval-$stem.log 2>&1
    log "FULL $stem rc=$? $(tail -1 logs/geneval-$stem.log)"
  else
    log "SMOKE FAILED $stem: $(tail -2 logs/geneval-$stem-smoke.log | tr '\n' ' ')"
  fi
done
log "END"
