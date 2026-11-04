#!/bin/bash
# 2026-10-15 A100 queue (gap campaign): PyTorch torch 2.8 reference, torch.compile, conditioning reuse and
# diffusers step caches; then edge-dit step-cache screens. One A100-PCIE-40GB allocation, one job at a time.
# Each new config runs a smoke first and continues to the full 1+3+10 protocol only if the smoke completed.
# edge-dit caches: full protocol only when the smoke skipped any step (screen -> confirm).
#
#   mkdir -p logs && sbatch scripts/gilbreth-gap-20261015.slurm
#
#SBATCH --job-name=gap-20261015
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --constraint=G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gap-20261015-%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
LOG=logs/gap-20261015-queue.log
log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
export HF_HOME=/scratch/gilbreth/$USER/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TORCH_PY=/scratch/gilbreth/$USER/envs/torch28-trt/bin/python
EDGE_PY=$HOME/.conda/envs/2025.06-py313/flux_env/bin/python
EDGE_CUDA=/apps/spack/gilbreth-r9/apps/cuda/12.6.0-gcc-11.5.0-a7cv7sp
EDGE_NVIDIA=$HOME/.conda/envs/2025.06-py313/flux_env/lib/python3.10/site-packages/nvidia
RUNS=results/runs
C=configs

job() {  # job <label> <command...>; true only if the newest run directory is complete
  local label=$1; shift
  local before; before=$(ls -t $RUNS 2>/dev/null | head -1)
  log "START $label :: $*"
  "$@" > "logs/gap-20261015-$label.log" 2>&1
  local rc=$? dir; dir=$(ls -t $RUNS | head -1); [[ "$dir" == "$before" ]] && dir="(none)"
  local status; status=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$RUNS/$dir/status.json" 2>/dev/null || echo unknown)
  log "END $label rc=$rc status=$status dir=$dir"
  [[ $rc -eq 0 && $status == complete ]]
}
pt() {  # pt <label> <config-stem>
  if job "$1-smoke" "$TORCH_PY" scripts/run_flux_portable.py --config "$C/$2-smoke.json" --kind attempt; then
    job "$1-full" "$TORCH_PY" scripts/run_flux_portable.py --config "$C/$2.json" --kind baseline
  else log "SKIP $1-full: smoke did not complete"; fi
}
ed() {  # ed <method>
  local m=$1
  if job "edgedit-$m-smoke" env LD_LIBRARY_PATH="$EDGE_NVIDIA/cudnn/lib:$EDGE_NVIDIA/cublas/lib:$EDGE_NVIDIA/cuda_nvrtc/lib:$EDGE_CUDA/lib64" \
      "$EDGE_PY" scripts/run_edgedit.py --config "$C/a100-edgedit-flux-klein-bf16-$m-smoke.json" --kind attempt; then
    local dir; dir=$(ls -td $RUNS/a100-edgedit-flux-klein-*__attempt__* | head -1)
    local skipped; skipped=$(python3 -c "import json,sys;s=json.load(open(sys.argv[1]))['skipped_steps_per_generation'];print('unknown' if s is None else sum(s))" "$dir/step-cache.json" 2>/dev/null || echo unknown)
    if [[ "$skipped" == 0 ]]; then log "SCREEN $m: 0 skipped steps; full protocol not run"
    else log "SCREEN $m: skipped=$skipped; running full"
      job "edgedit-$m-full" env LD_LIBRARY_PATH="$EDGE_NVIDIA/cudnn/lib:$EDGE_NVIDIA/cublas/lib:$EDGE_NVIDIA/cuda_nvrtc/lib:$EDGE_CUDA/lib64" \
        "$EDGE_PY" scripts/run_edgedit.py --config "$C/a100-edgedit-flux-klein-bf16-$m.json" --kind baseline
    fi
  else log "SKIP edgedit-$m-full: smoke did not complete"; fi
}

log "QUEUE BEGIN job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader)"
unset GGML_CUDA_CUBLAS_COMPUTE_TYPE
pt torch28-001 a100-flux-klein-torch28-bf16
pt compile-001 a100-flux-klein-torch28-bf16-compile
pt torch28-002 a100-flux-klein-torch28-bf16-condreuse
pt torch28-003 a100-flux-klein-torch28-bf16-fbcache
pt torch28-004 a100-flux-klein-torch28-bf16-taylorseer
for m in easycache ucache dbcache taylorseer cache-dit magcache dicache sencache; do ed $m; done
log "QUEUE END"
