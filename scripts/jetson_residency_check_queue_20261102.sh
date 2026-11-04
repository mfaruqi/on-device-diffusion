#!/usr/bin/env bash
# 2026-11-02 queue O: full-residency feasibility checks with tiled VAE decode (128 px / 0.25), one engine at a time.
#   1. PyTorch GGUF Q4_0 + NF4, encoder and transformer both resident (jetson-torch-flux-klein-009, vs 003)
#   2. Nunchaku INT4 r32 + NF4, both resident (jetson-nunchaku-flux-klein-003, vs 009: transformer checkpoint only)
#   3. sd.cpp all weights on CUDA0 (jetson-flux-klein-012 repeat attempt; the 1+3+10 run failed at generation index 4)
# Each check is 1 first + 1 warm-up + 4 measured generations; checks are independent, so a failure does not stop the queue.
# Usage (on the Jetson, no root; desktop session logged out):
#   setsid nohup bash scripts/jetson_residency_check_queue_20261102.sh > ~/results/residency-20261102.out 2>&1 < /dev/null &
# About 20 minutes. Every job runs under jetson_memguard.sh; guard thresholds are logged per job and are not part of any config.
set -uo pipefail
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
(( avail >= 5700 )) || { echo "MemAvailable ${avail} MB < 5700 MB: stop other jobs and log out of the desktop session first"; exit 1; }
cd "$(dirname "$0")/.."
REPO=$PWD
RUNS=$REPO/results/runs
Q=$HOME/results/residency-20261102
LOG=$Q/queue.log
mkdir -p "$Q"
SYS_PY=/usr/bin/python3
TORCH_PY=$HOME/venvs/torch-jp6-ailab/bin/python
SDBENCH=$HOME/tools/sd-bench-build-tiling-20261014/sd-bench
export HF_HOME=$HOME/hf HF_HUB_OFFLINE=1 PATH=/usr/local/cuda/bin:$PATH
C=configs

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

# run_job <label> <swap-growth-limit-MB> <command...>; returns 0 only if the newest run directory says complete.
run_job() {
  local label=$1 swap=$2; shift 2
  local before; before=$(ls -t "$RUNS" | head -1)
  log "START $label (MAX_SWAP_GROWTH_MB=$swap) :: $*"
  log "free: $(free -m | awk 'NR==2{print "used="$3" avail="$7} NR==3{print "swap="$3}' | tr '\n' ' ')"
  MAX_SWAP_GROWTH_MB=$swap bash scripts/jetson_memguard.sh "$Q/guard-$label" "$@" > "$Q/$label.log" 2>&1
  local rc=$?
  local dir; dir=$(ls -t "$RUNS" | head -1)
  [[ "$dir" == "$before" ]] && dir="(no new run directory)"
  cp "$Q/guard-$label/memguard.json" "$RUNS/$dir/memguard.json" 2>/dev/null
  local status; status=$($SYS_PY -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$RUNS/$dir/status.json" 2>/dev/null || echo unknown)
  log "END $label rc=$rc status=$status dir=$dir guard=$(tr -d '\n' < "$Q/guard-$label/memguard.json" 2>/dev/null | cut -c1-160)"
  sleep 30
  [[ $rc -eq 0 && "$status" == complete ]]
}

PT=( "$TORCH_PY" scripts/run_flux_portable.py --out-root "$RUNS" )
SD=( "$SYS_PY" scripts/run_jetson_sdcpp.py --binary "$SDBENCH" --out-root "$RUNS" )

log "QUEUE-O BEGIN $(nvpmodel -q 2>/dev/null | head -1) MemAvailable=${avail}MB"
run_job torch-009 1536 "${PT[@]}" --config "$C/jetson-torch-flux-klein-q4-nf4-512-resident-vaetile-check.json" --kind attempt
run_job nunchaku-003 1536 "${PT[@]}" --config "$C/jetson-nunchaku-flux-klein-int4-r32-512-resident-vaetile-check.json" --kind attempt
run_job sdcpp-012-check 768 "${SD[@]}" --config "$C/jetson-flux-klein-q4-512-resident-vaetile-check.json" --kind attempt
log "QUEUE-O END"
