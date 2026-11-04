#!/usr/bin/env bash
# 2026-11-02 queue P: sd.cpp all weights on CUDA0 with tiled VAE decode (jetson-flux-klein-012), full 1+3+10 protocol.
# Repeat of the 2026-10-15 full run, which failed at generation index 4; the 1+1+4 check of queue O completed.
# Conditioning cache off (conditioning-cache-size 0). No smoke: queue O's check is this config's smoke.
# Usage (on the Jetson, no root; desktop session logged out):
#   setsid nohup bash scripts/jetson_sdcpp_resident_full_queue_20261102.sh > ~/results/sdcpp-resident-20261102.out 2>&1 < /dev/null &
# About 6 minutes. Every job runs under jetson_memguard.sh; guard thresholds are logged per job and are not part of any config.
set -uo pipefail
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
(( avail >= 5700 )) || { echo "MemAvailable ${avail} MB < 5700 MB: stop other jobs and log out of the desktop session first"; exit 1; }
cd "$(dirname "$0")/.."
REPO=$PWD
RUNS=$REPO/results/runs
Q=$HOME/results/sdcpp-resident-20261102
LOG=$Q/queue.log
mkdir -p "$Q"
SYS_PY=/usr/bin/python3
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

SD=( "$SYS_PY" scripts/run_jetson_sdcpp.py --binary "$SDBENCH" --out-root "$RUNS" )

log "QUEUE-P BEGIN $(nvpmodel -q 2>/dev/null | head -1) MemAvailable=${avail}MB"
run_job sdcpp-012-full 768 "${SD[@]}" --config "$C/jetson-flux-klein-q4-512-resident-vaetile.json" --kind baseline
log "QUEUE-P END"
