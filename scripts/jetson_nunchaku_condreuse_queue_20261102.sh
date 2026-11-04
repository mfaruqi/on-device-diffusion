#!/usr/bin/env bash
# 2026-11-02 queue O: jetson-nunchaku-flux-klein-002, the Nunchaku SVDQuant configuration of queue N
# (jetson_nunchaku_queue_20261102.sh) with exact conditioning reuse. Smoke (first + one measured), then the full 1+3+10
# protocol if the smoke completed. The kernels and source patch are those queue N checked; the runner re-verifies the
# patch hash at load.
# Usage (on the Jetson, no root; desktop session logged out):
#   setsid nohup bash scripts/jetson_nunchaku_condreuse_queue_20261102.sh > ~/results/nunchaku-condreuse-20261102.out 2>&1 < /dev/null &
# About 10 minutes. One job at a time; every job runs under jetson_memguard.sh.
set -uo pipefail
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
(( avail >= 5700 )) || { echo "MemAvailable ${avail} MB < 5700 MB: log out of the desktop session first"; exit 1; }
cd "$(dirname "$0")/.."
REPO=$PWD
RUNS=$REPO/results/runs
Q=$HOME/results/nunchaku-condreuse-20261102
LOG=$Q/queue.log
mkdir -p "$Q"
SYS_PY=/usr/bin/python3
TORCH_PY=$HOME/venvs/torch-jp6-ailab/bin/python
export HF_HOME=$HOME/hf HF_HUB_OFFLINE=1 PATH=/usr/local/cuda/bin:$PATH
C=configs
N=jetson-nunchaku-flux-klein-int4-r32-512-condreuse

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

log "QUEUE-O BEGIN $(nvpmodel -q 2>/dev/null | head -1) MemAvailable=${avail}MB"
PT=( "$TORCH_PY" scripts/run_flux_portable.py --out-root "$RUNS" )
if run_job condreuse-smoke 1536 "${PT[@]}" --config "$C/$N-smoke.json" --kind attempt; then
  run_job condreuse-full 1536 "${PT[@]}" --config "$C/$N.json" --kind baseline
else
  log "SKIP condreuse-full: smoke did not complete"
fi
log "QUEUE-O END"
