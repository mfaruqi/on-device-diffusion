#!/usr/bin/env bash
# Rerun of the sd.cpp capture from jetson_counters_queue_20261030.sh, whose first attempt failed before profiling
# (run_jetson_sdcpp.py requires harness_arguments.runs to match the 1 + 0 + 1 protocol; fixed in
# profile_engine_counters.py). Same steps as that queue, for the one plan; waits until the first queue has finished,
# so one job runs at a time on the board.
# Usage (on the Jetson):  sudo -v && sudo setsid nohup bash scripts/jetson_counters_queue_20261030b.sh > ~/results/counters-20261030/queue-b.out 2>&1 < /dev/null &
set -uo pipefail
while pgrep -f "bash scripts/jetson_counters_queue_20261030.sh" > /dev/null; do sleep 60; done
set -- sdcpp-flux-klein-003
[[ $EUID -eq 0 && -n "${SUDO_USER:-}" ]] || { echo "run with sudo (Nsight Compute needs root on this board)"; exit 1; }
export HOME=/home/$SUDO_USER
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
(( avail >= 5700 )) || { echo "MemAvailable ${avail} MB < 5700 MB: log out of the desktop session first"; exit 1; }
cd "$(dirname "$0")/.."
RUNS=$PWD/results/runs
OUT=$HOME/results/counters-20261030
mkdir -p "$OUT"
LOG=$OUT/queue-b.log
SYS_PY=/usr/bin/python3
export HF_HOME=$HOME/hf HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=1

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

capture() {
  local plan=$1 label; label=$(basename "$plan" .json)
  local before; before=$(ls -t "$RUNS" | head -1)
  log "START $label free: $(free -m | awk 'NR==2{print "avail="$7} NR==3{print "swap="$3}' | tr '\n' ' ')"
  MAX_SWAP_GROWTH_MB=1536 bash scripts/jetson_memguard.sh "$OUT/guard-$label" "$SYS_PY" scripts/profile_engine_counters.py --plan "$plan" \
    > "$OUT/$label.log" 2>&1
  local rc=$?
  local dir; dir=$(ls -t "$RUNS" | head -1)
  [[ "$dir" == "$before" ]] && dir="(no new run directory)"
  cp "$OUT/guard-$label/memguard.json" "$RUNS/$dir/memguard.json" 2>/dev/null
  log "END $label rc=$rc dir=$dir status=$(tr -d '\n' < "$RUNS/$dir/status.json" 2>/dev/null | cut -c1-200)"
  sleep 30
}

log "QUEUE BEGIN $(nvpmodel -q 2>/dev/null | head -1) MemAvailable=${avail}MB"
PLANS=("$@")
(( ${#PLANS[@]} )) || PLANS=(sdcpp-flux-klein-003 edgedit-flux-klein-006 torch-flux-klein-004 torch-flux-klein-007)
for p in "${PLANS[@]}"; do
  capture configs/ncu/jetson-$p-counters.json
done
chown -R "$SUDO_USER:$SUDO_USER" "$RUNS" "$OUT" "$HOME/.cache" 2>/dev/null
log "QUEUE END"
