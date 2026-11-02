#!/usr/bin/env bash
# 2026-10-31: rerun of jetson_counters_queue_20261030.sh with kernel replay (application replay fails on this board
# beyond gpu__time_duration; scripts/jetson_ncu_probe3.sh). A plan given as <name>:<run dir> resumes that capture:
# its sequence pass is reused and only the counter window runs (the failed window's log is kept as window-application.log).
# Original header: Nsight Compute memory-traffic captures of one steady denoising pass (distilled klein 512², 4 steps) for
# the Jetson engines: sd.cpp (003), edge-dit (006), PyTorch eager (004) and torch.compile (007). Bytes are L2 <-> system
# memory sectors (validated by jetson_ncu_probe2.sh); see scripts/profile_engine_counters.py. Nsight Compute needs root
# on this board. Refuses to start unless MemAvailable >= 5700 MB (log out of the desktop session first). One capture at
# a time, each under jetson_memguard.sh; a failed capture keeps its run directory and the queue moves on. Run
# directories are chowned back to the invoking user at the end. Expect 2-4 hours.
# Usage (on the Jetson):  sudo -v && sudo setsid nohup bash scripts/jetson_counters_queue_20261031.sh > ~/results/counters-20261031/queue.out 2>&1 < /dev/null &
# Optional arguments: plan names or <name>:<run dir> (default: the two resumes below, then torch-007 and edge-dit).
set -uo pipefail
[[ $EUID -eq 0 && -n "${SUDO_USER:-}" ]] || { echo "run with sudo (Nsight Compute needs root on this board)"; exit 1; }
export HOME=/home/$SUDO_USER
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
(( avail >= 5700 )) || { echo "MemAvailable ${avail} MB < 5700 MB: log out of the desktop session first"; exit 1; }
cd "$(dirname "$0")/.."
RUNS=$PWD/results/runs
OUT=$HOME/results/counters-20261031
mkdir -p "$OUT"
LOG=$OUT/queue.log
SYS_PY=/usr/bin/python3
export HF_HOME=$HOME/hf HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=1

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

capture() {
  local name=${1%%:*} resume=() label
  local plan=configs/ncu/jetson-$name-counters.json; label=$(basename "$plan" .json)
  local before; before=$(ls -t "$RUNS" | head -1)
  if [[ "$1" == *:* ]]; then
    resume=(--resume "$RUNS/${1#*:}")
    [[ -f "$RUNS/${1#*:}/profile/window.log" ]] && mv "$RUNS/${1#*:}/profile/window.log" "$RUNS/${1#*:}/profile/window-application.log"
  fi
  log "START $label ${resume[*]} free: $(free -m | awk 'NR==2{print "avail="$7} NR==3{print "swap="$3}' | tr '\n' ' ')"
  MAX_SWAP_GROWTH_MB=1536 bash scripts/jetson_memguard.sh "$OUT/guard-$label" "$SYS_PY" scripts/profile_engine_counters.py --plan "$plan" \
    "${resume[@]}" > "$OUT/$label.log" 2>&1
  local rc=$?
  local dir; dir=$(ls -t "$RUNS" | head -1)
  [[ "$dir" == "$before" ]] && dir="(no new run directory)"
  (( ${#resume[@]} )) && dir=${1#*:}
  cp "$OUT/guard-$label/memguard.json" "$RUNS/$dir/memguard.json" 2>/dev/null
  log "END $label rc=$rc dir=$dir status=$(tr -d '\n' < "$RUNS/$dir/status.json" 2>/dev/null | cut -c1-200)"
  sleep 30
}

log "QUEUE BEGIN $(nvpmodel -q 2>/dev/null | head -1) MemAvailable=${avail}MB"
PLANS=("$@")
(( ${#PLANS[@]} )) || PLANS=(sdcpp-flux-klein-003:jetson-flux-klein-003__counters__20261030-211513
  torch-flux-klein-004:jetson-torch-flux-klein-004__counters__20261031-002432 torch-flux-klein-007 edgedit-flux-klein-006)
for p in "${PLANS[@]}"; do
  capture "$p"
done
chown -R "$SUDO_USER:$SUDO_USER" "$RUNS" "$OUT" "$HOME/.cache" 2>/dev/null
log "QUEUE END"
