#!/usr/bin/env bash
# Sequential Jetson queue for the 2026-10-14 engine-matching runs (gap campaign).
# Usage (on the Jetson, from the repo copy):  nohup bash scripts/jetson_gap_queue_20261014.sh > ~/results/gap-20261014/queue.out 2>&1 &
# One job at a time; nothing else should run on the board while it is going. Each new config runs as a
# smoke (first + one measured) and continues to the full 1+3+10 protocol only if the smoke completed.
# Every job runs under jetson_memguard.sh; guard thresholds are logged per job and are not part of any config.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
RUNS=$REPO/results/runs
LOG=$HOME/results/gap-20261014/queue.log
mkdir -p "$(dirname "$LOG")"
SYS_PY=/usr/bin/python3
TORCH_PY=$HOME/venvs/torch-jp6-ailab/bin/python
SDBENCH=$HOME/tools/sd-bench-build-tiling-20261014/sd-bench
export HF_HOME=$HOME/hf HF_HUB_OFFLINE=1

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

# run_job <label> <swap-growth-limit-MB> <command...>; returns 0 only if the newest run directory says complete.
run_job() {
  local label=$1 swap=$2; shift 2
  local before; before=$(ls -t "$RUNS" | head -1)
  log "START $label (MAX_SWAP_GROWTH_MB=$swap) :: $*"
  log "free: $(free -m | awk 'NR==2{print "used="$3" avail="$7} NR==3{print "swap="$3}' | tr '\n' ' ')"
  MAX_SWAP_GROWTH_MB=$swap bash scripts/jetson_memguard.sh "$HOME/results/gap-20261014/guard-$label" "$@" \
    > "$HOME/results/gap-20261014/$label.log" 2>&1
  local rc=$?
  local dir; dir=$(ls -t "$RUNS" | head -1)
  [[ "$dir" == "$before" ]] && dir="(no new run directory)"
  cp "$HOME/results/gap-20261014/guard-$label/memguard.json" "$RUNS/$dir/memguard.json" 2>/dev/null
  local status; status=$($SYS_PY -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$RUNS/$dir/status.json" 2>/dev/null || echo unknown)
  log "END $label rc=$rc status=$status dir=$dir guard=$(tr -d '\n' < "$HOME/results/gap-20261014/guard-$label/memguard.json" 2>/dev/null | cut -c1-160)"
  sleep 30
  [[ $rc -eq 0 && "$status" == complete ]]
}

# smoke_then_full <label> <swap-MB> <smoke-config> <full-config> <runner...>
smoke_then_full() {
  local label=$1 swap=$2 smoke=$3 full=$4; shift 4
  if run_job "$label-smoke" "$swap" "$@" --config "$smoke" --kind attempt; then
    run_job "$label-full" "$swap" "$@" --config "$full" --kind baseline
  else
    log "SKIP $label-full: smoke did not complete"
  fi
}

SD=( "$SYS_PY" scripts/run_jetson_sdcpp.py --binary "$SDBENCH" --out-root "$RUNS" )
ED=( "$SYS_PY" scripts/run_edgedit_jetson.py --out-root "$RUNS" )
PT=( "$TORCH_PY" scripts/run_flux_portable.py --out-root "$RUNS" )
C=configs

log "QUEUE BEGIN $(nvpmodel -q 2>/dev/null | head -1)"
# 1. sd.cpp reference (003 config) repeated with the tiling-capable harness: same-session, same-binary control.
run_job sdcpp-003-repeat 768 "${SD[@]}" --config $C/jetson-flux-klein-q4-512-disk-baseline.json --kind repeat
# 2-3. sd.cpp toggles
smoke_then_full sdcpp-010 768 $C/jetson-flux-klein-q4-512-disk-noflash-smoke.json $C/jetson-flux-klein-q4-512-disk-noflash.json "${SD[@]}"
smoke_then_full sdcpp-011 768 $C/jetson-flux-klein-q4-512-disk-vaetile-smoke.json $C/jetson-flux-klein-q4-512-disk-vaetile.json "${SD[@]}"
# 4-5. edge-dit baseline and flash-attention-off
smoke_then_full edgedit-006 768 $C/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-smoke.json $C/jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json "${ED[@]}"
smoke_then_full edgedit-007 768 $C/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-noflash-smoke.json $C/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-noflash.json "${ED[@]}"
# 6-8. PyTorch: 001 smoke already completed on 2026-10-12 (attempt 153448); swap-growth guard raised to 1536 MB
# because the 2026-10-12 baseline was stopped by slow swap growth while MemAvailable stayed >= 1198 MB.
run_job torch-001-full 1536 "${PT[@]}" --config $C/jetson-torch-flux-klein-q4-nf4-512.json --kind baseline
smoke_then_full torch-002 1536 $C/jetson-torch-flux-klein-q4-nf4-512-sdpa-math-smoke.json $C/jetson-torch-flux-klein-q4-nf4-512-sdpa-math.json "${PT[@]}"
smoke_then_full torch-003 1536 $C/jetson-torch-flux-klein-q4-nf4-512-vaetile-smoke.json $C/jetson-torch-flux-klein-q4-nf4-512-vaetile.json "${PT[@]}"
log "QUEUE END"
