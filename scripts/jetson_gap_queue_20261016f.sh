#!/usr/bin/env bash
# 2026-10-16 rerun of the Jetson Base queue (runner step-count fix; Base snapshot allow_patterns fix): klein Base (50 steps, CFG 4) — edge-dit reference + step-cache screens; PyTorch reference,
# conditioning reuse, TaylorSeer and torch.compile. Base generations take minutes each; expect roughly 10 hours.
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

ED=( "$SYS_PY" scripts/run_edgedit_jetson.py --out-root "$RUNS" )
PT=( "$TORCH_PY" scripts/run_flux_portable.py --out-root "$RUNS" )
C=configs
EB=jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile

cache_screen() {  # smoke; full only if the engine skipped/reused any step or did not report
  local m=$1
  if run_job "base-edgedit-$m-smoke" 768 "${ED[@]}" --config "$C/$EB-$m-smoke.json" --kind attempt; then
    local dir; dir=$(ls -td "$RUNS"/jetson-edgedit-flux-klein-base-*__attempt__* | head -1)
    local skipped; skipped=$($SYS_PY -c "import json,sys;s=json.load(open(sys.argv[1]))['skipped_steps_per_generation'];print('unknown' if s is None else sum(s))" "$dir/step-cache.json" 2>/dev/null || echo unknown)
    if [[ "$skipped" == "0" ]]; then log "SCREEN base $m: 0 skipped/reused steps; full protocol not run"
    else log "SCREEN base $m: skipped=$skipped; running full"; run_job "base-edgedit-$m-full" 768 "${ED[@]}" --config "$C/$EB-$m.json" --kind baseline; fi
  else log "SKIP base-edgedit-$m-full: smoke did not complete"; fi
}

log "QUEUE-F BEGIN $(nvpmodel -q 2>/dev/null | head -1)"
smoke_then_full base-edgedit-001 768 $C/$EB-smoke.json $C/$EB.json "${ED[@]}"
for m in easycache ucache dbcache taylorseer cache-dit magcache dicache; do cache_screen $m; done
TB=jetson-torch-flux-klein-base-q4-nf4-512
smoke_then_full base-torch-001 1536 $C/$TB-smoke.json $C/$TB.json "${PT[@]}"
smoke_then_full base-torch-002 1536 $C/$TB-condreuse-smoke.json $C/$TB-condreuse.json "${PT[@]}"
smoke_then_full base-torch-003 1536 $C/$TB-condreuse-taylorseer-smoke.json $C/$TB-condreuse-taylorseer.json "${PT[@]}"
TORCHINDUCTOR_COMPILE_THREADS=1 smoke_then_full base-torch-004 1536 $C/$TB-condreuse-compile-smoke.json $C/$TB-condreuse-compile.json "${PT[@]}"
log "QUEUE-F END"
