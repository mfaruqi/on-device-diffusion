#!/usr/bin/env bash
# 2026-11-02 queue N: Nunchaku SVDQuant on the Jetson, from the in-place sm_87 build (jetson_build_nunchaku_sm87.sh).
#   1. Patch the second allowlist (nunchaku/utils.py check_hardware_compatibility, which raises "Unsupported GPU
#      architecture 87" while loading a model) and check that the tree's diff hashes to the configs' patch_sha256.
#   2. Single-layer check (scripts/nunchaku_layer_check.py): four W4A4 layers of the community checkpoint vs their BF16
#      originals and vs the A100's outputs for the same inputs. Gate: every output finite and within rel_l2 <= 1e-2 of
#      the A100's.
#   3. Only if the gate passes: jetson-nunchaku-flux-klein-int4-r32-512 smoke (first + one measured), then the full
#      1+3+10 protocol if the smoke completed.
# Inputs (copied to the board beforehand): the checkpoint at the config's path, the layer bundle and the A100 reference
# outputs.safetensors under ~/models/nunchaku-layer-check/.
# Usage (on the Jetson, no root; desktop session logged out):
#   setsid nohup bash scripts/jetson_nunchaku_queue_20261102.sh > ~/results/nunchaku-20261102.out 2>&1 < /dev/null &
# About 35 minutes. One job at a time; every job runs under jetson_memguard.sh.
set -uo pipefail
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
(( avail >= 5700 )) || { echo "MemAvailable ${avail} MB < 5700 MB: log out of the desktop session first"; exit 1; }
cd "$(dirname "$0")/.."
REPO=$PWD
RUNS=$REPO/results/runs
Q=$HOME/results/nunchaku-20261102
LOG=$Q/queue.log
mkdir -p "$Q"
SYS_PY=/usr/bin/python3
TORCH_PY=$HOME/venvs/torch-jp6-ailab/bin/python
SRC=$HOME/tools/nunchaku-sm87-0c8379d
LC=$HOME/models/nunchaku-layer-check
export HF_HOME=$HOME/hf HF_HUB_OFFLINE=1 PATH=/usr/local/cuda/bin:$PATH
C=configs
N=jetson-nunchaku-flux-klein-int4-r32-512

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

log "QUEUE-N BEGIN $(nvpmodel -q 2>/dev/null | head -1) MemAvailable=${avail}MB"

# 1. Second allowlist line; the runner refuses to load if the diff differs from the configs' patch_sha256.
grep -q 'elif sm in \["75", "80", "86", "87", "89"\]' "$SRC/nunchaku/utils.py" \
  || sed -i 's/elif sm in \["75", "80", "86", "89"\]:/elif sm in ["75", "80", "86", "87", "89"]:/' "$SRC/nunchaku/utils.py"
want=$($SYS_PY -c "import json;print(json.load(open('$C/$N.json'))['optimizations']['quantization']['transformer']['patch_sha256'])")
got=$(git -C "$SRC" diff | $SYS_PY -c "import sys,hashlib;print(hashlib.sha256(sys.stdin.read().strip().encode()).hexdigest())")
git -C "$SRC" diff > "$Q/patch.diff"
[[ "$got" == "$want" ]] || { log "STOP patch sha256 $got != config $want (patch.diff)"; exit 1; }
log "patch ok ($(git -C "$SRC" diff --stat | tail -1))"

# 2. Single-layer check. Not a pipeline run: it writes layer-check.json, not status.json, so it is gated here.
LDIR=$RUNS/jetson-nunchaku-layer-check-001__layer__$(date -u +%Y%m%d-%H%M%S)
mkdir -p "$LDIR"
log "START layer-check -> $LDIR"
MAX_SWAP_GROWTH_MB=512 bash scripts/jetson_memguard.sh "$Q/guard-layer" "$TORCH_PY" scripts/nunchaku_layer_check.py run \
  --bundle "$LC/layer-bundle.safetensors" --reference "$LC/a100-outputs.safetensors" --nunchaku-dir "$SRC" \
  --out "$LDIR" > "$LDIR/stdout.txt" 2>&1
rc=$?
cp "$Q/guard-layer/memguard.json" "$LDIR/memguard.json" 2>/dev/null
gate=$($SYS_PY - "$LDIR/layer-check.json" <<'EOF'
import json, sys
try:
    rows = json.load(open(sys.argv[1]))['layers']
except Exception as error:  # noqa: BLE001
    print(f'fail: no layer-check.json ({type(error).__name__})'); sys.exit()
bad = [r['layer'] + f" M={r['tokens']}" for r in rows if not r['finite']
       or ('vs_reference' in r and not r['vs_reference']['rel_l2'] <= 1e-2)]
refs = sum('vs_reference' in r for r in rows)
print('pass' if rows and refs and not bad else f'fail: {bad or "no reference comparisons"}')
EOF
)
log "END layer-check rc=$rc gate=$gate"
[[ $rc -eq 0 && "$gate" == pass ]] || { log "STOP pipeline not run: layer check did not pass"; exit 1; }
sleep 30

# 3. Pipeline: smoke, then full protocol.
PT=( "$TORCH_PY" scripts/run_flux_portable.py --out-root "$RUNS" )
if run_job nunchaku-smoke 1536 "${PT[@]}" --config "$C/$N-smoke.json" --kind attempt; then
  run_job nunchaku-full 1536 "${PT[@]}" --config "$C/$N.json" --kind baseline
else
  log "SKIP nunchaku-full: smoke did not complete"
fi
log "QUEUE-N END"
