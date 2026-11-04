#!/usr/bin/env bash
# Build probe: compile the pinned Nunchaku source (tonera/nunchaku 0c8379d, the source inspected for the A100 runs)
# for the Jetson Orin Nano GPU (sm_87). One change from the pinned source: "87" is added to the SM allowlist in
# setup.py get_sm_targets(), which otherwise stops with "Unsupported SM 87" (wiki/methods/nunchaku-howto.md).
# Build only: `setup.py build_ext --inplace` inside the source tree, then a load test of the compiled extension and a
# listing of the GPU code it contains. No model, no inference, no kernel numerics; no existing environment is changed
# (the ailab torch venv is used read-only as the build interpreter; nothing is installed into it).
# Usage (on the Jetson, no root):  setsid nohup bash jetson_build_nunchaku_sm87.sh > ~/results/nunchaku-sm87-build.out 2>&1 < /dev/null &
# Runs under jetson_memguard.sh: one compile job (MAX_JOBS=1), killed if MemAvailable < 400 MB or swap grows > 1.5 GB.
set -uo pipefail
REV=0c8379d10b4b9017928d02fd0fdb2c26b991a8b6
STAMP=$(date -u +%Y%m%d-%H%M%S)
OUT=$HOME/results/nunchaku-sm87-build-$STAMP
SRC=$HOME/tools/nunchaku-sm87-${REV:0:7}
PY=$HOME/venvs/torch-jp6-ailab/bin/python
GUARD=$HOME/on-device-diffusion-gap/scripts/jetson_memguard.sh
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH MAX_JOBS=1 NUNCHAKU_INSTALL_MODE=FAST
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/probe.log"; }
result() {  # result <status> <detail>
  "$PY" - "$OUT" "$1" "$2" "$REV" "$SRC" <<'EOF'
import json, sys, pathlib
out, status, detail, rev, src = sys.argv[1:]
p = pathlib.Path(out)
guard = p / 'guard' / 'memguard.json'
json.dump({'status': status, 'detail': detail, 'source': 'https://github.com/tonera/nunchaku', 'revision': rev,
           'source_dir': src, 'change': 'setup.py get_sm_targets(): "87" added to the SM allowlist (patch.diff)',
           'build': 'python setup.py build_ext --inplace, MAX_JOBS=1, NUNCHAKU_INSTALL_MODE=FAST',
           'memguard': json.loads(guard.read_text()) if guard.exists() else None},
          open(p / 'build.json', 'w'), indent=2)
EOF
  log "RESULT $1: $2"
  exit 0
}

log "BEGIN nunchaku sm_87 build probe -> $OUT"
{ uname -a; nvcc --version | tail -2; "$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'sm',torch.cuda.get_device_capability())";
  g++ --version | head -1; ninja --version; free -m; df -h "$HOME" | tail -1; } > "$OUT/env.txt" 2>&1

if [[ ! -d "$SRC/.git" ]]; then
  git clone --quiet https://github.com/tonera/nunchaku.git "$SRC" >> "$OUT/git.log" 2>&1 || result failed "clone"
fi
git -C "$SRC" checkout --quiet "$REV" >> "$OUT/git.log" 2>&1 || result failed "checkout $REV"
git -C "$SRC" submodule update --init --recursive >> "$OUT/git.log" 2>&1 || result failed "submodules"
git -C "$SRC" submodule status --recursive > "$OUT/submodules.txt"

grep -q '"75", "80", "86", "87", "89"' "$SRC/setup.py" \
  || sed -i 's/\["75", "80", "86", "89", "120a", "121a"\]/["75", "80", "86", "87", "89", "120a", "121a"]/' "$SRC/setup.py"
git -C "$SRC" diff > "$OUT/patch.diff"
[[ $(grep -c '^[-+] ' "$OUT/patch.diff") -eq 2 ]] || result failed "patch did not apply as one line (see patch.diff)"
log "patched; building (one job; expect hours)"

cd "$SRC"
t0=$(date +%s)
MAX_SWAP_GROWTH_MB=1536 bash "$GUARD" "$OUT/guard" "$PY" setup.py build_ext --inplace > "$OUT/build.log" 2>&1
rc=$?
log "build rc=$rc after $(( $(date +%s) - t0 )) s"
[[ $rc -eq 0 ]] || result failed "build_ext rc=$rc (build.log, guard/memguard.json)"

SO=$(ls "$SRC"/nunchaku/_C*.so 2>/dev/null | head -1)
[[ -n "$SO" ]] || result failed "no nunchaku/_C*.so after build"
cuobjdump --list-elf "$SO" > "$OUT/cuobjdump-elf.txt" 2>&1
ls -l "$SO" > "$OUT/extension.txt"
cd "$OUT"
"$PY" - "$SO" > "$OUT/load-test.txt" 2>&1 <<'EOF'
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location('nunchaku._C', sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('loaded', sys.argv[1])
print('device', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print('attributes', sorted(a for a in dir(mod) if not a.startswith('__')))
EOF
lrc=$?
grep -q sm_87 "$OUT/cuobjdump-elf.txt" || result failed "extension has no sm_87 code (cuobjdump-elf.txt)"
[[ $lrc -eq 0 ]] || result failed "extension built for sm_87 but failed to load (load-test.txt)"
result complete "built and loaded; sm_87 code present"
