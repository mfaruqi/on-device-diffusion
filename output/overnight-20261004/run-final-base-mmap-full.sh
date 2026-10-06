#!/bin/bash
set -eu
cd /home/mfaruqi/on-device-diffusion-campaigns/20261005-final-base
if pgrep -f '^python3 scripts/run_jetson_sdcpp.py' >/dev/null || pgrep -x sd-bench >/dev/null || pgrep -x sd-bench-nvtx >/dev/null; then
  echo 'Refusing: existing benchmark active'; exit 1
fi
mkdir output/base-final-mmap-full-launch
export PYTHONUNBUFFERED=1
binary="$HOME/tools/sd-bench-build-cache-20261005/sd-bench"
python3 scripts/run_jetson_sdcpp.py --config configs/jetson-flux-klein-base-q4-512-mmap.json --binary "$binary" --out-root results/runs --kind baseline
