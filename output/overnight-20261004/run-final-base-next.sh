#!/bin/bash
set -eu
cd /home/mfaruqi/on-device-diffusion-campaigns/20261005-final-base
if pgrep -f '^python3 scripts/run_jetson_sdcpp.py' >/dev/null || pgrep -x sd-bench >/dev/null || pgrep -x sd-bench-nvtx >/dev/null; then
  echo 'Refusing: existing benchmark active'; exit 1
fi
mkdir output/base-final-next-launch
export PYTHONUNBUFFERED=1
binary="$HOME/tools/sd-bench-build-cache-20261005/sd-bench"
set +e
python3 scripts/run_jetson_sdcpp.py --config configs/jetson-flux-klein-base-q4-512-easycache-conditioning.json --binary "$binary" --out-root results/runs --kind baseline
code=$?
echo "combination full exit=$code"
# Mmap is an independently labelled option; preserve any combination failure.
python3 scripts/run_jetson_sdcpp.py --config configs/jetson-flux-klein-base-q4-512-mmap-smoke.json --binary "$binary" --out-root results/runs --kind attempt
code=$?
echo "Base mmap smoke exit=$code"
exit "$code"
