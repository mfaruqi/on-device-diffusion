#!/bin/bash
set -eu
cd /home/mfaruqi/on-device-diffusion-campaigns/20261005-distilled
# Exclusive serial device use; refuse rather than race an active benchmark.
if pgrep -x sd-bench >/dev/null || pgrep -x sd-bench-nvtx >/dev/null || pgrep -f '^python3 scripts/run_jetson_sdcpp.py' >/dev/null; then
  echo 'Refusing: benchmark already active.'; exit 1
fi
# This launch is one-shot; a retained marker prevents duplicate protocols.
mkdir output/distilled-full-launch
export PYTHONUNBUFFERED=1
binary="$HOME/tools/sd-bench-build-cache-20261005/sd-bench"
python3 scripts/run_jetson_sdcpp.py --config configs/jetson-flux-klein-q4-512-cache-control.json --binary "$binary" --out-root results/runs --kind baseline
# Control failure stops variants. Independently validated options preserve individual failures.
for option in easycache conditioning-cache no-prefetch mmap; do
  set +e
  python3 scripts/run_jetson_sdcpp.py --config "configs/jetson-flux-klein-q4-512-${option}.json" --binary "$binary" --out-root results/runs --kind baseline
  code=$?
  set -e
  echo "$option exit=$code"
done
