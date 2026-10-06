#!/bin/bash
set -eu
cd /home/mfaruqi/on-device-diffusion-campaigns/20261005-distilled
# Wait for the existing Base protocol; never overlap GPU measurements.
while pgrep -x sd-bench >/dev/null || pgrep -x sd-bench-nvtx >/dev/null || pgrep -f '^python3 scripts/run_jetson_sdcpp.py' >/dev/null; do sleep 30; done
export PYTHONUNBUFFERED=1
binary="$HOME/tools/sd-bench-build-cache-20261005/sd-bench"
python3 scripts/run_jetson_sdcpp.py --config configs/jetson-flux-klein-q4-512-cache-control-smoke.json --binary "$binary" --out-root results/runs --kind attempt
# These independent attempts preserve failures; full benchmarks wait for review.
set +e
python3 scripts/run_jetson_sdcpp.py --config configs/jetson-flux-klein-q4-512-easycache-smoke.json --binary "$binary" --out-root results/runs --kind attempt
echo "easycache exit=$?"
python3 scripts/run_jetson_sdcpp.py --config configs/jetson-flux-klein-q4-512-conditioning-cache-smoke.json --binary "$binary" --out-root results/runs --kind attempt
echo "conditioning-cache exit=$?"
python3 scripts/run_jetson_sdcpp.py --config configs/jetson-flux-klein-q4-512-no-prefetch-smoke.json --binary "$binary" --out-root results/runs --kind attempt
echo "no-prefetch exit=$?"
python3 scripts/run_jetson_sdcpp.py --config configs/jetson-flux-klein-q4-512-mmap-smoke.json --binary "$binary" --out-root results/runs --kind attempt
echo "mmap exit=$?"
