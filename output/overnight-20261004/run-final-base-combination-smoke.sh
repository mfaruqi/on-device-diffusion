#!/bin/bash
set -eu
cd /home/mfaruqi/on-device-diffusion-campaigns/20261005-final-base
# Wait for the existing serial queue as well as its child benchmark to finish.
while pgrep -f '^bash run-distilled-baselines.sh' >/dev/null || pgrep -f '^python3 scripts/run_jetson_sdcpp.py' >/dev/null || pgrep -x sd-bench >/dev/null || pgrep -x sd-bench-nvtx >/dev/null; do sleep 30; done
mkdir output/base-combination-smoke-launch
export PYTHONUNBUFFERED=1
python3 scripts/run_jetson_sdcpp.py --config configs/jetson-flux-klein-base-q4-512-easycache-conditioning-smoke.json --binary "$HOME/tools/sd-bench-build-cache-20261005/sd-bench" --out-root results/runs --kind attempt
# Full protocol needs independent validation and image diagnostics first.
