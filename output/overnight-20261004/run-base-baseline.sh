#!/bin/bash
set -euo pipefail
cd /home/mfaruqi/on-device-diffusion-campaigns/20261004-overnight
export PYTHONUNBUFFERED=1
if pgrep -x sd-bench || pgrep -x sd-bench-nvtx; then echo 'Refusing concurrent benchmark'; exit 1; fi
python3 scripts/run_jetson_sdcpp.py --config configs/jetson-flux-klein-base-q4-512-disk.json --binary "$HOME/tools/sd-bench-build-20261004/sd-bench" --out-root results/runs --kind baseline
