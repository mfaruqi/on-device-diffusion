#!/bin/bash
set -euo pipefail
cd /home/mfaruqi/on-device-diffusion-campaigns/20261005-options
export PYTHONUNBUFFERED=1
if pgrep -x sd-bench || pgrep -x sd-bench-nvtx; then echo 'Refusing concurrent benchmark'; exit 1; fi
test -e logs/cache-build.done
python3 scripts/run_jetson_sdcpp.py --config configs/jetson-flux-klein-base-q4-512-disk-smoke.json --binary "$HOME/tools/sd-bench-build-cache-20261005/sd-bench" --out-root results/runs --kind attempt
