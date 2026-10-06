#!/bin/bash
set -euo pipefail
cd /home/mfaruqi/on-device-diffusion-campaigns/20261004-overnight
export WANDB_DIR=/scratch/gilbreth/mfaruqi/wandb
~/.venvs/wandb/bin/python output/export_wandb-iso8601.py --entity mfaruqi-purdue-university --only a100-sdcpp-flux-klein-001__baseline__20261005-220435
~/.venvs/wandb/bin/python output/export_wandb-iso8601.py --entity mfaruqi-purdue-university --only a100-sdcpp-flux-klein-002__baseline__20261005-220636
~/.venvs/wandb/bin/python output/export_wandb-iso8601.py --entity mfaruqi-purdue-university --only a100-sdcpp-flux-klein-003__baseline__20261005-220737
~/.venvs/wandb/bin/python output/export_wandb-iso8601.py --entity mfaruqi-purdue-university --only a100-sdcpp-flux-klein-004__baseline__20261005-220938
~/.venvs/wandb/bin/python output/export_wandb-iso8601.py --entity mfaruqi-purdue-university --only a100-sdcpp-flux-klein-005__baseline__20261005-221039
~/.venvs/wandb/bin/python output/export_wandb-iso8601.py --entity mfaruqi-purdue-university --only a100-sdcpp-flux-klein-base-001__baseline__20261005-214624
~/.venvs/wandb/bin/python output/export_wandb-iso8601.py --entity mfaruqi-purdue-university --only a100-sdcpp-flux-klein-base-002__baseline__20261005-215731
~/.venvs/wandb/bin/python output/export_wandb-iso8601.py --entity mfaruqi-purdue-university --only a100-edgedit-flux-klein-001__attempt__20261005-221339
