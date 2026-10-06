#!/bin/bash
set -euo pipefail
cd /home/mfaruqi/on-device-diffusion-campaigns/20261004-overnight
export WANDB_DIR=/scratch/gilbreth/mfaruqi/wandb
~/.venvs/wandb/bin/python scripts/export_wandb.py --entity mfaruqi-purdue-university --only a100-sdcpp-flux-klein-base-002__attempt__20261005-191624
~/.venvs/wandb/bin/python scripts/export_wandb.py --entity mfaruqi-purdue-university --only a100-sdcpp-flux-klein-001__attempt__20261005-193229
~/.venvs/wandb/bin/python scripts/export_wandb.py --entity mfaruqi-purdue-university --only a100-sdcpp-flux-klein-002__attempt__20261005-193329
~/.venvs/wandb/bin/python scripts/export_wandb.py --entity mfaruqi-purdue-university --only a100-sdcpp-flux-klein-003__attempt__20261005-193430
~/.venvs/wandb/bin/python scripts/export_wandb.py --entity mfaruqi-purdue-university --only a100-sdcpp-flux-klein-004__attempt__20261005-193530
~/.venvs/wandb/bin/python scripts/export_wandb.py --entity mfaruqi-purdue-university --only a100-sdcpp-flux-klein-005__attempt__20261005-193630
~/.venvs/wandb/bin/python scripts/export_wandb.py --entity mfaruqi-purdue-university --only a100-sdcpp-flux-klein-base-001__profile__20261005-191522
~/.venvs/wandb/bin/python scripts/export_wandb.py --entity mfaruqi-purdue-university --only a100-edgedit-flux-klein-001__attempt__20261005-191822
