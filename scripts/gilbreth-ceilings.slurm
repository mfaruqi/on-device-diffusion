#!/bin/bash
#SBATCH --job-name=a100-ceilings
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
# standby: preemptible, 4 h max, does not use the lab's single normal-QoS A100
#SBATCH --qos=standby
# constraint G = the PCIe A100-40GB nodes used by every A100 benchmark
#SBATCH --constraint=G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/a100-ceilings-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1
# Same torch 2.8 environment as the hardware-counter capture.
PYTHON=/scratch/gilbreth/mfaruqi/envs/torch28-trt/bin/python
CONFIG=${CONFIG:-configs/a100-ceilings.json}
exec "$PYTHON" scripts/measure_ceilings_cuda.py --config "$CONFIG"
