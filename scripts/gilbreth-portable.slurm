#!/bin/bash
# Portable PyTorch runner (run_flux_portable.py) on one Gilbreth A100-PCIE-40GB.
# Submit from the repository root after scripts/gilbreth_setup_torch28.sh, smoke config first:
#
#   mkdir -p logs
#   sbatch --export=ALL,CONFIG=configs/a100-flux-klein-torch28-bf16-smoke.json,KIND=attempt scripts/gilbreth-portable.slurm
#   sbatch --export=ALL,CONFIG=configs/a100-flux-klein-torch28-bf16.json,KIND=baseline scripts/gilbreth-portable.slurm
#
# Compiler configs (TensorRT, torch.compile) compile lazily in the first generation; the longer
# time limit covers that. The first generation is reported separately.
#
#SBATCH --job-name=flux-portable
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --constraint=G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/flux-portable-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV="${VENV:-/depot/you139/mfaruqi/on-device-diffusion/venvs/torch28-trt}"
PYTHON="$VENV/bin/python"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/$USER/huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
: "${CONFIG:?set CONFIG=configs/<name>.json}"

echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is) python=$PYTHON config=$CONFIG"
nvidia-smi
"$PYTHON" scripts/run_flux_portable.py --config "$CONFIG" --kind "${KIND:-attempt}"
echo "end=$(date -Is)"
