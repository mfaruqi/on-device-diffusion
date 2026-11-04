#!/bin/bash
#SBATCH --job-name=engine-counters
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
# standby: preemptible, 4 h max, does not use the lab's single normal-QoS A100
#SBATCH --qos=standby
# constraint G = the PCIe A100-40GB nodes used by every A100 benchmark
#SBATCH --constraint=G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:45:00
#SBATCH --output=logs/engine-counters-%j.out
# Usage: sbatch --export=ALL,PLAN=configs/ncu/<plan>.json scripts/gilbreth-engine-counters.slurm
# Add RESUME=results/runs/<counters run> to reuse an existing capture's CSVs and rerun only the missing pass.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${PLAN:?set PLAN=configs/ncu/<plan>.json}"
export HF_HOME=/scratch/gilbreth/mfaruqi/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset GGML_CUDA_CUBLAS_COMPUTE_TYPE
# Library paths the C++ engines were benchmarked with (same as gilbreth-sdcpp/edgedit.slurm).
CUDA_ROOT=/apps/spack/gilbreth-r9/apps/cuda/12.6.0-gcc-11.5.0-a7cv7sp
NV=$HOME/.conda/envs/2025.06-py313/flux_env/lib/python3.10/site-packages/nvidia
LAUNCHER=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['launcher'])" "$PLAN")
if [[ "$LAUNCHER" == sdcpp || "$LAUNCHER" == edgedit* ]]; then
  export PATH="$CUDA_ROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cublas/lib:$NV/cuda_nvrtc/lib:$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
echo "job=$SLURM_JOB_ID node=$(hostname) plan=$PLAN launcher=$LAUNCHER start=$(date -Is)"
# The wrapper only needs numpy + the standard library; each runner uses its own interpreter from the plan.
$HOME/.conda/envs/2025.06-py313/flux_env/bin/python scripts/profile_engine_counters.py --plan "$PLAN" ${RESUME:+--resume "$RESUME"}
echo "end=$(date -Is)"
