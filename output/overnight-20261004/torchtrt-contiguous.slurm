#!/bin/bash
#SBATCH --job-name=torchtrt-layout-gate
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --constraint=G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=logs/torchtrt-contiguous-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/scratch/gilbreth/mfaruqi/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
TRT_PYTHON=/scratch/gilbreth/mfaruqi/envs/torchtrt-2.5.0/bin/python
# Bound the gate, leaving time to preserve timeout/termination status before Slurm kills the job.
timeout --kill-after=30 1200 "$TRT_PYTHON" torchtrt-contiguous.py
result=$?
if [ "$result" -ne 0 ]; then
  GATE_EXIT="$result" "$TRT_PYTHON" - <<'PY'
import json,os
from pathlib import Path
p=Path('output')/('torchtrt-contiguous-'+os.environ['SLURM_JOB_ID'])/'status.json'
if p.exists():
 s=json.loads(p.read_text())
 if s.get('status')=='running':
  s.update(status='failed',error='Gate process terminated; exit='+os.environ['GATE_EXIT']+'; see Slurm log')
  p.write_text(json.dumps(s,indent=2)+'\n')
PY
fi
exit "$result"
