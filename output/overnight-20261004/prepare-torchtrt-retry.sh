#!/bin/bash
set -euo pipefail
cd /home/mfaruqi/on-device-diffusion-campaigns/20261005-options
TRT_ENV=/scratch/gilbreth/mfaruqi/envs/torchtrt-2.5.0
# Isolated dependency gate: never installs into flux_env. CUDA build differs and is recorded.
test -x "$TRT_ENV/bin/python"
mkdir -p /scratch/gilbreth/mfaruqi/envs
"$TRT_ENV/bin/python" -m pip install --timeout 30 --retries 2 pip==25.0.1
"$TRT_ENV/bin/python" -m pip install --timeout 30 --retries 2 'torch==2.5.1+cu124' --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
"$TRT_ENV/bin/python" -m pip install --timeout 30 --retries 2 'torch-tensorrt==2.5.0' 'diffusers==0.40.0' 'transformers==5.16.1' 'accelerate==1.14.0' 'safetensors==0.8.0' 'huggingface-hub==1.30.0' 'Pillow==12.3.0' 'sentencepiece==0.2.2' 'numpy==2.2.6' 'psutil==7.2.2'
"$TRT_ENV/bin/python" -m pip freeze > logs/torchtrt-freeze.txt
"$TRT_ENV/bin/python" -m pip check
"$TRT_ENV/bin/python" - <<'PY'
import json,torch,torch_tensorrt,tensorrt,diffusers,transformers
from pathlib import Path
Path('logs/torchtrt-import.json').write_text(json.dumps({'torch':torch.__version__,'cuda_build':torch.version.cuda,'torch_tensorrt':torch_tensorrt.__version__,'tensorrt':tensorrt.__version__,'diffusers':diffusers.__version__,'transformers':transformers.__version__,'scope':'Import only on login node; no inference; GPU compile support untested.'},indent=2))
PY
printf 'complete\n' > logs/torchtrt-prepare.done
