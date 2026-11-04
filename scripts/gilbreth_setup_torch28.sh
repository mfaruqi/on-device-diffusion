#!/bin/bash
# One-time setup of the torch 2.8 + Torch-TensorRT environment for the A100 compiler screens.
# Run on a Gilbreth login node from the repository root (downloads wheels; no GPU needed):
#   bash scripts/gilbreth_setup_torch28.sh
# Kept separate from flux_env (torch 2.5.1), which the existing A100 baselines use.
set -euo pipefail
VENV=${VENV:-/depot/you139/mfaruqi/on-device-diffusion/venvs/torch28-trt}
BASE_PYTHON=${BASE_PYTHON:-$HOME/.conda/envs/2025.06-py313/flux_env/bin/python}   # Python 3.10, as in flux_env
[[ -x "$VENV/bin/python" ]] || "$BASE_PYTHON" -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip
# torch and torch_tensorrt must come from the same CUDA 12.6 index so their builds match.
# PyPI as an extra index: torch_tensorrt depends on packages (dllist) that the PyTorch index does not host.
"$PY" -m pip install -q --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple \
    "torch==2.8.0" "torch_tensorrt==2.8.0"
"$PY" -m pip install -q "diffusers==0.40.0" "transformers==5.16.1" accelerate safetensors sentencepiece psutil "gguf>=0.10"
"$PY" - <<'EOF'
import torch, torch_tensorrt, tensorrt, diffusers, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torch_tensorrt", torch_tensorrt.__version__, "tensorrt", tensorrt.__version__)
print("diffusers", diffusers.__version__, "transformers", transformers.__version__)
assert hasattr(diffusers, "Flux2KleinPipeline")
EOF
"$PY" -m pip freeze > "$VENV/pip-freeze.txt"
echo "SETUP_OK $VENV"
