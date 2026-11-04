#!/usr/bin/env bash
# T0: PyTorch environment check for the Jetson Orin Nano (gap campaign, rung T0).
# Run on the Jetson:  bash jetson_env_probe.sh <output-dir>
# Creates a venv with a JetPack 6 / CUDA 12.6 PyTorch build and the diffusers and transformers versions
# used by the A100 reference, then checks each component the T1/T2 rungs need.
# Every check is recorded, pass or fail; nothing is substituted silently.
# STACK=ailab (default): torch 2.8.0, torchao 0.13.0, bitsandbytes from NVIDIA's Jetson AI Lab index.
# STACK=nvidia: NVIDIA's JetPack 6.1 redist wheel (torch 2.5.0a0). Recorded failure on 2026-10-11: built
# without torch.distributed, so diffusers/torchao cannot import it, and transformers rejects "2.5.0a0".
set -uo pipefail

OUT=${1:-$HOME/results/jetson-torch-env-$(date -u +%Y%m%d-%H%M%S)}
STACK=${STACK:-ailab}
AILAB_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/
if [[ "$STACK" == "nvidia" ]]; then
  VENV=${VENV:-$HOME/venvs/torch-jp6}
  TORCH_SPEC=https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
  TORCHAO_SPEC="torchao==0.7.0"; BNB_SPEC=bitsandbytes; INDEX_ARGS=()
else
  VENV=${VENV:-$HOME/venvs/torch-jp6-ailab}
  TORCH_SPEC="torch==2.8.0"; TORCHAO_SPEC="torchao==0.13.0"; BNB_SPEC="bitsandbytes==0.48.0.dev0+ff389db"
  # --index-url, not --extra-index-url: torch must come from the Jetson index, never PyPI's CPU build.
  INDEX_ARGS=(--index-url "$AILAB_INDEX")
fi
# Same versions as the A100 reference environment (wiki/systems/pytorch-diffusers.md).
DIFFUSERS_VER=0.40.0
TRANSFORMERS_VER=5.16.1
mkdir -p "$OUT"
exec > >(tee "$OUT/probe.log") 2>&1

echo "== system"
head -1 /etc/nv_tegra_release; uname -r; python3 --version
nvpmodel -q 2>/dev/null | head -1; free -h | sed -n 2p

[[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip

step() {  # step <name> <command...>; records pass/fail per step
  local name=$1; shift
  echo "== $name"
  if "$@"; then echo "$name ok" >> "$OUT/steps.txt"; else echo "$name FAILED" >> "$OUT/steps.txt"; fi
}

echo "stack=$STACK torch=$TORCH_SPEC" > "$OUT/stack.txt"
step install_torch "$PY" -m pip install -q "${INDEX_ARGS[@]}" "numpy<2" "$TORCH_SPEC"
# Torch builds for JetPack may link libcusparseLt, which JetPack does not ship; the PyPI wheel provides it in user space.
step install_cusparselt "$PY" -m pip install -q nvidia-cusparselt-cu12
CUSPARSELT_LIB=$("$PY" -c "import nvidia.cusparselt, os; print(os.path.join(list(nvidia.cusparselt.__path__)[0], 'lib'))" 2>/dev/null || true)
export LD_LIBRARY_PATH="${CUSPARSELT_LIB:+$CUSPARSELT_LIB:}${LD_LIBRARY_PATH:-}"
echo "$CUSPARSELT_LIB" > "$OUT/cusparselt_lib.txt"

step install_diffusers "$PY" -m pip install -q "diffusers==$DIFFUSERS_VER" "transformers==$TRANSFORMERS_VER" accelerate safetensors "gguf>=0.10" sentencepiece
step install_torchao "$PY" -m pip install -q "${INDEX_ARGS[@]}" --no-deps "$TORCHAO_SPEC"
step install_bitsandbytes "$PY" -m pip install -q "${INDEX_ARGS[@]}" --no-deps "$BNB_SPEC"

cat > "$OUT/checks.py" <<'EOF'
import json, sys, time, traceback
out = sys.argv[1]
only = sys.argv[2]
res = {}
def check(name, fn):
    if name != only:
        return
    try:
        res[name] = {"ok": True, "value": fn()}
    except Exception as e:
        res[name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    print(name, json.dumps(res[name])[:400])

import importlib.metadata as md
def versions():
    v = {}
    for p in ["torch", "diffusers", "transformers", "accelerate", "gguf", "torchao", "bitsandbytes", "numpy"]:
        try: v[p] = md.version(p)
        except Exception: v[p] = None
    return v
check("versions", versions)

def cuda():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    p = torch.cuda.get_device_properties(0)
    return {"device": p.name, "cc": f"{p.major}.{p.minor}", "total_gib": p.total_memory / 2**30,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "bf16": torch.cuda.is_bf16_supported()}
check("cuda", cuda)

def matmul_bf16():
    import torch
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
    for _ in range(3): (a @ b)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20): (a @ b)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 20
    return {"ms": dt * 1e3, "tflops": 2 * 2048**3 / dt / 1e12}
check("matmul_bf16", matmul_bf16)

def sdpa_backends():
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    q = torch.randn(1, 24, 1536, 128, device="cuda", dtype=torch.bfloat16); k = torch.randn_like(q); v = torch.randn_like(q)
    r = {}
    for name, be in [("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("math", SDPBackend.MATH)]:
        try:
            with sdpa_kernel(be):
                F.scaled_dot_product_attention(q, k, v); torch.cuda.synchronize()
            r[name] = "ok"
        except Exception as e:
            r[name] = f"{type(e).__name__}: {str(e)[:200]}"
    return r
check("sdpa_backends", sdpa_backends)

def flux2_classes():
    import diffusers
    names = ["Flux2KleinPipeline", "Flux2Transformer2DModel", "GGUFQuantizationConfig", "AutoencoderKLFlux2"]
    return {n: hasattr(diffusers, n) for n in names}
check("diffusers_flux2", flux2_classes)

def gguf_cuda_dequant():
    import torch, gguf
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor, GGUFParameter
    return {"gguf": gguf.__version__ if hasattr(gguf, "__version__") else "present", "diffusers_gguf_utils": True}
check("gguf", gguf_cuda_dequant)

def torchao_int4():
    import torch
    from torchao.quantization import quantize_
    try:
        from torchao.quantization import Int4WeightOnlyConfig as cfg  # torchao >= 0.10
    except ImportError:
        from torchao.quantization import int4_weight_only as cfg
    m = torch.nn.Sequential(torch.nn.Linear(2560, 2560, bias=False)).to("cuda", torch.bfloat16)
    quantize_(m, cfg())
    y = m(torch.randn(8, 2560, device="cuda", dtype=torch.bfloat16)); torch.cuda.synchronize()
    return {"out_shape": list(y.shape), "finite": bool(torch.isfinite(y).all())}
check("torchao_int4_cuda", torchao_int4)

def bnb_nf4():
    import torch, bitsandbytes as bnb
    lin = bnb.nn.Linear4bit(2560, 2560, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
    lin = lin.to("cuda")
    y = lin(torch.randn(8, 2560, device="cuda", dtype=torch.bfloat16)); torch.cuda.synchronize()
    return {"out_shape": list(y.shape), "finite": bool(torch.isfinite(y).all())}
check("bitsandbytes_nf4_cuda", bnb_nf4)

json.dump(res, open(f"{out}/check-{only}.json", "w"), indent=1)
EOF
echo "== checks"
for c in versions cuda matmul_bf16 sdpa_backends diffusers_flux2 gguf torchao_int4_cuda bitsandbytes_nf4_cuda; do
  "$PY" "$OUT/checks.py" "$OUT" "$c"; rc=$?
  [[ -f "$OUT/check-$c.json" ]] || echo "{\"$c\": {\"ok\": false, \"error\": \"process exited $rc before writing a result\"}}" > "$OUT/check-$c.json"
  echo "check $c exit=$rc" >> "$OUT/steps.txt"
done

"$PY" -m pip freeze > "$OUT/pip-freeze.txt"
echo "== steps"; cat "$OUT/steps.txt"
echo "PROBE_DONE $OUT"
