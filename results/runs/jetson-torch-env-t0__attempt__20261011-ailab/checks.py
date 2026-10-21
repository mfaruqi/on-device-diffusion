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
