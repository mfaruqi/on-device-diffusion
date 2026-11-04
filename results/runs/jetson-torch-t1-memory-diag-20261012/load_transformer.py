import gc, json, time, torch, diffusers
from collections import Counter
def avail(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) // 1024
snap = "/home/mfaruqi/hf/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294"
a0 = avail()
t = diffusers.Flux2TransformerModel.from_single_file if hasattr(diffusers, "Flux2TransformerModel") else diffusers.Flux2Transformer2DModel.from_single_file
m = t("/home/mfaruqi/models/flux2-klein/flux-2-klein-4b-Q4_0.gguf",
      quantization_config=diffusers.GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
      config=snap, subfolder="transformer", torch_dtype=torch.bfloat16)
gc.collect(); a1 = avail()
by = Counter(); n = Counter()
for p in m.parameters():
    k = type(p).__name__ + ":" + str(getattr(p, "quant_type", p.dtype)); by[k] += p.numel() * p.element_size(); n[k] += 1
print(json.dumps({"avail_before_mb": a0, "avail_after_load_cpu_mb": a1, "device": str(next(m.parameters()).device),
                  "param_bytes_mb": {k: round(v / 2**20, 1) for k, v in by.items()}, "param_counts": dict(n)}, indent=1))
m.to("cuda"); torch.cuda.synchronize(); gc.collect()
print(json.dumps({"avail_after_to_cuda_mb": avail(), "cuda_alloc_mb": torch.cuda.memory_allocated() / 2**20}))
