---
type: finding
summary: For FLUX.2 klein at 1024² on the A100, the VAE decode — not the transformer — sets the peak memory in both PyTorch and sd.cpp.
status: supported
confidence: high
rq: [RQ1, RQ3]
sources: [../../experiments/a100-flux-klein-001.md, ../../experiments/a100-sdcpp-flux-klein-001.md]
updated: 2026-09-28
---

# The VAE decode sets FLUX.2 klein's peak memory

**Claim.** The device-wide peak during a generation is **20.61 GiB**
([`measured.device_used_peak_gib.median`](../../results/runs/a100-flux-klein-001-20260923-221530/summary.json))
with PyTorch and **22.88 GiB**
([`measured.device_used_peak_gib.median`](../../results/runs/a100-sdcpp-flux-klein-001-20260925-114328/summary.json))
with sd.cpp, and in both it's reached during VAE decode. Denoise steps peak about 1.8 GiB (PyTorch
allocator) and 5.5 GiB (sd.cpp, device-wide) lower.

**Evidence.** Per-stage peaks in both records
([PyTorch](../../experiments/a100-flux-klein-001.md#memory), [sd.cpp](../../experiments/a100-sdcpp-flux-klein-001.md#memory-device-wide-nvml-gib)).

**Interpretation.** Under a memory budget the decode peak, not the weights of the denoiser, is the first
thing to plan around. Tiled decode or freeing the transformer before decode are the levers (RQ3).

Related: [stage residency](../concepts/stage-residency.md), [im2col in sd.cpp](sdcpp-vae-decode-dominated-by-im2col.md).
