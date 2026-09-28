---
type: finding
summary: In the PyTorch reference on A100-PCIE, about a third of each FLUX.2 klein denoise step is unfused memory-bound ops; GEMMs already run near peak and the GPU is almost never idle.
status: supported
confidence: high
rq: [RQ3]
sources: [../../experiments/a100-flux-klein-001.md]
updated: 2026-09-28
---

# A third of a PyTorch denoise step is unfused memory-bound ops

**Claim.** Of a **246.4 ms**
([`stages.denoise_step_1.span_ms`](../../results/runs/a100-flux-klein-001-20260923-221717-profile/profile/denoise_kernels.json))
denoise step, about 83 ms goes to elementwise multiply/add, dtype casts, decomposed RMSNorm, and RoPE
concatenation. GEMMs run at 68–82% of the BF16 peak, and the GPU idles only
**2.3 ms**
([`stages.denoise_step_1.idle_ms`](../../results/runs/a100-flux-klein-001-20260923-221717-profile/profile/denoise_kernels.json)).

**Evidence.** [Kernel breakdown](../../experiments/a100-flux-klein-001.md#kernel-mix-within-one-denoise-step-profiled-generation-job-11807396).

**Interpretation.** On this GPU the headroom is in fusing the non-GEMM ops, or in lower-precision GEMM
kernels. Launch overhead and GEMM scheduling have little room. A `torch.compile` variant would test
the fusion part.

Related: [RQ3](../rq/rq3.md), [pytorch-diffusers](../systems/pytorch-diffusers.md).
