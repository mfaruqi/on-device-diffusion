---
type: finding
summary: With identical BF16 weights, per-step GEMM time on A100-PCIE is within ~7% between PyTorch (cuBLAS) and sd.cpp (cuBLAS via ggml).
status: supported
confidence: medium
rq: [RQ1]
sources: [../../experiments/compare-a100-pytorch-vs-sdcpp-flux-klein.md]
updated: 2026-09-28
---

# BF16 GEMM time transfers across PyTorch and stable-diffusion.cpp

**Claim.** The transformer's matrix multiplies take **123.5 ms**
([`stages.denoise_step_1.by_group.gemm.ms`](../../results/runs/a100-flux-klein-001-20260923-221717-profile/profile/denoise_kernels.json))
per step in PyTorch and **116.2 ms**
([`stages.denoise_step_1.by_group.gemm.ms`](../../results/runs/a100-sdcpp-flux-klein-001-20260925-124703-profile/profile/denoise_kernels.json))
in sd.cpp. Both dispatch to cuBLAS BF16 tensor-core kernels (`ampere_*16816gemm_bf16`).

**Evidence.** The weights are value-identical ([equivalence report](../../configs/sdcpp-weights.check.json)), with
matched workload and GPU type ([comparison](../../experiments/compare-a100-pytorch-vs-sdcpp-flux-klein.md#matching)).
PyTorch's large GEMMs reach 68–82% of the A100's BF16 peak ([record](../../experiments/a100-flux-klein-001.md#kernel-mix-within-one-denoise-step-profiled-generation-job-11807396)).

**Interpretation.** For these shapes, GEMM efficiency is a property of the library, not the engine.
The engines differ in what surrounds the GEMMs.

Related: [ggml FP32 activations](ggml-fp32-activations-add-conversion-and-idle-time.md), [RQ1](../rq/rq1.md).
