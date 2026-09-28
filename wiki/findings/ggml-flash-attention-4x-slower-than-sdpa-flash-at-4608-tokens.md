---
type: finding
summary: For FLUX.2 klein's attention shape (B1, H24, S4608, D128) on A100-PCIE, ggml's F16 flash-attention kernel takes ~4× the time of PyTorch SDPA-flash.
status: supported
confidence: medium
rq: [RQ1]
sources: [../../experiments/compare-a100-pytorch-vs-sdcpp-flux-klein.md]
updated: 2026-09-28
---

# ggml flash attention is ~4× slower than PyTorch SDPA-flash at 4608 tokens

**Claim.** With the same model, shape and GPU, attention in one denoise step takes **37.3 ms**
([`stages.denoise_step_1.by_group.attention.ms`](../../results/runs/a100-flux-klein-001-20260923-221717-profile/profile/denoise_kernels.json))
with PyTorch SDPA-flash and **148.6 ms**
([`stages.denoise_step_1.by_group.attention.ms`](../../results/runs/a100-sdcpp-flux-klein-001-20260925-124703-profile/profile/denoise_kernels.json))
with sd.cpp's `flash_attn_ext_f16`. That is about 180 vs 44 TFLOP/s, and it accounts for most of the per-step gap between the engines.

**Evidence.** Profiled runs on A100-PCIE-40GB, 25 attention calls per step
([comparison](../../experiments/compare-a100-pytorch-vs-sdcpp-flux-klein.md#kernel-time-per-denoise-step-profiled-runs-mean-of-4-steps-ms)).

**Scope.** One shape, one GPU type, sd.cpp `master-919-19bbbca`. Confidence is medium until another
shape or GPU confirms it.

**Open.** The cause isn't established ([open question](../open-questions.md#why-is-ggmls-flash-attention-kernel-4-slower-than-pytorch-sdpa-flash-at-4608-tokens)).

Related: [flash attention](../concepts/flash-attention.md), [stable-diffusion.cpp](../systems/stable-diffusion-cpp.md), [RQ1](../rq/rq1.md).
