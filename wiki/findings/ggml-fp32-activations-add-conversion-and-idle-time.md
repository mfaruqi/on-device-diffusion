---
type: finding
summary: sd.cpp's FP32-activation design adds ~37 ms of dtype conversion, ~12 ms of host-device copies and ~20 ms of GPU idle per denoise step compared with PyTorch.
status: supported
confidence: medium
rq: [RQ1]
sources: [../../experiments/compare-a100-pytorch-vs-sdcpp-flux-klein.md, ../../experiments/a100-sdcpp-flux-klein-001.md]
updated: 2026-09-28
---

# ggml's FP32 activations cost ~70 ms per denoise step

**Claim.** Per step, sd.cpp spends **56.9 ms**
([`stages.denoise_step_1.by_group.copy/cast.ms`](../../results/runs/a100-sdcpp-flux-klein-001__profile__20260925-124703/profile/denoise_kernels.json))
on copy and dtype-conversion kernels, versus 19.6 ms in PyTorch. The GPU is idle **22.0 ms**
([`stages.denoise_step_1.idle_ms`](../../results/runs/a100-sdcpp-flux-klein-001__profile__20260925-124703/profile/denoise_kernels.json))
per step, versus about 2 ms in PyTorch. sd.cpp also copies about 10 ms per step between host and device.

**Evidence.** ggml keeps activations in FP32 between ops, and converts to BF16 for GEMMs and to FP16 for
flash attention ([comparison](../../experiments/compare-a100-pytorch-vs-sdcpp-flux-klein.md#matching)).

**Interpretation.** These costs belong to the engine design, not the model. They're candidates for
fusion, or for keeping activations in half precision.

Related: [activation precision](../concepts/activation-precision.md), [stable-diffusion.cpp](../systems/stable-diffusion-cpp.md).
