---
type: concept
summary: The dtype intermediate tensors are kept in between ops (BF16, FP16, FP32) — separate from weight precision, and a source of conversion overhead and numerical differences across engines.
status: active
aliases: [activation dtype, compute precision]
updated: 2026-09-28
---

# Activation precision

"BF16 weights" doesn't mean the same computation in every engine. What also matters is the dtype that
activations are kept in between ops, and the dtype each kernel computes in:
- **PyTorch reference**: weights and activations BF16 throughout ([record](../../experiments/a100-flux-klein-001.md#setup)).
- **stable-diffusion.cpp**: activations FP32 between ops, converted to BF16 for cuBLAS GEMMs, to FP16
  for flash attention; the VAE GEMMs run in FP16 ([comparison](../../experiments/compare-a100-pytorch-vs-sdcpp-flux-klein.md#matching)).

Consequences: extra conversion kernels ([finding](../findings/ggml-fp32-activations-add-conversion-and-idle-time.md)),
and outputs that are close but not bit-identical across engines even from the same initial noise.
Precision per component is a planner choice in the proposal, and low-bit storage alone doesn't imply
faster execution ([overview](../project/overview.md#optimization-families-registry)).
