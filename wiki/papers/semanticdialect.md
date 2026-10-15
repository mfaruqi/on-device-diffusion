---
type: paper
summary: SemanticDialect — 4-bit mixed-format (per-block "dialect") PTQ for video DiTs with activation residual decomposition; speedups are projected, not measured, and need hardware support.
status: ingested
ref: Jang and Tambe (Stanford), SemanticDialect — Semantic-Aware Mixed-Format Quantization for Video Diffusion Transformers, arXiv 2603.02883v3
source: https://arxiv.org/abs/2603.02883v3
updated: 2026-10-10
---

# SemanticDialect

Sources: [paper v3](https://arxiv.org/abs/2603.02883v3).
Review date: 2026-10-10; literature evidence only.

## Mechanism and evidence

- FB4 format: each block of values picks one of 32 4-bit "dialects" (value grids) under a shared
  power-of-two scale, with 10 bits of metadata per block; selection uses lookup tables instead of
  per-value MSE search (§3.1–3.2, [paper](https://arxiv.org/abs/2603.02883v3)).
- Quantization-sensitive layers (modulation, final MLP, temporal-attention QKV) re-quantize the residual,
  Act ≈ Q(Act) + Q(Δ), in the same 4-bit format; this decomposition gives the largest ablation gain
  (§3.3, Table 3, [paper](https://arxiv.org/abs/2603.02883v3)).
- Semantic-aware dialect assignment ties attention-correlated tokens to a shared sub-formatbook; it is
  skipped in early timesteps and refreshed periodically, a form of cross-step reuse of quantization
  metadata (§3.4, [paper](https://arxiv.org/abs/2603.02883v3)).
- Evaluated on Open-Sora 1.0 and 2.0 (the latter FLUX-style) with VBench, against MXFP4, NVFP4,
  ViDiT-Q and Q-VDiT, on H100 (Table 1, [paper](https://arxiv.org/abs/2603.02883v3)).

## Interpretation and integration checks

- The memory savings are measured on H100; the latency result is a Blackwell MXFP8 projection, and
  GEMMs run in FP16 with on-the-fly decoding. Low-bit storage without native kernels is not a speedup
  ([paper](https://arxiv.org/abs/2603.02883v3), [overview](../project/overview.md#optimization-families-registry)).
- No Jetson, Metal or WebGPU kernels exist for FB4; it is an admission candidate only if a compatible
  kernel can be generated (e.g. through TVM) and its overhead measured
  ([D-011](../project/decisions.md#d-011-build-the-compiled-pipeline-mlc-style-on-tvm-with-an-engine-level-planner-fallback-proposed)).
- Hypothesis: per-layer sensitivity (which layers need decomposition) is the kind of per-component
  precision choice the planner selects; relevant to the selective-quantization item in the
  [October 7 memo](../../raw/meetings/2026-10-07.md).
- Video-only evaluation; transfer to FLUX.2 klein images is untested. Compare with
  [SVDQuant](svdquant.md) (low-rank outlier branch) before choosing a low-bit candidate.
- Scope: Week 5 compatible variants, Wan video workload once resumed; [RQ1](../rq/rq1.md), [RQ3](../rq/rq3.md).
