---
type: paper
summary: ShiftAddLLM — post-training reparameterization of LLM linear weights into binary planes with power-of-two scales, evaluated by LUT-based shift-and-add kernels; weight-only, activations stay FP16.
status: ingested
ref: You, Guo, Fu, Zhou, Shi, Zhang, Kundu, Yazdanbakhsh and Lin, ShiftAddLLM — Accelerating Pretrained LLMs via Post-Training Multiplication-Less Reparameterization, NeurIPS 2024, arXiv 2406.05981v4
source: https://arxiv.org/abs/2406.05981v4
updated: 2026-10-26
---

# ShiftAddLLM

Sources: [paper v4](https://arxiv.org/abs/2406.05981v4), [HTML](https://arxiv.org/html/2406.05981v4).
Review date: 2026-10-26; literature evidence only. The paper is named as the starting point for
diffusion shift-add work in the [October 7 memo](../../raw/meetings/2026-10-07.md#to-do).

## Mechanism and evidence

- Each linear weight matrix is approximated by binary-coding quantization, W ≈ Σ αᵢbᵢ with
  bᵢ ∈ {−1, +1}; the scales are rounded to powers of two, so scaling becomes a shift (§3, Eq. 2,
  [paper](https://arxiv.org/html/2406.05981v4)).
- Activations stay FP16. For every 8 activations the kernel precomputes a 256-entry table of signed
  partial sums; each plane's 8 sign bits index it, and the partial sums are added (§4.1,
  [paper](https://arxiv.org/html/2406.05981v4)).
- The fit jointly reduces weight error and output-activation error. Column-wise scales give the
  accuracy variant; block-wise scales fit the LUT kernel and give the latency variant (§4.2,
  [paper](https://arxiv.org/html/2406.05981v4)).
- Mixed precision: a per-layer sensitivity proxy and integer programming assign 2, 3 or 4 bits under an
  average budget; K projections and the first MLP layer receive more bits (§4.3,
  [paper](https://arxiv.org/html/2406.05981v4)).
- Evaluated on OPT, LLaMA and other families with perplexity and zero-shot tasks; latency on one
  A100-80GB with custom CUDA. Energy comes from an Eyeriss-like accelerator model, not measured silicon
  (§5, [paper](https://arxiv.org/html/2406.05981v4)).

## Interpretation and integration checks

- The LUT is built per activation vector and shared by every output row, which suits batch-1 LLM decode.
  A klein denoise call processes 4,096 image and 512 text tokens at 1024²
  ([model](../systems/flux2-klein-4b.md)), so each token needs its own tables. Interpretation: for
  the token-heavy transformer linears, the LUT kernel is unlikely to beat tensor-core GEMMs, which already run
  near BF16 peak on A100 ([finding](../findings/pytorch-denoise-step-third-in-unfused-memory-bound-ops.md)).
  The format may still pay off through memory, for single-vector layers, and on devices without strong
  matrix units.
- Weight-only storage reduces memory but is not a speedup without a compatible kernel
  ([overview](../project/overview.md#optimization-families-registry), [SVDQuant](svdquant.md)).
- Hypothesis: binary planes nest, so reading the first k planes gives a coarser version of the same weights
  without a second copy. That would make precision selectable per step or per memory budget at run time.
  Untested; truncated sums would need their own scales.
- A ShiftAdd checkpoint is a compressed checkpoint, reported separately from execution-only changes
  ([evaluation rules](../project/overview.md#evaluation-rules)).
- Successor implementation and kernel changes: [ShiftAddQwen](shiftaddqwen.md).
- Scope: Week 5 quantized variant ([milestones](../project/milestones.md)); per-layer and per-step
  precision for [RQ3](../rq/rq3.md); kernel transfer across devices for [RQ1](../rq/rq1.md).
