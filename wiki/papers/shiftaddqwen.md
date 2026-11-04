---
type: paper
summary: ShiftAddQwen — ShiftAddLLM's binary-plane format refitted for Qwen3.5-9B with an output-aware GPTQ fit, measured mixed precision and scale distillation, run through a LUT-free patched llama.cpp (CUDA, CPU).
status: ingested
ref: Purdue EcoAI Lab, ShiftAddQwen repository and notes "ShiftAddQwen mat-vec — symbols and the sign-select identity", commit 12d995a5 (2026-10-23)
source: https://github.com/Purdue-EcoAI/ShiftAddQwen/tree/12d995a51a514b3757e29213c5545004e2e0e347
updated: 2026-10-26
---

# ShiftAddQwen

Sources: [README](https://github.com/Purdue-EcoAI/ShiftAddQwen/blob/12d995a51a514b3757e29213c5545004e2e0e347/README.md),
[mat-vec notes](https://github.com/Purdue-EcoAI/ShiftAddQwen/blob/12d995a51a514b3757e29213c5545004e2e0e347/docs/shiftadd_matvec.tex),
both at commit `12d995a5`. Review date: 2026-10-26; repository and documentation evidence only, not run here.

## Mechanism and evidence

- Keeps the [ShiftAddLLM](shiftaddllm.md) weight format: each row is Q sign planes with one scale per plane
  and group of 64, 128 or 256 weights, Q = 2 to 6, about Q + 0.3 bits per weight at group 128
  (notes §1, §6.1).
- Replaces the kernel. Decode quantizes activations to int8 (one scale per 32) and uses
  Σ bⱼyⱼ = 2·Σ₍bitⱼ=1₎ yⱼ − Σ yⱼ: sign bits become `dp4a` byte masks, with no lookup table to build or
  store. Prefill dequantizes the planes to FP16 and runs cuBLAS GEMMs on tensor cores (notes §3–5).
  The README lists an MMQ path for prefill batches below 64 and cuBLAS above.
- Replaces the fit: a GPTQ column sweep with act-order against the layer Hessian, scales re-solved for
  output error, bits and group size per linear chosen by a knapsack over measured end-to-end KLD, then
  scale distillation against the BF16 teacher (notes §5, §6.2).
- Converts the Qwen3.5-9B decoder linears; keeps DeltaNet gates, convolutions, norms, embedding (Q3_K),
  LM head (Q5_K) and the MTP block in llama.cpp formats (README).
- Kernels exist for CUDA and CPU (AVX2, AVX-512, NEON); no Metal path is listed (README).
- Reported on A100: lower KLD than k-quants of equal or larger size from 3.9 to 5.1 GB, and higher decode and
  prefill throughput than Q4_K_M (notes §6, README). Numbers stay in the repository.

## Interpretation and integration checks

- The repository's own prefill path supports the [ShiftAddLLM](shiftaddllm.md#interpretation-and-integration-checks)
  reading: when many tokens share each weight, the binary planes serve as a storage format and are
  expanded for tensor-core GEMMs. klein's denoise calls are of that kind
  ([model](../systems/flux2-klein-4b.md)).
- Transferable to a diffusion pipeline: the fit, the sensitivity-driven knapsack and scale distillation.
  Changes needed: calibration activations at the sampled timesteps, a velocity- or image-level
  sensitivity signal instead of next-token KLD, and a kernel for the compiled TVM path on CUDA, Metal and
  WebGPU ([D-011](../project/decisions.md#d-011-build-the-compiled-pipeline-mlc-style-on-tvm-with-an-engine-level-planner-fallback-proposed)).
- klein's Qwen3 text encoder holds half of the BF16 weights for a small share of warm latency
  ([finding](../findings/text-encoder-half-of-weights-little-of-time.md)); it is the component closest to
  the repository's target. Its conditioning output, not next-token KLD, is the loss to preserve.
- Jetson SM87 and Metal support are not established ([Nunchaku precedent](../methods/nunchaku-howto.md)).
- Scope: Week 5 quantized variant ([milestones](../project/milestones.md)); [RQ1](../rq/rq1.md),
  [RQ3](../rq/rq3.md).
