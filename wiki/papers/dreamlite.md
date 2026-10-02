---
type: paper
summary: DreamLite — compact distilled phone workload; Android paper timings use precomputed prompts, while the iOS reference uses MLX text encoding and Core ML diffusion stages.
status: ingested
ref: Feng et al., DreamLite — A Lightweight On-Device Unified Model for Image Generation and Editing, arXiv 2603.28713v1
source: https://arxiv.org/html/2603.28713v1
updated: 2026-09-28
---

# DreamLite

Sources: [paper v1](https://arxiv.org/html/2603.28713v1), [official repository](https://github.com/ByteVisionLab/DreamLite),
[iOS deployment guide](https://github.com/ByteVisionLab/DreamLite/tree/main/deploy).
Review date: 2026-09-28; source review only, no local export or device validation.

## Mechanism and evidence

- A compact pruned U-Net supports generation and editing via spatial latent concatenation. Progressive
  training and subsequent step distillation produce the mobile workload (§3,
  [paper](https://arxiv.org/html/2603.28713v1)).
- The Android deployment uses W8A8 U-Net execution and precomputed prompt embeddings. Its fast-phone
  claim must not be interpreted as fresh-prompt full-pipeline timing (§4.6,
  [paper](https://arxiv.org/html/2603.28713v1)).
- The iOS reference uses a quantized MLX text encoder with FP16 Core ML U-Net/VAE stages, and modified
  Swift model code for hidden-state extraction ([deployment guide](https://github.com/ByteVisionLab/DreamLite/tree/main/deploy)).
- The repository identifies separate base/mobile checkpoints and gated weight access
  ([repository](https://github.com/ByteVisionLab/DreamLite)).

## Interpretation and integration checks

- DreamLite is a separate workload, never an optimized FLUX result
  ([proposal scope](../project/overview.md#fallbacks-and-scope-control)).
- Measure text encoding, transfers, denoising and decoding for fresh prompts; label precomputed
  conditioning separately ([paper](https://arxiv.org/html/2603.28713v1),
  [measurement rules](../project/overview.md#evaluation-rules)).
- Verify weight access, checkpoint revision, export scripts, Swift dependency versions and actual
  device memory before claiming feasibility. The deployment guide's requirements are not a local test
  ([guide](https://github.com/ByteVisionLab/DreamLite/tree/main/deploy)).
- Scope: Week 1 review, Week 2 export/access verification, Week 4 capture; RQ1 stage placement and
  RQ3 constrained plans ([milestones](../project/milestones.md), [RQs](../project/overview.md#research-questions)).
