---
type: paper
summary: TeaCache — adaptive residual reuse from timestep-modulated input differences; polynomial calibration and threshold selection are part of its integration cost.
status: ingested
ref: Liu et al., Timestep Embedding Tells — It's Time to Cache for Video Diffusion Model, CVPR 2025, arXiv 2411.19108v2
source: https://arxiv.org/html/2411.19108v2
updated: 2026-09-28
---

# TeaCache

Sources: [paper v2](https://arxiv.org/html/2411.19108v2), [official implementation](https://github.com/ali-vilab/TeaCache).
Review date: 2026-09-28; literature evidence only.

## Mechanism and evidence

- Differences in timestep-modulated inputs estimate output changes. A fitted polynomial rescales the
  indicator; accumulated change triggers recomputation at a chosen threshold (§3.3,
  [paper](https://arxiv.org/html/2411.19108v2)).
- The cached quantity is the transformer residual. Reuse still updates the model output with the
  current input; it is not simply removal of sampler steps (§3.4,
  [paper](https://arxiv.org/html/2411.19108v2)).
- Main video experiments use A800 GPUs, polynomial fitting on sampled prompts, and VBench plus paired
  image metrics. The official repository also supplies FLUX and Wan integrations (§4,
  [paper](https://arxiv.org/html/2411.19108v2), [code](https://github.com/ali-vilab/TeaCache)).

## Interpretation and integration checks

- Adaptive reuse is existing work. Our registry should record the supported region, polynomial,
  threshold, residual buffers and refresh behavior; quality eligibility must use separate calibration
  data ([paper](https://arxiv.org/html/2411.19108v2), [registry contract](../project/overview.md#system-components)).
- **Hypothesis:** calibration, cache traffic and decisions may outweigh saved compute on a short
  schedule. Compare against no reuse and count the full overhead
  ([proposal](../project/overview.md#planner-objective)).
- Before integration, verify the exact checkpoint and schedule; repository support for FLUX does not
  establish FLUX.2 klein feasibility ([code](https://github.com/ali-vilab/TeaCache)).
- Scope: Week 4 first policy; RQ1 transfer and RQ3 runtime overhead
  ([milestones](../project/milestones.md), [RQs](../project/overview.md#research-questions)).
