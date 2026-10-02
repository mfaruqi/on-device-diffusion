---
type: paper
summary: MetaSchedule — composable tensor-program search spaces, validated traces and measured-cost-guided search; precedent for RQ2, not evidence for diffusion plan quality.
status: ingested
ref: Shao et al., Tensor Program Optimization with Probabilistic Programs, NeurIPS 2022, arXiv 2205.13603v2
source: https://arxiv.org/html/2205.13603v2
updated: 2026-09-28
---

# MetaSchedule

Sources: [paper v2](https://arxiv.org/html/2205.13603v2), [MLC Web SD usage](https://github.com/mlc-ai/web-stable-diffusion#how).
Review date: 2026-09-28; literature evidence only.

## Mechanism and evidence

- Program analysis, sampled choices and transformations form composable search spaces. Later choices
  depend on the transformed program rather than an independent grid of settings (§3,
  [paper](https://arxiv.org/html/2205.13603v2)).
- Search mutates recorded traces, rejects invalid candidates and uses a learned latency model updated
  from hardware measurements. The default predictor uses boosted trees (§4,
  [paper](https://arxiv.org/html/2205.13603v2)).
- Web Stable Diffusion uses TensorIR/MetaSchedule and preserves tuned transformations for later builds
  ([MLC source](https://github.com/mlc-ai/web-stable-diffusion#how)).

## Interpretation and integration checks

- Search-space composition, measured-cost ranking and reusable tuning history are established
  precedents ([paper](https://arxiv.org/html/2205.13603v2), [MLC review](mlc-compiler-precedent.md)).
- The proposed planner adds diffusion-region choices, pipeline residency and quality eligibility to
  its decision problem; kernel schedule validity alone does not establish acceptable approximation
  error ([proposed objective](../project/overview.md#planner-objective)).
- Keep performance history keyed by graph, workload, device and implementation versions; maintain
  quality eligibility separately. Compare rules, lookup and learned ranking under equal profiling
  budgets, including search cost ([history design](../project/overview.md#system-components), [RQ2](../rq/rq2.md)).
- Scope: Week 2 history schema, Weeks 5–6 search and ranking; RQ2
  ([milestones](../project/milestones.md)).
