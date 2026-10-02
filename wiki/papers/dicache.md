---
type: paper
summary: DiCache — shallow online probes control residual reuse and align cached trajectories; probe overhead and quality thresholds remain explicit costs.
status: ingested
ref: Bu et al., DiCache — Let Diffusion Model Determine Its Own Cache, arXiv 2508.17356v2
source: https://arxiv.org/html/2508.17356v2
updated: 2026-09-28
---

# DiCache

Sources: [paper v2](https://arxiv.org/html/2508.17356v2), [official implementation](https://github.com/Bujiazi/DiCache).
Review date: 2026-09-28; literature evidence only.

## Mechanism and evidence

- A shallow-layer probe estimates output change online. Accumulated estimated error controls cache
  refresh; full computation resumes from the probe when needed (§3.2,
  [paper](https://arxiv.org/html/2508.17356v2)).
- Dynamic trajectory alignment combines cached residuals using the current probe trajectory.
  Probe depth and reuse threshold remain configurable (§3.2–4.3,
  [paper](https://arxiv.org/html/2508.17356v2)).
- Experiments include Wan 2.1, HunyuanVideo and FLUX.1-dev; the paper also evaluates combination with
  Sparse VideoGen. Its limitations explicitly acknowledge probe cost (§4–5,
  [paper](https://arxiv.org/html/2508.17356v2)).

## Interpretation and integration checks

- Both adaptive caching and combining acceleration techniques precede this project. The proposed
  contribution needs joint-selection evidence under matched constraints
  ([paper](https://arxiv.org/html/2508.17356v2), [RQ3](../rq/rq3.md)).
- Measure probe time, residual-history memory, alignment work and refresh frequency. A feature-error
  threshold is not itself a guarantee of the proposal's output-quality requirement
  ([paper](https://arxiv.org/html/2508.17356v2), [quality protocol](../project/overview.md#evaluation-rules)).
- Check the target architecture and checkpoint before integration; FLUX.1 results do not establish
  FLUX.2 klein support ([code](https://github.com/Bujiazi/DiCache)).
- Scope: Week 6 second policy/runtime guards; RQ1 transfer and RQ3 overhead
  ([milestones](../project/milestones.md), [RQs](../project/overview.md#research-questions)).
