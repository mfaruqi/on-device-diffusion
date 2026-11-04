---
type: experiment-record
id: a100-flux-klein-upstream-lowplane-fused-001
status: partial
device: a100-sxm4-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-lowplane-fused-001__attempt__20261103-calibration]
updated: 2026-11-03
---

# a100-flux-klein-upstream-lowplane-fused-001: fixed-selection one-to-four-plane image diagnostic

## Setup

Week 5 precision variants, RQ1/RQ3. Extend the
[mixed-fused image reference](a100-flux-klein-upstream-mixed-fused-001.md)
with two-plane and one-plane fits for exactly the same ten fused matrices.
All other forty converted matrices stay at four planes; other transformer
weights and all activations stay BF16. No reselection or new prompts.

[Config](../shiftadd/configs/a100-upstream-lowplane-fused-001.json),
[Slurm launcher](../shiftadd/configs/a100-upstream-lowplane-fused-001.slurm),
[raw evidence](../results/runs/a100-flux-klein-upstream-lowplane-fused-001__attempt__20261103-calibration/),
[metrics](../wiki/methods/baseline-metrics.md).

Five cases: BF16, all-four-plane, mixed three/four, mixed two/four, mixed one/four.
Each uses the same four calibration prompts and seeds, 512×512, four steps,
guidance 1, batch one and pinned revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
One image per prompt/case; no repeat timing or formal held-out quality evaluation.
Weights reconstruct into dense BF16. This cannot establish packed inference
latency, runtime memory or energy gains.

## Status

Awaiting completed and independently verified [two-plane](m5-flux-klein-shiftadd-023.md)
and [one-plane](m5-flux-klein-shiftadd-024.md) fits. Local orchestration checks both
fits, assembles the candidates, verifies remote artifact hashes and then submits
one A100 SXM4 job capped at fifteen minutes. No image results claimed yet.

## Next experiment
- Validate paired references and all twenty generated images after the allocation completes.
