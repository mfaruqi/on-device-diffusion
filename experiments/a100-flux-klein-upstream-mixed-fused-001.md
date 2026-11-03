---
type: experiment-record
id: a100-flux-klein-upstream-mixed-fused-001
status: partial
device: a100-sxm4-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-mixed-fused-001__attempt__20261103-calibration]
updated: 2026-11-03
---

# a100-flux-klein-upstream-mixed-fused-001: mixed three/four-plane image diagnostic

Status: **partial**, job 12149223 queued for an A100 SXM4 allocation.

## Question

Does the [fit-selected mixed candidate](m5-flux-klein-shiftadd-022.md) preserve
calibration image behavior relative to the all-four-plane attention reference?
Week 5 precision variants, RQ1/RQ3. Only the plane allocation changes.

## Setup

[Configuration](../shiftadd/configs/a100-upstream-mixed-fused-001.json),
[Slurm launcher](../shiftadd/configs/a100-upstream-mixed-fused-001.slurm),
[raw evidence](../results/runs/a100-flux-klein-upstream-mixed-fused-001__attempt__20261103-calibration/),
[metric definitions](../wiki/methods/baseline-metrics.md).

Distilled Klein revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`,
512×512, four steps, guidance 1, batch one, BF16 activations.
Three cases: unchanged BF16, all 50 selected matrices at four planes, and the
saved candidate with ten fused matrices at three planes and forty matrices at four.
The other 59 matrix weights remain unchanged. The four prompts/seeds match the
[all-attention control](a100-flux-klein-upstream-all-attention-001.md): teapot 17,
lighthouse 23, lemons 31, robot 47. These are calibration prompts, not held-out
quality evaluation. One generation per case/prompt; no repeated timing protocol.

All selected weights reconstruct to dense BF16. This run cannot establish packed
GPU latency, resident-memory or energy benefits. The allocation is capped at ten
minutes and selects N-feature SXM4 nodes. All 100 selected source-file hashes
were verified on Gilbreth before submission.

## Results

Pending allocation (`AssocGrpGRES`: the shared account GPU allowance is occupied);
no images or new quality measurements claimed.

## Next experiment
- Validate the generated images, exact reference reproduction and matrix-replacement audits when the allocation completes.
