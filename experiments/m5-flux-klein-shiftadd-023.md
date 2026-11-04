---
type: experiment-record
id: m5-flux-klein-shiftadd-023
status: partial
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-023__attempt__20261103-selected-p2]
updated: 2026-11-03
---

# m5-flux-klein-shiftadd-023: selected fused projections at 2 planes

## Setup

Week 5 precision variants, RQ1/RQ3. Refit the same ten fused matrices selected in
[mixed allocation 022](m5-flux-klein-shiftadd-022.md) using 2 upstream BCQ planes.
The selection is fixed at blocks 0–8 and 11; no reselection using new errors.
Only plane count changes. Group 128, u8/shared FP16 scales, superblock 256,
15 fitting iterations, damping 0.01, act-order and scale refitting match the source.
Pinned checkpoint `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
The first eight BF16 calibration calls fit; the other eight validate.
No held-out prompts or sequential quantized-prefix calibration are used.

[Config](../shiftadd/configs/upstream-bcq-selected-fused-p2-001.json), [raw evidence](../results/runs/m5-flux-klein-shiftadd-023__attempt__20261103-selected-p2/),
[metrics](../wiki/methods/baseline-metrics.md).
CPU numerical fitting with four threads per process; no inference timing.

## Results

Fitting in progress. No quality or storage result claimed until independent reload verification.

## Next experiment
- Assemble with forty existing four-plane matrices and run matched calibration images.
