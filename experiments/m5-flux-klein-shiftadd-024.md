---
type: experiment-record
id: m5-flux-klein-shiftadd-024
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-024__attempt__20261103-selected-p1]
updated: 2026-11-04
---

# m5-flux-klein-shiftadd-024: selected fused projections at 1 planes

## Setup

Week 5 precision variants, RQ1/RQ3. Refit the same ten fused matrices selected in
[mixed allocation 022](m5-flux-klein-shiftadd-022.md) using 1 upstream BCQ planes.
The selection is fixed at blocks 0–8 and 11; no reselection using new errors.
Only plane count changes. Group 128, u8/shared FP16 scales, superblock 256,
15 fitting iterations, damping 0.01, act-order and scale refitting match the source.
Pinned checkpoint `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
The first eight BF16 calibration calls fit; the other eight validate.
No held-out prompts or sequential quantized-prefix calibration are used.

[Config](../shiftadd/configs/upstream-bcq-selected-fused-p1-001.json), [raw evidence](../results/runs/m5-flux-klein-shiftadd-024__attempt__20261103-selected-p1/),
[metrics](../wiki/methods/baseline-metrics.md).
CPU numerical fitting with four threads per process; no inference timing.

## Results

All ten matrices were fitted and independently reloaded. All 160 per-call
output-error comparisons reproduce, and all artifact hashes match.
The ten files occupy 121,661,720 bytes.
Assembly with the forty unchanged four-plane matrices occupies 735,120,760 bytes
(0.684635 GiB), 31.52% less than
all fifty four-plane files. This is subset file storage, not complete model
storage or live memory. [Aggregates](../results/runs/m5-flux-klein-shiftadd-024__attempt__20261103-selected-p1/aggregate.json).

Mean relative output L2 over fitting calls: 0.08491021;
over calibration-validation calls: 0.36852157.
These local errors use BF16-prefix inputs and do not model error propagation.

## Next experiment
- [Fixed-selection image diagnostic](a100-flux-klein-upstream-lowplane-fused-001.md).
