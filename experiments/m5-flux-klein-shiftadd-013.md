---
type: experiment-record
id: m5-flux-klein-shiftadd-013
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-013__attempt__20261102-kv-p3]
updated: 2026-11-02
---

# m5-flux-klein-shiftadd-013: three-plane upstream K/V fits

## Setup

Week 5 precision variants, RQ1/RQ3. Reduce the upstream BCQ plane count from
four to three for the complete first-block K and V matrices. The reference is
[the four-plane K/V fit](m5-flux-klein-shiftadd-011.md). Checkpoint revision,
calibration capture, output rows, group size 128, u8 scale format, superblock 256,
15 fitting iterations, act-order and damping 0.01 remain identical.
The fit uses the first eight captured calls (256 sampled tokens); eight other
calls from two separate calibration prompts supply validation. No held-out
prompts are used. Shared-input provenance is in the reference record.

[Config](../shiftadd/configs/upstream-bcq-full-kv-p3-001.json),
[raw conversion and reload evidence](../results/runs/m5-flux-klein-shiftadd-013__attempt__20261102-kv-p3/),
and [metric definitions](../wiki/methods/baseline-metrics.md).
This is CPU fitting and projection-error evaluation, with no image generation
or performance measurement. Upstream source hashes are verified before fitting.

## Results

| Projection | Saved file bytes | Payload bits/weight | Mean fit output L2 | Mean calibration-validation output L2 |
|---|---:|---:|---:|---:|
| K | 3,859,484 | 3.2500 | 0.00446693 | 0.03211501 |
| V | 3,859,484 | 3.2500 | 0.00788445 | 0.06174716 |

Each artifact contains 3,833,856 packed payload bytes, versus 18,874,368 original
BF16 bytes. Counting the artifact header, file size is reduced by
79.55% against BF16 payload.
All 32 per-call error measurements reproduce after independent artifact reload;
artifact hashes match. Packed decoding also agrees with the upstream fitter's
returned dense weights. No extra damping retry occurred.

## Delta versus four-plane reference

The four-plane reference uses 5,112,860 file bytes per matrix; three planes save
1,253,376 file bytes (24.51%) per matrix. K calibration-validation error
increases from 0.02000314 to 0.03211501; V increases from 0.03411849 to 0.06174716.
Reference values come from the [four-plane record and controls](m5-flux-klein-shiftadd-011.md).
The representation uses three sign planes plus scale metadata, not a standard
GPU INT3 compute format.

## Interpretation

Three planes trade more projection error for lower storage in both matrices.
These measurements supply candidates for a mixed-precision selection; they do
not establish that K or V is safe to reduce in the complete pipeline. The fit
and validation split contain only four calibration prompts in total, and the
inputs come from an unquantized model prefix. Image quality, GPU resident memory,
latency and energy remain unmeasured for these weights.

## Next experiment
- Compare a three-plane K or V replacement against the paired four-plane Q/K/V image reference.
