---
type: experiment-record
id: m5-flux-klein-shiftadd-019
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-019__attempt__20261103-single-activation]
updated: 2026-11-03
---

# m5-flux-klein-shiftadd-019: all single-stream fused projection activation errors

## Setup

Week 5 precision variants, RQ1/RQ3. For each of the
[20 fitted single-stream fused matrices](m5-flux-klein-shiftadd-016.md), compare original
captured inputs against grouped symmetric A8 and A4 inputs, group size 32.
Each activation case is evaluated separately for original BF16 weights and
fixed four-plane upstream BCQ weights. Outputs are compared with the original
weights and inputs using FP32 CPU matrix multiplication.

[Config](../shiftadd/configs/upstream-all-single-activation-error-001.json),
[raw evidence](../results/runs/m5-flux-klein-shiftadd-019__attempt__20261103-single-activation/),
[metric definitions](../wiki/methods/baseline-metrics.md).
Checkpoint, fitted artifacts and capture hashes are recorded. All 16 calls per
matrix are evaluated: eight fitting calls and eight calibration-validation calls,
from four calibration prompts total. The prefix was BF16. No held-out prompts
or image generation are used in this diagnostic.

## Results

All 1,920 comparisons are finite (20 matrices × 16 calls × two weight cases ×
three activation cases). The 320 original-input/upstream comparisons reproduce
the prior weight-only errors, within rtol 1e-5 and atol 1e-8; all 320 BF16/original
reference comparisons are exactly zero. Artifact hashes and full-row coverage
are checked by the runner.

The table aggregates calibration-validation calls only, equally weighted over
matrices and calls. Maximum is the largest individual call error.

| Weights | Inputs | Mean relative output L2 | Maximum relative output L2 |
|---|---|---:|---:|
| bf16 | original | 0.00000000 | 0.00000000 |
| bf16 | a8 | 0.00402667 | 0.00514219 |
| bf16 | a4 | 0.07291846 | 0.09335206 |
| upstream_u8 | original | 0.06321579 | 0.09599587 |
| upstream_u8 | a8 | 0.06334223 | 0.09611844 |
| upstream_u8 | a4 | 0.09589081 | 0.12514937 |

## Interpretation

Grouped A8 adds little aggregate error to the upstream-weight result in this
captured-input diagnostic; grouped A4 produces a substantially larger increase.
The BF16-weight controls show the same ordering for activation error alone.
This supports testing A8 in the image pipeline for this subset, but does not
establish combined-model quality or a bit-allocation rule for other layers.
The arithmetic uses reconstructed floating values, so it demonstrates neither
GPU latency/energy savings nor actual activation-memory reduction at runtime.

## Next experiment
- Add grouped A8 to all 20 upstream single-stream fused projections after the weight-only image comparison.
