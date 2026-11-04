---
type: experiment-record
id: m5-flux-klein-shiftadd-017
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-017__attempt__20261103-dual-activation]
updated: 2026-11-03
---

# m5-flux-klein-shiftadd-017: all dual-stream projection activation errors

## Setup

Week 5 precision variants, RQ1/RQ3. For each of the
[30 fitted dual-stream matrices](m5-flux-klein-shiftadd-015.md), compare original
captured inputs against grouped symmetric A8 and A4 inputs, group size 32.
Each activation case is evaluated separately for original BF16 weights and
fixed four-plane upstream BCQ weights. Outputs are compared with the original
weights and inputs using FP32 CPU matrix multiplication.

[Config](../shiftadd/configs/upstream-all-dual-activation-error-001.json),
[raw evidence](../results/runs/m5-flux-klein-shiftadd-017__attempt__20261103-dual-activation/),
[metric definitions](../wiki/methods/baseline-metrics.md).
Checkpoint, fitted artifacts and capture hashes are recorded. All 16 calls per
matrix are evaluated: eight fitting calls and eight calibration-validation calls,
from four calibration prompts total. The prefix was BF16. No held-out prompts
or image generation are used in this diagnostic.

## Results

All 2,880 comparisons are finite (30 matrices × 16 calls × two weight cases ×
three activation cases). The 480 original-input/upstream comparisons reproduce
the prior weight-only errors, within rtol 1e-5 and atol 1e-8; all 480 BF16/original
reference comparisons are exactly zero. Artifact hashes and full-row coverage
are checked by the runner.

The table aggregates calibration-validation calls only, equally weighted over
matrices and calls. Maximum is the largest individual call error.

| Weights | Inputs | Mean relative output L2 | Maximum relative output L2 |
|---|---|---:|---:|
| bf16 | original | 0.00000000 | 0.00000000 |
| bf16 | a8 | 0.00367765 | 0.00831872 |
| bf16 | a4 | 0.06366733 | 0.14260970 |
| upstream_u8 | original | 0.03889433 | 0.08868404 |
| upstream_u8 | a8 | 0.03907282 | 0.08883614 |
| upstream_u8 | a4 | 0.07419878 | 0.15181942 |

## Interpretation

Grouped A8 adds little aggregate error to the upstream-weight result in this
captured-input diagnostic; grouped A4 produces a substantially larger increase.
The BF16-weight controls show the same ordering for activation error alone.
This supports testing A8 in the image pipeline for this subset, but does not
establish combined-model quality or a bit-allocation rule for other layers.
The arithmetic uses reconstructed floating values, so it demonstrates neither
GPU latency/energy savings nor actual activation-memory reduction at runtime.

## Next experiment
- Add grouped A8 to all 30 upstream dual-stream projections after the weight-only image comparison.
