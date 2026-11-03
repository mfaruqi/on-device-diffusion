---
type: experiment-record
id: m5-flux-klein-shiftadd-002
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-002__attempt__20261102-first-rows, m5-flux-klein-shiftadd-002__attempt__20261102-spread-rows]
updated: 2026-11-02
---

# m5-flux-klein-shiftadd-002: diagonal activation-weighted slice fitting

## Question and setup

Week 5 precision calibration, RQ1/RQ3. Does a diagonal second-moment objective
improve actual output error at unchanged p4 storage? CPU numerical analysis of
saved BF16-origin inputs from [capture](a100-flux-klein-shiftadd-005.md).
Source weights are pinned distilled Klein. First two prompts fit group scale
selection; last two are calibration validation, not the project's held-out set.

Two row selections are separately retained: first 32 output rows, and 32 rows
spread across the complete matrix. Exact indices and configs are in
[spread evidence](../results/runs/m5-flux-klein-shiftadd-002__attempt__20261102-spread-rows/)
and [first-row evidence](../results/runs/m5-flux-klein-shiftadd-002__attempt__20261102-first-rows/).
Both use four sign planes and group size 128. Diagonal weights are E[x_i²],
normalized by their mean, from fitting inputs only. The radix exponent candidates
and packed size are unchanged. Evaluation uses FP32 matrix multiplication, not
BF16 GPU rounding or an integer kernel.

## Results

Mean per-call relative output L2 over eight calibration validation calls using
spread rows, from `aggregate-validation.json`:

| Slice | Fit | Original inputs | A8 simulation | A4 simulation |
|---|---|---:|---:|---:|
| Query | Weight-only | 0.081639 | 0.082107 | 0.242219 |
| Query | Diagonal activation | 0.045345 | 0.045960 | 0.222394 |
| Fused | Weight-only | 0.079554 | 0.081215 | 0.318894 |
| Fused | Diagonal activation | 0.080051 | 0.081780 | 0.320034 |

The first-row selection suggests improvement in both modules, whereas spread
rows only support improvement for the query slice. Selecting the first outputs
of a fused projection does not cover all its output regions.

## Interpretation and limits

The diagonal objective helps one sampled slice but not the other. Naive per-token
A4 causes substantially more error here than A8. These are layer-slice diagnostics,
not full-layer, image-quality, speed or energy results. No global application of
this fit is justified; correlations omitted by the diagonal model can matter.
