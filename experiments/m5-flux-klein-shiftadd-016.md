---
type: experiment-record
id: m5-flux-klein-shiftadd-016
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-016__attempt__20261103-all-single]
updated: 2026-11-03
---

# m5-flux-klein-shiftadd-016: upstream fits for all single-stream fused projections

## Setup

Week 5 precision variants, RQ1/RQ3. Expand the
[first fused-projection fit](m5-flux-klein-shiftadd-010.md) to all 20
`single_transformer_blocks.*.attn.to_qkv_mlp_proj` matrices. Each complete
27648×3072 matrix includes Q/K/V and MLP output rows. All use four upstream
BCQ planes, group 128, u8/shared-FP16 scales, superblock 256, act-order,
15 fitting iterations and damping 0.01. The BF16 source is the error reference.

Pinned distilled Klein revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
The [expanded BF16 capture](a100-flux-klein-shiftadd-activation-capture-002.md)
provides 16 calls per matrix. The first eight calls (256 sampled tokens from
two prompts) fit the weights; the other eight from two calibration-validation
prompts evaluate them. No held-out evaluation or quantized-prefix recalibration.

[Config](../shiftadd/configs/upstream-bcq-all-single-attention-001.json),
[raw evidence](../results/runs/m5-flux-klein-shiftadd-016__attempt__20261103-all-single/),
[metric definitions](../wiki/methods/baseline-metrics.md).
This is CPU fitting and numerical evaluation, not an inference benchmark.

## Results

The 20 matrices contain 1,698,693,120 weights. Total packed payload is
915,701,760 bytes at 4.3125 bits/weight. Files including headers and row indices
occupy 920,146,480 bytes (877.520 MiB), versus 3,397,386,240 BF16
payload bytes (3240 MiB): 72.92% smaller.
These sizes cover only this 20-matrix subset, not the complete transformer,
text encoder, VAE or runtime memory.

| Single-stream block | Mean fit output L2 | Mean calibration-validation output L2 |
|---|---:|---:|
| 0 | 0.00521787 | 0.05825082 |
| 1 | 0.00537447 | 0.05935484 |
| 2 | 0.00519245 | 0.05744581 |
| 3 | 0.00481843 | 0.05538851 |
| 4 | 0.00412230 | 0.04896448 |
| 5 | 0.00468462 | 0.05835145 |
| 6 | 0.00486117 | 0.06119784 |
| 7 | 0.00526971 | 0.06052589 |
| 8 | 0.00570101 | 0.06324972 |
| 9 | 0.00588856 | 0.06314332 |
| 10 | 0.00599917 | 0.06397153 |
| 11 | 0.00585408 | 0.06123952 |
| 12 | 0.00628384 | 0.06428344 |
| 13 | 0.00679948 | 0.06680988 |
| 14 | 0.00676708 | 0.06401862 |
| 15 | 0.00677629 | 0.06559224 |
| 16 | 0.00706982 | 0.06846225 |
| 17 | 0.00694990 | 0.06859109 |
| 18 | 0.00691954 | 0.07129996 |
| 19 | 0.00809426 | 0.08417453 |

All 320 recorded per-call errors reproduce after independent packed-artifact
reload, with matching hashes. The block-zero artifact is byte-identical to the
previous full fused fit. Packed decoding agrees with the upstream dense output;
no extra damping retry occurred.

## Interpretation

Upstream fitting now covers every fused QKV/MLP projection. Validation errors
exceed fit errors, so calibration coverage remains a limitation. Local errors do
not establish accumulated image behavior when all matrices are replaced together.
The artifacts retain arbitrary u8-scaled plane factors; they are not a native
GPU INT4 format. No GPU memory, latency or energy savings are measured here.

## Next experiment
- Combine these 20 matrices with the verified 30 dual-stream matrices in a matched 50-matrix image comparison.

## Matched conventional projection controls

Existing radix p4 and symmetric INT4 files were hash-checked and decoded for
the same 20 complete matrices. All 640 control comparisons on identical captured
inputs are finite. [Raw controls](../results/runs/m5-flux-klein-shiftadd-016__attempt__20261103-all-single/controls/) preserve
per-call errors, configs and artifact hashes. Means equally weight matrices and calls.

| Method | Mean fit output L2 | Mean calibration-validation output L2 |
|---|---:|---:|
| radix_p4 | 0.08551937 | 0.08605991 |
| uniform_int4 | 0.07677707 | 0.07727822 |
| upstream_u8 | 0.00593220 | 0.06321579 |

Upstream has lower mean calibration-validation error than INT4 for all 20
matrices. The methods differ in scale freedom, fitting objective and payload
size (4.3125 bits/weight upstream versus 4.25 for these controls). This is not
an isolated calibration ablation or evidence of combined image quality.
