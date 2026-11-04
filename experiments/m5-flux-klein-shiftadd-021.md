---
type: experiment-record
id: m5-flux-klein-shiftadd-021
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-021__attempt__20261103-single-p3]
updated: 2026-11-03
---

# m5-flux-klein-shiftadd-021: three-plane upstream fits for all single-stream fused projections

## Setup

Week 5 precision variants, RQ1/RQ3. Starting from the
[four-plane configuration](m5-flux-klein-shiftadd-016.md), reduce only the
plane count to three for all 20
`single_transformer_blocks.*.attn.to_qkv_mlp_proj` matrices. Each complete
27648×3072 matrix includes Q/K/V and MLP output rows. All use three upstream
BCQ planes, group 128, u8/shared-FP16 scales, superblock 256, act-order,
15 fitting iterations and damping 0.01. The BF16 source is the error reference.

Pinned distilled Klein revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
The [expanded BF16 capture](a100-flux-klein-shiftadd-activation-capture-002.md)
provides 16 calls per matrix. The first eight calls (256 sampled tokens from
two prompts) fit the weights; the other eight from two calibration-validation
prompts evaluate them. No held-out evaluation or quantized-prefix recalibration.

[Config](../shiftadd/configs/upstream-bcq-all-single-p3-001.json),
[raw evidence](../results/runs/m5-flux-klein-shiftadd-021__attempt__20261103-single-p3/),
[metric definitions](../wiki/methods/baseline-metrics.md).
This is CPU fitting and numerical evaluation, not an inference benchmark.

## Results

The 20 matrices contain 1,698,693,120 weights. Total packed payload is
690,094,080 bytes at 3.25 bits/weight. Files including headers and row indices
occupy 694,538,800 bytes (662.364 MiB), versus 3,397,386,240 BF16
payload bytes (3240 MiB): 79.56% smaller.
These sizes cover only this 20-matrix subset, not the complete transformer,
text encoder, VAE or runtime memory.

| Single-stream block | Mean fit output L2 | Mean calibration-validation output L2 |
|---|---:|---:|
| 0 | 0.01037037 | 0.10106220 |
| 1 | 0.01073347 | 0.10339389 |
| 2 | 0.01038860 | 0.10070918 |
| 3 | 0.00961946 | 0.09750907 |
| 4 | 0.00820977 | 0.08649984 |
| 5 | 0.00928641 | 0.10315259 |
| 6 | 0.00963040 | 0.10690021 |
| 7 | 0.01053101 | 0.10596646 |
| 8 | 0.01143771 | 0.11068509 |
| 9 | 0.01183647 | 0.11073115 |
| 10 | 0.01208227 | 0.11204863 |
| 11 | 0.01179223 | 0.10754398 |
| 12 | 0.01272256 | 0.11290751 |
| 13 | 0.01372739 | 0.11775261 |
| 14 | 0.01356816 | 0.11282465 |
| 15 | 0.01365070 | 0.11532110 |
| 16 | 0.01422467 | 0.12009461 |
| 17 | 0.01391124 | 0.11918821 |
| 18 | 0.01366818 | 0.12341652 |
| 19 | 0.01575197 | 0.14155172 |

All 320 recorded per-call errors reproduce after independent packed-artifact
reload, with matching hashes. Packed decoding agrees with the upstream dense
output; no extra damping retry occurred.

## Interpretation

Three-plane alternatives are saved for every fused QKV/MLP projection. Validation errors
exceed fit errors, so calibration coverage remains a limitation. Local errors do
not establish accumulated image behavior when all matrices are replaced together.
The artifacts retain arbitrary u8-scaled plane factors; they are not a native
GPU INT4 format. No GPU memory, latency or energy savings are measured here.

## Next experiment
- Select a bounded mixed-plane candidate using fitting-call errors, then validate its image behavior separately.
