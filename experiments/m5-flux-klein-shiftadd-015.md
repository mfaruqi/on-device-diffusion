---
type: experiment-record
id: m5-flux-klein-shiftadd-015
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-015__attempt__20261103-all-dual]
updated: 2026-11-03
---

# m5-flux-klein-shiftadd-015: upstream fits for all dual-stream Q/K/V matrices

## Setup

Week 5 precision variants, RQ1/RQ3. Expand the four-plane upstream BCQ procedure
to all 30 image/text Q/K/V matrices in the five dual-stream transformer blocks.
The BF16 source weights are the projection-error reference. The fitting method,
group 128, u8/shared-FP16 scales, superblock 256, act-order, 15 iterations and
damping 0.01 match the earlier [query](m5-flux-klein-shiftadd-009.md) and
[K/V](m5-flux-klein-shiftadd-011.md) fits.

Pinned distilled Klein revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
The [expanded capture](a100-flux-klein-shiftadd-activation-capture-002.md)
provides 16 calls per matrix; the first eight (256 tokens from two prompts)
are fitting inputs and the other eight are calibration-validation inputs.
All captured prefixes are BF16; no sequential quantized-prefix recalibration
or held-out evaluation is performed.

[Configuration](../shiftadd/configs/upstream-bcq-all-dual-attention-001.json),
[raw fitting and verification evidence](../results/runs/m5-flux-klein-shiftadd-015__attempt__20261103-all-dual/),
[metric definitions](../wiki/methods/baseline-metrics.md).
This is CPU fitting and numerical validation, with no inference benchmark.

## Results

All 30 matrices are saved at four planes plus scale metadata, 4.3125 payload
bits per weight. Total packed payload: 152,616,960 bytes. Artifact files including
headers and row indices: 153,385,800 bytes (146.280 MiB), versus
566,231,040 BF16 payload bytes (540 MiB), a 72.91% reduction.
These values cover this 30-matrix subset, excluding other transformer weights,
text encoder and VAE. They are file sizes, not live memory.

Per-family errors below aggregate five blocks, eight calls per split per matrix;
the maximum column is the largest individual calibration-validation call.

| Projection family | Mean fit output L2 | Mean validation output L2 | Maximum validation call L2 |
|---|---:|---:|---:|
| add_q_proj | 0.00199681 | 0.03229832 | 0.05109317 |
| add_k_proj | 0.00188898 | 0.02933882 | 0.04642887 |
| add_v_proj | 0.00307939 | 0.04876184 | 0.06873094 |
| to_q | 0.00320001 | 0.03285683 | 0.04943831 |
| to_k | 0.00303388 | 0.03059383 | 0.04676601 |
| to_v | 0.00573367 | 0.05951632 | 0.08868404 |

All 480 saved-artifact per-call errors reproduce independently, and hashes match.
First-block image Q/K/V artifacts are byte-identical to the previous fits.
The converter checks packed decoding against the upstream dense output and
rejects extra damping retries; this run completed successfully.

## Interpretation

The method now covers every dual-stream attention Q/K/V matrix, including the
text branch. Calibration-validation errors exceed fitting errors, so the small
fit set does not justify a general quality claim. The saved weights can be used
for a broader image diagnostic; quality under combined replacement remains
unmeasured here. GPU resident-memory, latency and energy benefits are not
established by packed storage or CPU projection checks.

## Next experiment
- Compare all 30 replacements against matched BF16, radix and conventional INT4 image controls.
