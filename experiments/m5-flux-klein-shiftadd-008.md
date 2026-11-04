---
type: experiment-record
id: m5-flux-klein-shiftadd-008
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-008__attempt__20261102-u8, m5-flux-klein-shiftadd-008__attempt__20261102-pot]
updated: 2026-11-02
---

# m5-flux-klein-shiftadd-008: pinned ShiftAddQwen fitter applied to FLUX weight slices

## Setup

Week 5 precision calibration, RQ1/RQ3. Apply upstream `saq.gptq_bcq.quantize`
from ShiftAddQwen commit `12d995a51a514b3757e29213c5545004e2e0e347` to the
same two 32-spread-row FLUX slices used in prior numerical diagnostics.
The [source](https://github.com/Purdue-EcoAI/ShiftAddQwen/blob/12d995a51a514b3757e29213c5545004e2e0e347/saq/gptq_bcq.py)
and `bcq.py` are vendored without algorithm edits under their MIT license;
source hashes and license are preserved in `shiftadd/vendor/shiftaddqwen/`.

Four planes, group 128, superblock 256, block size 128, 15 fitting iterations,
one sweep/refit round, group-preserving activation order and diagonal damping
0.01 times the mean Hessian diagonal. H=XᵀX/N uses the first eight captured calls
(256 token samples, first two calibration prompts). Remaining eight calls are
calibration validation, not the project held-out set. Input dimension is 3072,
so the empirical Hessian is rank-deficient; damping is material to the fit.
The upstream implementation's extra damping retries did not occur (empty fit logs).
The adapter fails rather than accepting an unconfigured extra-damping retry.

Compare two scale storage formats; only this setting changes between runs:
`u8` (uint8 scale codes times a shared FP16 scale) and `pot` (four-bit exponents
relative to a shared FP16 scale). Neither is identical to the earlier pure
power-of-two exponent-only baseline. This is a slice application of the upstream
fitter, not the full sequential Qwen quantization, mixed allocation or distillation
pipeline. All evaluation products are FP32 with original captured inputs.

## Results

Mean per-call relative output L2. Payload bits include scales, excluding file
headers and output-row metadata.

| Scale format | Slice | Payload bits/weight | Fitting calls | Calibration-validation calls |
|---|---|---:|---:|---:|
| u8 | transformer_blocks | 4.3125 | 0.002342 | 0.017345 |
| u8 | single_transformer_blocks | 4.3125 | 0.006079 | 0.067605 |
| pot | transformer_blocks | 4.1875 | 0.017469 | 0.037229 |
| pot | single_transformer_blocks | 4.1875 | 0.014111 | 0.096809 |

Each u8 slice uses 52,992 payload bytes; pot uses 51,456. Codes are saved as actual
packed sign planes; pot exponents are packed into nibbles. All 64 recorded output
metrics reproduce after independently reloading packed files. File hashes,
reconstruction checks and results are archived:
- [m5-flux-klein-shiftadd-008__attempt__20261102-u8](../results/runs/m5-flux-klein-shiftadd-008__attempt__20261102-u8/)
- [m5-flux-klein-shiftadd-008__attempt__20261102-pot](../results/runs/m5-flux-klein-shiftadd-008__attempt__20261102-pot/)

## Chunked decoder verification

Independent per-group scale reconstruction matches the chunked loader exactly
for row chunks of 1, 7 and 32 on both matrices and both scale formats. All
1,179,648 compared values match; per-format evidence is saved as
`chunk-decoder-verification.json` in the run directories above. This check
covers decoding of the saved slices, not full-model execution.

## Quantized activation arithmetic verification

A CPU reference multiplies grouped A4/A8 integer activations by unpacked sign
planes using INT32 sums, followed by floating scale multiplication. It does not
reconstruct dense weights in the arithmetic path. Across both matrices, both
scale formats and all 16 captured calls, 128 checks agree with decoded-weight
matmul within relative L2 tolerance 1e-5. Activations use group size 32.
Per-call errors and artifact hashes are preserved in
`integer-arithmetic-verification.json` in each run directory above. This proves
numerical equivalence of the CPU reference, not a packed GPU implementation,
activation quality acceptance, or energy savings.

## Interpretation and limits

The u8 scale option has lower validation error on both slices at greater storage.
Large fitting/validation gaps, especially in the fused projection, limit confidence
in the small calibration set. The result motivates broader calibration before
full-transformer conversion. Comparing with earlier strict radix results changes
both the fitter and the scale family, so improvements cannot be attributed solely
to the Hessian sweep. No generated images, latency, live-memory or energy result
is established for these newly fitted slices.
