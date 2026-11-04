---
type: experiment-record
id: m5-flux-klein-shiftadd-011
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-011__attempt__20261102-full-kv]
updated: 2026-11-02
---

# m5-flux-klein-shiftadd-011: upstream BCQ conversion of first-block K and V

## Setup
Week 5 precision variants, RQ1/RQ3. Local CPU conversion of the full 3072×3072
K and V matrices in transformer block 0 of distilled Klein revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`.
Pinned upstream ShiftAddQwen fitter, four planes, group 128, u8 scale codes,
FP16 shared scales, act-order GPTQ, 15 fitting iterations, damping 0.01.
Only the target matrices change relative to the complete-query fitting setup.

The [original capture](a100-flux-klein-shiftadd-005.md) supplies Q inputs.
The installed FLUX implementation passes the same hidden_states object to Q,
K and V; this was checked in local and remote source, and with a local same-object
projection probe. Derived K/V input arrays are byte-identical copies of those
Q arrays, with explicit [mapping and hashes](../results/runs/m5-flux-klein-shiftadd-011__attempt__20261102-full-kv/input-provenance.json).
No new samples were captured. The first eight calls (256 sampled tokens from
teapot/lighthouse prompts) fit the weights; the remaining eight calls are separate
calibration-validation inputs. All inputs come from the BF16 model prefix.

[Configuration, storage, numerical results and reload verification](../results/runs/m5-flux-klein-shiftadd-011__attempt__20261102-full-kv/).
[Metric definitions](../wiki/methods/baseline-metrics.md).
No pipeline execution or performance measurement is part of this conversion.

## Results

| Projection | File bytes | Payload bits/weight | Mean fit output L2 | Mean calibration-validation output L2 |
|---|---:|---:|---:|---:|
| to_k | 5,112,860 | 4.3125 | 0.00259261 | 0.02000314 |
| to_v | 5,112,860 | 4.3125 | 0.00422330 | 0.03411849 |

Each BF16 matrix has 18,874,368 payload bytes. Each compressed artifact has
5,087,232 payload bytes and 5,112,860 file bytes, a 72.91% reduction against
BF16 payload when counting the compressed file header. Metadata files are
reported separately from these per-matrix artifact sizes.

All 32 per-call output errors were recomputed from reloaded artifacts using the
independent chunked decoder and agreed within rtol 1e-5, atol 1e-8.
Artifact hashes and full output-row coverage were checked by conversion.

## Interpretation
The saved artifacts extend upstream conversion to first-block K and V.
The larger calibration-validation errors show a fitting/generalization gap;
these layer-output errors do not establish full-model image quality.
Storage savings do not establish resident GPU memory, latency or energy savings.

## Next experiment
- Compare the combined first-block Q/K/V replacements with paired BF16 and conventional controls.

## Quantized activation arithmetic verification

The complete K/V matrices passed 64 CPU arithmetic comparisons (two matrices,
16 captured calls, A4/A8 with activation group size 32). INT32 sign-plane sums
with the saved upstream scales agree with independently decoded-weight FP32
multiplication; maximum relative disagreement is 9.28713e-7.
[Verification evidence](../results/runs/m5-flux-klein-shiftadd-011__attempt__20261102-full-kv/integer-arithmetic-verification.json).
This checks arithmetic equivalence against already quantized activations, not
quality versus unquantized activations or GPU throughput.

## Same-source conventional controls

The existing radix p4 and uniform INT4 artifacts were decoded and evaluated on
all 3072 output rows, using the same captured inputs and split as the upstream
fit. Artifact SHA-256 checks passed. Each control has 16 per-call measurements
per matrix; all 64 errors are finite. Products use FP32 on the local CPU.

| Projection | Fit | Payload bits/weight | Mean fitting-call L2 | Mean calibration-validation L2 |
|---|---|---:|---:|---:|
| K | radix_p4 | 4.2500 | 0.05882893 | 0.05828238 |
| K | uniform_int4 | 4.2500 | 0.04914532 | 0.04868425 |
| K | upstream BCQ u8 | 4.3125 | 0.00259261 | 0.02000314 |
| V | radix_p4 | 4.2500 | 0.07973508 | 0.08090911 |
| V | uniform_int4 | 4.2500 | 0.06841496 | 0.06931517 |
| V | upstream BCQ u8 | 4.3125 | 0.00422330 | 0.03411849 |

[Per-call K control evidence](../results/runs/m5-flux-klein-shiftadd-011__attempt__20261102-full-kv/k-controls.json)
and [V control evidence](../results/runs/m5-flux-klein-shiftadd-011__attempt__20261102-full-kv/v-controls.json).
The upstream fit has lower output error on both matrices in both splits, with
a slightly larger payload. This comparison changes fitting method and scale
freedom together; it does not isolate calibration awareness or establish an
equal-storage advantage. The calibration-validation split contains only two
prompts and is not the formal held-out evaluation. No image-quality, latency,
resident-memory or energy claim follows from these projection errors.
