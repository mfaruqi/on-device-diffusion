---
type: experiment-record
id: m5-flux-klein-shiftadd-014
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-014__attempt__20261103-q-p3]
updated: 2026-11-03
---

# m5-flux-klein-shiftadd-014: three-plane upstream query fit

## Setup

Week 5 precision variants, RQ1/RQ3. Reduce the complete first-block query
matrix from four to three upstream BCQ sign planes, relative to the
[four-plane query reference](m5-flux-klein-shiftadd-009.md).
Checkpoint revision, capture, all 3072 output rows, group size 128,
u8 scale format, superblock 256, 15 fit iterations and damping 0.01 are unchanged.
The first eight captured calls (256 sampled tokens from two calibration prompts)
are fitting inputs; eight calls from two other calibration prompts are validation.
No held-out prompts are used. Inputs come from the BF16 model prefix.

[Config](../shiftadd/configs/upstream-bcq-full-query-p3-001.json),
[raw evidence](../results/runs/m5-flux-klein-shiftadd-014__attempt__20261103-q-p3/), and
[metric definitions](../wiki/methods/baseline-metrics.md).
This is CPU fitting and projection-error evaluation, with no image generation
or performance measurement. Upstream source hashes are recorded and checked.

## Results

| Saved file bytes | Payload bits/weight | Mean fit output L2 | Mean calibration-validation output L2 |
|---:|---:|---:|---:|
| 3,859,484 | 3.2500 | 0.00395669 | 0.02888011 |

The artifact contains 3,833,856 packed payload bytes versus
18,874,368 BF16 weight bytes. Including its header, the saved file is
79.55% smaller than BF16 payload.
All 16 per-call errors reproduce after independent packed-artifact reload;
the artifact hash matches. The converter also verifies packed decoding against
the upstream fitter's dense result.

## Delta versus four-plane reference

The four-plane file uses 5,112,860 bytes. Three planes save
1,253,376 bytes (24.51%), while mean
calibration-validation output error increases from 0.01713898 to
0.02888011. These are local matrix output errors,
not full-pipeline image quality measurements.

## Interpretation

The three-plane query artifact offers a smaller candidate for mixed-precision
selection at increased projection error. The four-prompt capture is too small
to establish general quality. Three sign planes plus scale metadata are not a
native GPU INT3 execution format. GPU resident memory, latency and energy
remain unmeasured for this artifact.

## Next experiment
- Test the three-plane query replacement against the four-plane Q/K/V image reference.

## Quantized activation arithmetic verification

The complete three-plane query artifact passed 32 CPU arithmetic comparisons:
16 captured calls with both A4 and A8 inputs at group size 32. INT32 sign-plane
sums with FP32 scaling agree with multiplication by independently decoded
weights using the same quantized inputs. Maximum relative disagreement is
8.1910594e-07.
[Evidence](../results/runs/m5-flux-klein-shiftadd-014__attempt__20261103-q-p3/integer-arithmetic-verification.json).
This verifies the arithmetic mapping, not GPU speed or activation quality.
