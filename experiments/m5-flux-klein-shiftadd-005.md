---
type: experiment-record
id: m5-flux-klein-shiftadd-005
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-005__attempt__20261102-int4, m5-flux-klein-shiftadd-005__attempt__20261102-int8]
updated: 2026-11-02
---

# m5-flux-klein-shiftadd-005: same-source conventional quantization control

## Question and setup

Week 5 execution variants, RQ1/RQ3. Compare conventional symmetric INT4/INT8
with radix p4 on the same pinned source weights, 32 spread output rows, and
saved calibration inputs. Only the weight format changes. All products are FP32
on the local CPU. Group size is 128. Conventional scales are FP32, selected by
weight-only squared error from five clipping fractions: 1, .95, .9, .85, .8.
Signed integer ranges are −7…7 and −127…127; no learned zero point is used.
Radix also uses a bounded five-exponent weight-only search, but the candidate
sets differ, so this is a comparison of these specified fits rather than an
optimal quantizer comparison. No validation activations select the scales.

INT4 stores actual signed nibbles plus FP32 scales: 4.25 bits per weight,
exactly matching the radix p4 payload with four int8 exponent arrays.
INT8 costs 8.25 bits per weight. File headers are reported separately.
Stored files are reloaded for reconstruction and output evaluation.

The same three activation settings are evaluated for each weight format:
original BF16-origin inputs, per-token A8, or grouped A4 (32 channels).
The last two calibration prompts/eight calls provide validation; project
held-out prompts remain untouched.

## Results

Mean per-call relative output L2 on calibration validation. Reference radix rows
come from the [rank-zero control](../results/runs/m5-flux-klein-shiftadd-004__attempt__20261102-r0/aggregate-validation.json),
which exactly reproduced the earlier grouped-activation diagnostic.

| Weight format | Module | Weight payload bytes | Original inputs | A8 per token | A4 group 32 |
|---|---|---:|---:|---:|---:|
| INT4 | transformer_blocks | 52,224 | 0.070878 | 0.071506 | 0.085021 |
| INT4 | single_transformer_blocks | 52,224 | 0.081584 | 0.082814 | 0.134201 |
| INT8 | transformer_blocks | 101,376 | 0.001977 | 0.007426 | 0.044863 |
| INT8 | single_transformer_blocks | 101,376 | 0.005567 | 0.018125 | 0.105657 |
| Radix p4 | transformer_blocks | 52,224 | 0.081639 | 0.082107 | 0.095928 |
| Radix p4 | single_transformer_blocks | 52,224 | 0.079554 | 0.081215 | 0.133765 |

Raw conventional evidence:
- [m5-flux-klein-shiftadd-005__attempt__20261102-int4](../results/runs/m5-flux-klein-shiftadd-005__attempt__20261102-int4/)
- [m5-flux-klein-shiftadd-005__attempt__20261102-int8](../results/runs/m5-flux-klein-shiftadd-005__attempt__20261102-int8/)

## Interpretation and limits

At equal payload, conventional INT4 has lower mean output error on the query
slice; radix p4 is slightly better on the fused slice. The current radix fit has
no demonstrated general advantage over this conventional control. INT8 reduces
weight-induced error at a larger payload, while A4 activation error remains
substantial. These are two small slices and exploratory calibration results,
not full-model quality or matched hardware throughput. Symmetric INT4 here uses
15 values including zero; radix uses 16 odd signed levels without zero except
for zero-scale groups. Kernel support is not tested by this CPU analysis.

## Next experiment

- Extend the same-source control to full-transformer image calibration before any superiority claim.
