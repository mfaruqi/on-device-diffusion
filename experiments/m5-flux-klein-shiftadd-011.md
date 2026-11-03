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
