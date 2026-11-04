---
type: experiment-record
id: m5-flux-klein-shiftadd-003
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-003__attempt__20261102-g3072, m5-flux-klein-shiftadd-003__attempt__20261102-g128, m5-flux-klein-shiftadd-003__attempt__20261102-g32]
updated: 2026-11-02
---

# m5-flux-klein-shiftadd-003: grouped A4 activation diagnostic

## Question and setup

Week 5 execution variants, RQ1/RQ3. Change only activation quantization group
size from 3072 (one scale per token) to 128 or 32 channels. Each labelled config
retains the same four-plane weights, two fitting methods, 32 spread output rows,
and saved inputs as the [prior slice diagnostic](m5-flux-klein-shiftadd-002.md).
Pinned source revision and exact row indices are in each run's config and summary.
All multiplication uses FP32 on the local CPU. Eight calls from the last two
calibration prompts form validation; the project's held-out prompts are untouched.
No inference timings, energy or generated images are measured.

## Results

Mean per-call relative output L2 on calibration validation; original BF16-origin
inputs multiplied by original weights provide the FP32 reference.

| Channels per activation scale | Module | Weight fit | Mean relative output L2 |
|---|---|---|---:|
| 3072 | single_transformer_blocks | weight_only | 0.318894 |
| 3072 | single_transformer_blocks | diagonal_activation | 0.320034 |
| 3072 | transformer_blocks | weight_only | 0.242219 |
| 3072 | transformer_blocks | diagonal_activation | 0.222394 |
| 128 | single_transformer_blocks | weight_only | 0.175046 |
| 128 | single_transformer_blocks | diagonal_activation | 0.177040 |
| 128 | transformer_blocks | weight_only | 0.124493 |
| 128 | transformer_blocks | diagonal_activation | 0.100991 |
| 32 | single_transformer_blocks | weight_only | 0.133765 |
| 32 | single_transformer_blocks | diagonal_activation | 0.133695 |
| 32 | transformer_blocks | weight_only | 0.095928 |
| 32 | transformer_blocks | diagonal_activation | 0.067776 |

Raw evidence:
- [m5-flux-klein-shiftadd-003__attempt__20261102-g3072](../results/runs/m5-flux-klein-shiftadd-003__attempt__20261102-g3072/)
- [m5-flux-klein-shiftadd-003__attempt__20261102-g128](../results/runs/m5-flux-klein-shiftadd-003__attempt__20261102-g128/)
- [m5-flux-klein-shiftadd-003__attempt__20261102-g32](../results/runs/m5-flux-klein-shiftadd-003__attempt__20261102-g32/)

## Interpretation and limits

Smaller activation groups lower mean output error in both sampled modules for
both fitting methods. These slices do not establish image quality or performance.
Ideal packed A4 plus one FP32 scale per group costs 4 + 32/G bits per activation:
4.0104 at G=3072, 4.25 at G=128 and 5 at G=32. This excludes alignment and
workspace; this implementation stores int8 codes and performs simulated quantization,
so those ideal sizes are estimates, not measured resident memory. Groupwise scales
also add execution work; lower numerical error alone does not establish a faster format.

## Next experiment

- Evaluate grouped activation hooks on a separately labelled full-pipeline calibration run.

## Packed arithmetic verification

A separate correctness check uses integer activations directly with unpacked
sign planes, int32 partial sums, exponent factors and per-group activation scales.
It does not reconstruct dense weights in the tested path. The independent dense
reference reconstructs weights and dequantizes activations before FP32 matmul.
Both A4 and A8, both weight fits, both modules, and all 16 captured calls are
checked at each group size. Evidence is saved under each run's
`arithmetic-verification/` directory with source artifact hashes.

| Activation group size | Comparisons | Maximum relative L2 vs dense reference |
|---|---:|---:|
| 3072 | 128 | 9.73013995e-07 |
| 128 | 128 | 9.24307821e-07 |
| 32 | 128 | 1.0750123e-06 |

All pass the declared 1e-5 relative-L2 threshold. Differences are FP32 accumulation
order effects. This proves agreement for the tested slices, not throughput,
energy, full-layer accuracy or integer tensor-core compatibility.

## Actual packed activation payload check

A4 codes now round-trip through two signed two's-complement nibbles per byte
(first value in the low nibble). A8 codes retain one byte per value. All 384
real-input arithmetic comparisons pass after code packing and unpacking.
The checks count actual NumPy payload and FP32 scale array bytes per 32×3072
activation sample, compared with 196,608 bytes for the same BF16 sample.

| Activation bits | Group size | Code bytes | Scale bytes | Total bytes | Reduction vs BF16 |
|---|---|---:|---:|---:|---:|
| 4 | 3072 | 49,152 | 128 | 49,280 | 74.93% |
| 8 | 3072 | 98,304 | 128 | 98,432 | 49.93% |
| 4 | 128 | 49,152 | 3,072 | 52,224 | 73.44% |
| 8 | 128 | 98,304 | 3,072 | 101,376 | 48.44% |
| 4 | 32 | 49,152 | 12,288 | 61,440 | 68.75% |
| 8 | 32 | 98,304 | 12,288 | 110,592 | 43.75% |

Evidence is in each run's `packed-activation-verification/` directory. These are
array payload sizes, excluding headers, shape metadata, alignment, workspaces
and transient unpacked arrays. The CPU reference still unpacks codes before
multiplication; these counts do not establish compressed GPU execution or peak
memory savings. Packing adds no numerical error beyond the quantization already
measured. No wall-clock or energy claim is made.

## Exact integer-format bridge verification

The saved four-plane radix weights also pass exact reconstruction through
signed INT8 codes, or signed INT4 codes with the affine relation 2q+1 and a
per-group input-sum correction. `integer-radix-bridge-001.json` under the g32
run preserves all 768 arithmetic comparisons (both fit methods/modules, all
calls, A4/A8, three activation group sizes and both weight-code targets).
Maximum relative difference from dense reconstruction is about 1.01e-6.
This is CPU numerical verification, not a GPU kernel or performance result.
The mapping requires the verified radix exponent relationship and does not
establish an exact mapping for arbitrary ShiftAdd plane exponents.
