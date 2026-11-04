---
type: experiment-record
id: m5-flux-klein-shiftadd-009
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-009__attempt__20261102-full-query]
updated: 2026-11-02
---

# m5-flux-klein-shiftadd-009: full query-projection upstream BCQ fit

## Setup

Week 5 precision calibration, RQ1/RQ3. Extend the pinned upstream fitter from
[the slice experiment](m5-flux-klein-shiftadd-008.md) to all 3072 output rows of
`transformer_blocks.0.attn.to_q.weight` (3072 input channels). Source revision,
upstream commit, q=4, group 128, superblock 256, u8 scale format, fitting iterations,
damping and calibration split are unchanged. Only output-row coverage expands.
This is one complete linear matrix, not the complete transformer.

Fit uses 256 token samples from the first eight calls/two calibration prompts;
validation uses the other two calibration prompts. All 3072 output channels are
evaluated using original captured inputs and FP32 matrix products. Saved weights
are packed sign planes with uint8 per-plane scale codes and shared FP16 scales.
No additional Hessian damping retry occurred. Controls reuse the same original
weight matrix, inputs and split with the previously saved radix/conventional fits.
Control artifact hashes are recorded. No image generation or GPU execution occurs.

## Results

Mean per-call relative output L2:

| Fit | Payload bits/weight | Fitting calls | Calibration validation |
|---|---:|---:|---:|
| radix_p4 | 4.2500 | 0.067655 | 0.066683 |
| uniform_int4 | 4.2500 | 0.048979 | 0.048448 |
| Upstream GPTQ/BCQ u8 scales | 4.3125 | 0.002265 | 0.017139 |

Upstream packed payload: **5,087,232 bytes**; saved NPZ:
**5,112,860 bytes**, including output-row indices and file headers.
Original matrix BF16 payload is 18,874,368 bytes. Relative weight L2 is
0.183414; lower weight-space error is not the fit objective.
All 16 saved-artifact output metrics reproduce after independent reload.
[Configuration, source provenance, per-call evidence, controls and verification](../results/runs/m5-flux-klein-shiftadd-009__attempt__20261102-full-query/).

## Loader verification

The dense BF16 loader was checked at 35 independently reconstructed scalar
positions across row chunks and scale-group boundaries. All values matched.
A partial-row artifact was rejected before parameter mutation.
[Verification](../results/runs/m5-flux-klein-shiftadd-009__attempt__20261102-full-query/loader-verification.json).
This verifies loading correctness, not GPU execution or inference performance.

## Interpretation and limits

The upstream fit reduces calibration-validation output error on this complete
query projection, at a slightly larger payload than the controls. It changes
both scale freedom and fitting method, so this is not an isolated Hessian ablation
or an equal-storage superiority claim. The fitting/validation gap remains large,
and only two prompts supply validation. Inputs come from the original BF16 model;
sequential calibration through an already-quantized prefix is not performed.
One projection cannot establish full-transformer image quality, kernel support,
latency, resident memory or energy. This u8-scale representation is not the
restricted radix family used by the exact integer-code bridge.

Next experiment: paired single-projection image check after calibration review.
