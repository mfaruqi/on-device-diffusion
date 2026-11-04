---
type: experiment-record
id: m5-flux-klein-shiftadd-010
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-010__attempt__20261102-full-fused]
updated: 2026-11-02
---

# m5-flux-klein-shiftadd-010: complete fused-projection upstream BCQ fit

## Setup

Week 5 precision calibration, RQ1/RQ3. Relative to the u8 fused-projection
[slice reference](m5-flux-klein-shiftadd-008.md), only output-row coverage expands
from 32 sampled rows to all 27,648 rows. The matrix is
`single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight`, with 3,072
input channels. Source revision, pinned upstream fitter, group 128, four planes,
u8 scales, superblock 256, 15 fitting iterations and 0.01 Hessian damping are
unchanged. There was no additional damping retry.

The first eight captured calls provide 256 fitting tokens from two calibration
prompts; the remaining eight calls use two other calibration prompts. These are
not held-out evaluation prompts. Inputs come from the original BF16 prefix.
This is a CPU numerical conversion, with no inference timing or energy measurement.
[Config, environment, hashes and raw evidence](../results/runs/m5-flux-klein-shiftadd-010__attempt__20261102-full-fused/).
Metrics follow [baseline definitions](../wiki/methods/baseline-metrics.md).

## Results

Mean per-call relative output L2 against original BF16 weights on the same inputs:

| Fit | Payload bits/weight | Fitting calls | Calibration validation |
|---|---:|---:|---:|
| Radix p4 | 4.2500 | 0.085917 | 0.087064 |
| Conventional INT4 | 4.2500 | 0.079232 | 0.080402 |
| Upstream GPTQ/BCQ u8 | 4.3125 | 0.005218 | 0.058251 |

Saved packed payload: **45,785,088 bytes**; NPZ file:
**46,007,324 bytes**, including output-row indices and headers.
The original matrix BF16 payload is 169,869,312 bytes.
Relative weight L2 is 0.138893. All 16 output metrics
reproduced after independently reloading the packed artifact. Source/control
hashes are recorded in the run evidence. Original weight files are preserved.

## Interpretation and limits

The fitted complete projection has lower calibration-validation output error
than these two controls, at a slightly larger payload. Scale freedom and fitting
method both differ, so this is not an equal-storage comparison or an isolated
Hessian ablation. The gap between fitting and validation errors remains large;
two validation prompts cannot establish robustness. This result does not measure
image quality, full-transformer error accumulation, GPU memory, latency or energy.
The u8/FP16 scale family differs from the restricted pure-power-of-two radix family.

Next experiment: paired full-projection image diagnostic with the remaining weights BF16.
