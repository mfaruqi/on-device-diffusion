---
type: experiment-record
id: a100-flux-klein-upstream-all-attention-activation-001
status: complete
device: a100-sxm4-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-all-attention-activation-001__attempt__20261103-calibration]
updated: 2026-11-03
---

# a100-flux-klein-upstream-all-attention-activation-001: grouped A8 and A4 across the upstream attention subset

Status: **complete**, job 12141633, 528 seconds allocation wall time, not inference latency.

## Setup

Week 5 precision variants, RQ1/RQ3. Keep the same upstream four-plane weights
for all 50 attention/fused matrices, and separately change their inputs to
symmetric grouped A8 or A4, group size 32. The scope is 30 image/text Q/K/V
matrices in five dual-stream blocks and 20 fused QKV/MLP projections in the
single-stream blocks. The other 59 matrix weights and their inputs remain
unmodified by these hooks. BF16 and the weight-only candidate are repeated
within the run; the latter matches the
[completed 50-matrix comparison](a100-flux-klein-upstream-all-attention-001.md).

A100-SXM4-40GB on gilbreth-n001, Torch 2.5.1+cu121. The scheduler uses feature N
to preserve the SXM4 hardware family. Distilled Klein revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`, 512×512, four steps, guidance 1,
batch one. Four calibration prompts/seeds: teapot 17, lighthouse 23, three
lemons 31, ECO AI robot 47. The first two supplied fitting inputs; the latter
two are calibration validation. No reserved held-out prompts were used.

[Executed configuration, environment, audits and raw measurements](../results/runs/a100-flux-klein-upstream-all-attention-activation-001__attempt__20261103-calibration/).
[Metric definitions](../wiki/methods/baseline-metrics.md).
One generation per case/prompt, without repeated timing measurements.
Packed weights reconstruct to dense BF16. Activation rounding/dequantization
uses floating GEMMs; this is not a packed integer W4A4 or W4A8 backend.

## Results

Relative latent L2 against paired BF16 (zero for the BF16 case):

| Prompt | Upstream weights, BF16 inputs | Same weights, A8 inputs | Same weights, A4 inputs |
|---|---:|---:|---:|
| cal01 | 0.1956372 | 0.1931256 | 0.2219598 |
| cal02 | 0.3091596 | 0.3434626 | 0.3801785 |
| cal03 | 0.3365283 | 0.3658480 | 0.4558232 |
| cal04 | 0.4791622 | 0.5082831 | 0.4742941 |

All 16 outputs are finite and have the configured dimensions. The BF16 and
weight-only latents exactly match the earlier run, and their eight PNGs are
byte-identical. Each candidate replaces exactly the intended 50 matrices.
For each activation mode, audits verify four calls per module per prompt:
800 calls for A8 and 800 for A4, all at group size 32. Independent CPU norms
match within rtol 1e-5 and atol 1e-8 to allow FP32 reduction variation.

## Observations

The eight new A8/A4 images were inspected alongside previously inspected,
byte-identical references. All scenes remain recognizable, both robot signs
read ECO AI, and both activation variants still show two visible lemons rather
than the requested three. A4 changes lighthouse/building details, teapot
reflections and robot proportions. No obvious scene collapse appears in this
four-prompt diagnostic.

## Interpretation

A8 has lower BF16 latent distance than A4 on the teapot, lighthouse and lemon
prompts; A4 has lower distance on the robot. Relative to the weight-only case,
A8 is closer only for the teapot, while A4 is closer only for the robot.
These prompt-specific rankings do not establish formal quality admission or
consistent superiority. The scope includes all Q/K/V projections but not every
transformer matrix. Floating reconstruction and fake quantization establish no
resident GPU memory, inference-latency or energy advantage.

## Next experiment
- Evaluate these candidates with declared calibration quality criteria and a compatible packed execution backend.

## Published evidence

[Verified W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-upstream-all-attention-activation-001__attempt__20261103-calibration) contains all 16 images, metrics and the evidence artifact. [Receipt](../results/runs/a100-flux-klein-upstream-all-attention-activation-001__attempt__20261103-calibration/wandb-verification.json).
