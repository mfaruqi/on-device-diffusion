---
type: experiment-record
id: a100-flux-klein-upstream-qkv-calibration-001
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-qkv-calibration-001__attempt__20261103-calibration]
updated: 2026-11-03
---

# a100-flux-klein-upstream-qkv-calibration-001: first-block Q/K/V image comparison

Status: **complete**, job 12138121, 75 seconds (allocation wall time, not inference latency).

## Question
How do three compressed representations of first-block Q/K/V affect generation
across four calibration prompts against paired BF16 references?

## Setup
Week 5 precision variants, RQ1/RQ3. A100 SXM4 on gilbreth-n012; the device slug
follows the existing cluster registry. Distilled Klein revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`, 512×512, four steps, guidance 1,
batch one. All candidates change exactly three matrices:
`transformer_blocks.0.attn.to_q.weight`, `to_k.weight` and `to_v.weight`.
Other weights and activations remain BF16. Packed weights are reconstructed
to BF16 before execution; there is no integer GPU kernel in this diagnostic.

cal01: red teapot, seed 17; cal02: watercolor lighthouse, seed 23;
cal03: three lemons beside a green bottle, seed 31; cal04: robot with ECO AI sign,
seed 47. Upstream fitting used cal01/cal02 inputs; cal03/cal04 are separate
calibration-validation prompts, not formal held-out evaluation.
Upstream BCQ u8 uses 4.3125 payload bits/weight; radix and uniform INT4 use 4.25.
The upstream bundle combines [Q](m5-flux-klein-shiftadd-009.md) and
[K/V](m5-flux-klein-shiftadd-011.md) artifacts without refitting.

[Executed configuration, environment, audits and measurements](../results/runs/a100-flux-klein-upstream-qkv-calibration-001__attempt__20261103-calibration/).
[Metric definitions](../wiki/methods/baseline-metrics.md).
One generation per case/prompt, without a repeated latency benchmark protocol.

## Results
Relative latent L2 against BF16 (lower means closer, not necessarily better prompt
adherence). BF16 is zero on all prompts.

| Prompt | Radix p4 | Uniform INT4 | Upstream BCQ u8 |
|---|---:|---:|---:|
| cal01 | 0.1131187 | 0.1349570 | 0.0354433 |
| cal02 | 0.3065345 | 0.2702241 | 0.0898966 |
| cal03 | 0.2805542 | 0.3214133 | 0.1460473 |
| cal04 | 0.4355848 | 0.4892730 | 0.1423134 |

All 16 outputs are finite and images have the configured dimensions. All four
BF16 latents exactly match the earlier calibration references. Each candidate
audit lists exactly the three configured replacements. Independent FP32 norm
recomputation differs by at most 7.1525574e-07 absolute; rtol 1e-5 and atol 1e-8
allow for CPU reduction differences. Original measurements are preserved.

## Observations
Visual inspection of all 16 images finds recognizable red teapots, watercolor
lighthouses, lemon/bottle photographs and robots. Every robot sign reads ECO AI.
All four lemon images show two visible lemons, despite the requested three.
Uniform INT4 adds a small mouth to the robot face; other versions lack it.
Details of highlights, rocks and reflections vary across candidates.

## Interpretation
Upstream BCQ is closer to BF16 on all four prompts, including both separate
calibration-validation prompts. This is evidence for this first-block subset,
not full-transformer quality superiority. Payload sizes differ, and fitting
method and scale freedom change together. The shared lemon-count failure shows
that closeness to BF16 does not establish prompt adherence. No latency, resident
GPU memory or energy savings are established by dense BF16 reconstruction.

## Next experiment
- Add grouped A8 inputs to the same upstream Q/K/V subset with matched prompts and seeds.

## Published evidence

[Verified W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-upstream-qkv-calibration-001__attempt__20261103-calibration) includes all 16 images, quality metrics
and the evidence artifact. [Receipt](../results/runs/a100-flux-klein-upstream-qkv-calibration-001__attempt__20261103-calibration/wandb-verification.json).
