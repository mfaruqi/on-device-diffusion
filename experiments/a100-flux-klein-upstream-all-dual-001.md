---
type: experiment-record
id: a100-flux-klein-upstream-all-dual-001
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-all-dual-001__attempt__20261103-calibration]
updated: 2026-11-03
---

# a100-flux-klein-upstream-all-dual-001: all dual-stream Q/K/V image comparison

Status: **complete**, job 12140419, 88 seconds allocation wall time, not inference latency.

## Setup

Week 5 precision variants, RQ1/RQ3. Compare BF16 with radix p4, conventional
symmetric INT4 and upstream BCQ on the same 30 image/text Q/K/V matrices across
all five dual-stream blocks. Remaining transformer weights and all activations
stay BF16. The upstream artifacts come from the
[verified 30-matrix fit](m5-flux-klein-shiftadd-015.md).
Packed weights reconstruct to dense BF16 before execution.

A100 SXM4 on gilbreth-n012, Torch 2.5.1+cu121; device slug follows the existing
cluster registry. Distilled Klein revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`, 512×512, four steps, guidance 1,
batch one. Four calibration prompts/seeds: teapot (17), lighthouse (23), three
lemons (31), ECO AI robot (47). The first two supplied fitting inputs; the
latter two are calibration-validation prompts. No held-out prompts were used.

[Executed configuration, environment, audits and measurements](../results/runs/a100-flux-klein-upstream-all-dual-001__attempt__20261103-calibration/).
[Metric definitions](../wiki/methods/baseline-metrics.md).
One generation per case/prompt, without repeated timing measurements.
Upstream uses 4.3125 payload bits/weight; radix and uniform INT4 use 4.25.
The fitting objective and scale family also differ, so the comparison does not
isolate activation calibration from representation changes.

## Results

Relative latent L2 against paired BF16 (BF16 is zero for each prompt):

| Prompt | Radix p4 | Uniform INT4 | Upstream BCQ |
|---|---:|---:|---:|
| cal01 | 0.2483013 | 0.2454210 | 0.1238229 |
| cal02 | 0.3083410 | 0.2931840 | 0.1827682 |
| cal03 | 0.4491242 | 0.3995665 | 0.3757953 |
| cal04 | 0.3554004 | 0.2758551 | 0.2393647 |

All 16 outputs are finite and have the configured dimensions. Each candidate
audit lists exactly the same 30 requested replacements. All four BF16 latents
exactly reproduce the earlier calibration run and its PNGs are byte-identical.
Independent CPU norms pass rtol 1e-5, atol 1e-8, allowing FP32 reduction variation.

## Observations

The 12 candidate images were inspected alongside the previously inspected,
byte-identical BF16 references. Every case retains recognizable teapot,
lighthouse, lemon/bottle and robot scenes; all robot signs read ECO AI.
All lemon images show two visible lemons rather than the requested three.
Uniform INT4 changes the robot's face frame to a dark rim, and adds a prominent
foreground rock to the lighthouse image. Other small geometry/reflection changes
appear across candidates.

## Interpretation

Upstream BCQ has lower BF16 latent distance than both controls on all four
prompts, including the two calibration-validation prompts. This supports broader
upstream fitting within the tested dual-stream subset, but does not establish
formal matched quality, full-transformer superiority, or an advantage at equal
bit budgets. The 20 fused single-stream projections are unchanged in this run.
Dense reconstruction establishes no GPU resident-memory, latency or energy savings.

## Next experiment
- Add grouped A8 inputs to these 30 upstream projections against the same weight-only reference.

## Published evidence

[Verified W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-upstream-all-dual-001__attempt__20261103-calibration) contains all 16 images, metrics and the evidence artifact. [Receipt](../results/runs/a100-flux-klein-upstream-all-dual-001__attempt__20261103-calibration/wandb-verification.json).
