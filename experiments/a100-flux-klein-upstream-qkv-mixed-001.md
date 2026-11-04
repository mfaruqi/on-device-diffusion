---
type: experiment-record
id: a100-flux-klein-upstream-qkv-mixed-001
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-qkv-mixed-001__attempt__20261103-calibration]
updated: 2026-11-03
---

# a100-flux-klein-upstream-qkv-mixed-001: first-block mixed-plane Q/K/V images

Status: **complete**, job 12139916, 57 seconds allocation wall time (not inference latency).

## Setup

Week 5 precision variants, RQ1/RQ3. Relative to the
[four-plane Q/K/V reference](a100-flux-klein-upstream-qkv-calibration-001.md),
reduce only K or only V to three planes in separate cases. Each case replaces
exactly `transformer_blocks.0.attn.to_q.weight`, `to_k.weight`, `to_v.weight`.
Other weights and all activations stay BF16. Bundles combine existing
[four-plane fits](m5-flux-klein-shiftadd-011.md) with the
[three-plane K/V fits](m5-flux-klein-shiftadd-013.md), without refitting.
Before submission, copied artifact hashes and independent BF16 decoding were
checked; only the intended K or V matrix differs from the four-plane bundle.

A100 SXM4 on gilbreth-n012, with the existing cluster registry device slug.
Distilled Klein revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`, 512×512,
four steps, guidance 1, batch one. Prompts and seeds match the reference:
teapot (17), lighthouse (23), three lemons (31), robot with ECO AI sign (47).
The first two prompts supplied fitting inputs; the latter two are separate
calibration-validation prompts, not held-out evaluation.

[Executed config, environment, bundle provenance, audits and measurements](../results/runs/a100-flux-klein-upstream-qkv-mixed-001__attempt__20261103-calibration/).
[Metric definitions](../wiki/methods/baseline-metrics.md).
One generation per case/prompt. Packed weights reconstruct to dense BF16;
there is no packed GPU execution or repeated latency benchmark.

## Results

Relative latent L2 against paired BF16 (zero for BF16):

| Prompt | Q4/K4/V4 | Q4/K3/V4 | Q4/K4/V3 |
|---|---:|---:|---:|
| cal01 | 0.0354433 | 0.0801918 | 0.0669129 |
| cal02 | 0.0898966 | 0.0921766 | 0.1750685 |
| cal03 | 0.1460473 | 0.1474769 | 0.2248293 |
| cal04 | 0.1423134 | 0.1139243 | 0.3000044 |

Each mixed bundle has 14,085,204 artifact file bytes versus 15,338,580 for the
four-plane bundle: 1,253,376 bytes saved, 8.17% of this three-matrix bundle.
These are packed file sizes, not resident GPU memory or whole-model savings.

All 16 outputs are finite with configured dimensions. BF16 and four-plane
reference latents exactly reproduce the earlier run, and their eight PNG files
are byte-identical. Candidate audits list exactly three replacements.
Independent CPU norms agree within rtol 1e-5 and atol 1e-8.

## Observations

The eight mixed images were inspected alongside the previously inspected,
byte-identical references. All remain recognizable. Robot signs read ECO AI;
all lemon images show two visible lemons, not the requested three. V3 changes
lighthouse rocks/waves and bottle reflections more visibly than K3.

## Interpretation

On the two calibration-validation prompts, K3 stays near the four-plane latent
error for lemons and is closer to BF16 for the robot. V3 increases the error
on both, despite equal storage savings. K3 is therefore the more promising
candidate within this small first-block diagnostic. Lower latent error does
not establish better prompt adherence, and no general mixed-precision policy
or matched-quality claim follows from four prompts. GPU latency, resident
memory and energy savings remain unmeasured.

## Next experiment
- Add a Q3/K4/V4 case with the same prompts and four-plane reference.

## Published evidence

[Verified W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-upstream-qkv-mixed-001__attempt__20261103-calibration) contains all 16 images, metrics and the evidence artifact. [Receipt](../results/runs/a100-flux-klein-upstream-qkv-mixed-001__attempt__20261103-calibration/wandb-verification.json).
