---
type: experiment-record
id: a100-flux-klein-upstream-all-dual-a8-001
status: complete
device: a100-sxm4-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-all-dual-a8-001__attempt__20261103-calibration]
updated: 2026-11-03
---

# a100-flux-klein-upstream-all-dual-a8-001: grouped A8 with upstream all dual-stream Q/K/V

Status: **complete**, job 12140559, 66 seconds allocation wall time (not inference latency).

## Setup

Week 5 precision variants, RQ1/RQ3. Add grouped A8 inputs (group size 32) to
the same upstream four-plane Q/K/V weights as the
[weight-only reference](a100-flux-klein-upstream-all-dual-001.md).
The selected matrices are all 30 image/text Q/K/V projections across the five
dual-stream blocks. Only their inputs receive A8 rounding and reconstruction.
The 20 single-stream fused projections and all other weights remain BF16.

A100 SXM4 on gilbreth-n009. Placement was broadened from n012 to matching N-feature
SXM4 nodes while pending; the update receipt is preserved with the run.
Pinned distilled Klein revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`,
512×512, four steps, guidance 1, batch one. Four calibration prompts and seeds
match the reference: teapot (17), lighthouse (23), three lemons (31), robot
with ECO AI sign (47). Fitting used the first two prompts; the latter two are
calibration-validation prompts, not held-out evaluation.

[Executed config, environment, audits and raw measurements](../results/runs/a100-flux-klein-upstream-all-dual-a8-001__attempt__20261103-calibration/).
[Metric definitions](../wiki/methods/baseline-metrics.md).
One generation per case/prompt; no repeated latency protocol. Packed weights
are decoded to BF16; activations use fake quantization with floating GEMMs.

## Results

Relative latent L2 against paired BF16 (zero for the BF16 case):

| Prompt | Upstream weights, BF16 inputs | Same weights, A8 inputs |
|---|---:|---:|
| cal01 | 0.1238229 | 0.1286395 |
| cal02 | 0.1827682 | 0.1854916 |
| cal03 | 0.3757953 | 0.5501183 |
| cal04 | 0.2393647 | 0.2157204 |

All 12 outputs are finite and have the configured image dimensions. Both BF16
and weight-only latents exactly match the earlier 30-matrix comparison. Each weight
audit lists precisely 30 matrix replacements. Activation audits verify four
calls per selected module per prompt, 480 total, all at eight bits and group size
32. Independent CPU norm checks use rtol 1e-5 and atol 1e-8 for FP32 reduction
variation. Original measurements are preserved.

## Observations

The four new A8 images were inspected; the eight BF16 and weight-only PNGs
are byte-identical to previously inspected references. All scenes remain
recognizable and the A8 robot sign reads ECO AI. The A8 lemon image has three
visible lemons, matching the requested count; both reference cases have two.
A8 also changes reflections and small scene details.

## Interpretation

A8 moves one latent closer to BF16 and three farther away. The lemon image
has the largest increase in BF16 distance but satisfies the requested count,
illustrating that reference distance is not a prompt-quality metric. Four
prompts do not establish consistent improvement or formal matched quality. It does not test A8 across the entire transformer. Dense reconstruction
and fake quantization provide no evidence of latency, resident GPU memory or
energy savings.

## Next experiment
- Measure a compatible packed execution path separately from this fake-quantization diagnostic.

## Published evidence

[Verified W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-upstream-all-dual-a8-001__attempt__20261103-calibration) contains all 12 images, metrics and the evidence artifact. [Receipt](../results/runs/a100-flux-klein-upstream-all-dual-a8-001__attempt__20261103-calibration/wandb-verification.json).
