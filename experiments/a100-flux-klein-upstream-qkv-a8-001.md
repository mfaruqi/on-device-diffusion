---
type: experiment-record
id: a100-flux-klein-upstream-qkv-a8-001
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-qkv-a8-001__attempt__20261103-calibration]
updated: 2026-11-03
---

# a100-flux-klein-upstream-qkv-a8-001: grouped A8 with upstream first-block Q/K/V

Status: **complete**, job 12139754, 71 seconds allocation wall time (not inference latency).

## Setup

Week 5 precision variants, RQ1/RQ3. Add grouped A8 inputs (group size 32) to
the same upstream four-plane Q/K/V weights as the
[weight-only reference](a100-flux-klein-upstream-qkv-calibration-001.md).
The three selected matrices are `transformer_blocks.0.attn.to_q.weight`,
`to_k.weight` and `to_v.weight`. Only their inputs receive A8 rounding and
reconstruction. Remaining weights and activations stay BF16.

A100 SXM4 on gilbreth-n012; device slug follows the existing cluster registry.
Pinned distilled Klein revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`,
512×512, four steps, guidance 1, batch one. Four calibration prompts and seeds
match the reference: teapot (17), lighthouse (23), three lemons (31), robot
with ECO AI sign (47). Fitting used the first two prompts; the latter two are
calibration-validation prompts, not held-out evaluation.

[Executed config, environment, audits and raw measurements](../results/runs/a100-flux-klein-upstream-qkv-a8-001__attempt__20261103-calibration/).
[Metric definitions](../wiki/methods/baseline-metrics.md).
One generation per case/prompt; no repeated latency protocol. Packed weights
are decoded to BF16; activations use fake quantization with floating GEMMs.

## Results

Relative latent L2 against paired BF16 (zero for the BF16 case):

| Prompt | Upstream weights, BF16 inputs | Same weights, A8 inputs |
|---|---:|---:|
| cal01 | 0.0354433 | 0.0348475 |
| cal02 | 0.0898966 | 0.1038023 |
| cal03 | 0.1460473 | 0.1048216 |
| cal04 | 0.1423134 | 0.1772023 |

All 12 outputs are finite and have the configured image dimensions. Both BF16
and weight-only latents exactly match the earlier Q/K/V comparison. Each weight
audit lists precisely three matrix replacements. Activation audits verify four
calls per selected module per prompt, 48 total, all at eight bits and group size
32. Independent CPU norm checks use rtol 1e-5 and atol 1e-8 for FP32 reduction
variation. Original measurements are preserved.

## Observations

All 12 images were inspected. Teapots, watercolor lighthouses, bottle/lemons
and robots remain recognizable. All three robot signs read ECO AI. All three
lemon images contain two visible lemons rather than the requested three.
A8 alters reflections and small scene details without an obvious visual collapse.

## Interpretation

A8 moves two latents closer to BF16 and two farther away. This small diagnostic
does not establish consistent improvement or matched image quality. It supports
continuing evaluation of this activation setting for the selected first-block
subset. It does not test A8 across the entire transformer. Dense reconstruction
and fake quantization provide no evidence of latency, resident GPU memory or
energy savings.

## Next experiment
- Measure a compatible packed execution path separately from this fake-quantization diagnostic.

## Published evidence

[Verified W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-upstream-qkv-a8-001__attempt__20261103-calibration) contains all 12 images, metrics and the evidence artifact. [Receipt](../results/runs/a100-flux-klein-upstream-qkv-a8-001__attempt__20261103-calibration/wandb-verification.json).
