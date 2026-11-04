---
type: experiment-record
id: a100-flux-klein-upstream-fused-calibration-001
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-fused-calibration-001__attempt__20261102-calibration]
updated: 2026-11-02
---

# a100-flux-klein-upstream-fused-calibration-001: four-prompt fused-projection comparison

Status: **complete**, job 12135879, 86 seconds.

## Question
How do three compressed representations of one fused projection affect outputs
across four existing calibration prompts, against paired BF16 references?

## Setup
Week 5 precision variants, RQ1/RQ3. A100 SXM4 on gilbreth-n012; the device slug
follows the existing cluster registry. Distilled Klein revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`, 512×512, four steps, guidance 1,
batch one. All candidates change only
`single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight` (27648×3072).
Other transformer weights, text encoder, VAE and activations remain BF16.
Compressed weights are reconstructed to BF16 for execution.

cal01: red teapot, seed 17; cal02: watercolor lighthouse, seed 23;
cal03: three lemons beside a green bottle, seed 31; cal04: robot with ECO AI sign,
seed 47. The upstream fit used captured inputs from cal01/cal02. cal03/cal04
are separate calibration-validation prompts, not the reserved held-out set.
Upstream u8 BCQ uses 4.3125 payload bits/weight; radix and conventional INT4 use
4.25. The [fitting record](m5-flux-klein-shiftadd-010.md) specifies the artifact.

[Executed configuration, environment, audits and measurements](../results/runs/a100-flux-klein-upstream-fused-calibration-001__attempt__20261102-calibration/).
[Metric definitions](../wiki/methods/baseline-metrics.md).
One generation per case/prompt; no discarded warm-ups or repeated timing protocol.

## Results
Relative latent L2 against the paired BF16 reference (lower means closer, not
necessarily better prompt adherence). BF16 is zero for all four prompts.

| Prompt | Radix p4 | Conventional INT4 | Upstream u8 BCQ |
|---|---:|---:|---:|
| cal01 | 0.0852470 | 0.1078333 | 0.0784985 |
| cal02 | 0.2079249 | 0.2087722 | 0.1723511 |
| cal03 | 0.5088275 | 0.3479211 | 0.2440732 |
| cal04 | 0.2368041 | 0.1366692 | 0.1768416 |

All 16 outputs are finite with correctly sized images. Every BF16 latent exactly
matches the earlier calibration reference. Each candidate audit replaces exactly
one expected matrix. Independent local norm recomputation differs from recorded
FP32 norms by less than 5e-7 absolute; validation uses rtol 1e-5 and atol 1e-8
for CPU reduction differences. The raw recorded measurements remain unchanged.

## Observations
Visual inspection of all 16 images finds recognizable teapots, watercolor
lighthouses and robots; every robot sign reads ECO AI. On cal03, radix produces
three visible lemons, while BF16, conventional INT4 and upstream BCQ show two.
The radix image thus follows the requested count despite its larger latent deviation.

## Interpretation
Upstream BCQ has the smallest latent difference among compressed candidates on
cal01, cal02 and cal03; conventional INT4 is closest on cal04. Two of the three
upstream wins use prompts that supplied fitting inputs. The two separate
calibration-validation prompts split the latent ranking between upstream and INT4.
The lemon count demonstrates why latent closeness cannot replace prompt-quality
assessment. These four prompts do not establish general quality superiority.
No packed-runtime latency, resident GPU memory or energy saving is measured.

## Next experiment
- Expand calibration capture to additional attention projections before fitting a larger upstream subset.

## Published evidence

[Verified W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-upstream-fused-calibration-001__attempt__20261102-calibration) includes all 16 images, quality metrics
and the evidence artifact. [Receipt](../results/runs/a100-flux-klein-upstream-fused-calibration-001__attempt__20261102-calibration/wandb-verification.json).
