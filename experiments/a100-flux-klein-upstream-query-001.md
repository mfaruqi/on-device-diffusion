---
type: experiment-record
id: a100-flux-klein-upstream-query-001
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-query-001__attempt__20261102-query]
updated: 2026-11-02
---

# a100-flux-klein-upstream-query-001: single-query upstream fit image comparison

## Setup

Week 5 precision execution variants, RQ1/RQ3. Job 12134140, gilbreth-n012,
A100 SXM4, 70-second batch elapsed. Device slug follows the existing cluster
registry; actual hardware is in the environment file. Distilled Klein revision
 e7b7dc27f91deacad38e78976d1f2b499d76a294, 512×512, four steps, guidance 1,
batch one, seed 0, cat/sign calibration prompt. The text encoder, VAE and all
transformer weights except `transformer_blocks.0.attn.to_q.weight` remain BF16.

Four paired cases: unchanged BF16, radix p4, symmetric conventional INT4, and
upstream GPTQ/BCQ with u8/shared-FP16 scales. Each candidate replaces the same
3072×3072 matrix, then uses dense BF16 execution. Controls have 4.25 payload
bits/weight; the upstream fit has 4.3125. This is not equal-storage or an isolated
fitter ablation. The upstream fit used captured teapot/lighthouse activations;
the cat prompt was not in that fitting capture. It is still a repeatedly used
calibration prompt, not a held-out evaluation example.

[Configuration, environment, metrics, replacement audits and validation](../results/runs/a100-flux-klein-upstream-query-001__attempt__20261102-query/).
Single generation per case; first/warmup/repeated timing protocol is absent.
[Metric definitions](../wiki/methods/baseline-metrics.md).

## Results

| Case | Relative latent L2 vs paired BF16 |
|---|---:|
| bf16 | 0.0000000 |
| radix-p4-query | 0.1014581 |
| uniform-int4-query | 0.0924427 |
| upstream-u8-query | 0.0661509 |

All latents are finite. The paired BF16 latent exactly matches the earlier
reference, and all candidate audits contain exactly the selected projection.
Recomputed latent metrics agree within the validator's tolerance. All four
images decode at 512×512 and visually retain the cat, sign and readable
“hello world.” Composition differences are small in unblinded inspection;
no perceptual or prompt-quality ranking is established.
Images and latent arrays remain in `shiftadd/runs/a100-upstream-query-001/`.

## Interpretation and limits

The upstream candidate has lower final latent distance on this prompt than the
other single-projection candidates. This does not establish full-transformer
quality or matched-quality superiority. Scaling freedoms, payload size and
calibration-based fitting all differ. One modified projection cannot expose
error accumulation across quantized layers. The captured fit did not calibrate
through a quantized prefix. All GEMMs remain floating, so generation times are
diagnostic only; no packed runtime memory, speed or energy result is established.

Next experiment: separately labelled fused-projection image comparison.

W&B upload verified finished status, all four images, diagnostic metrics and the
evidence artifact: [run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-upstream-query-001__attempt__20261102-query). No model weights uploaded.
