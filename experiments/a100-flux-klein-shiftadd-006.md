---
type: experiment-record
id: a100-flux-klein-shiftadd-006
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-shiftadd-006__attempt__20261102-full-activation]
updated: 2026-11-02
---

# a100-flux-klein-shiftadd-006: grouped activation quantization across the transformer

## Setup

Week 5 execution variants, RQ1/RQ3. Job 12133980 completed on gilbreth-n012
(A100 SXM4; the device slug follows the existing cluster registry). Distilled
Klein revision e7b7dc27f91deacad38e78976d1f2b499d76a294, 512×512, four steps,
guidance 1, batch one, seed 0, cat/sign calibration prompt. Text encoder and VAE
remain BF16. The reference candidate reconstructs the saved group-128 radix p4
weights into dense BF16 for all 109 transformer matrices. Two separately labelled
variants add A8 or A4 input quantization, group 32, to all 109 linear modules.
Activation hooks round/dequantize to the original floating dtype; GEMMs remain BF16.
The BF16 checkpoint is also run as a paired reference.

[Configuration, environment, metrics, audits and validation](../results/runs/a100-flux-klein-shiftadd-006__attempt__20261102-full-activation/).
A single generation per case is a quality diagnostic, without the repeated timing
protocol. Recorded generation-to-latent times do not establish inference speed.
Metric definitions: [baseline metrics](../wiki/methods/baseline-metrics.md).

## Results

| Case | Relative latent L2 vs paired BF16 |
|---|---:|
| bf16 | 0.0000000 |
| radix-p4 | 0.7684926 |
| radix-p4-all-a8-g32 | 0.7619766 |
| radix-p4-all-a4-g32 | 0.7915170 |

All latents are finite and images decode at 512×512. Each candidate replaces
exactly 109 weight matrices. Each activation variant records exactly four hook
calls for every selected module: 436 calls total. The BF16 latent exactly matches
the earlier paired reference. Recorded latent metrics reproduce within the
validator's tolerance; the validation is not a quality-acceptance gate.

Visual inspection: all four images show a recognizable cat holding a sign.
BF16 clearly renders “hello world.” The weight-only and A8 images have very similar
composition and distort the first word. A4 changes the cat/sign proportions more
and renders the requested phrase clearly in this image. This unblinded observation
is not a formal score. Lower distance to BF16 need not mean better text fidelity.
Images and latent arrays remain in `shiftadd/runs/a100-full-activation-001/`.

## Interpretation and limits

Grouped A8 and A4 inputs do not collapse the image on this calibration prompt.
This supports further calibration, not full quality admission. Only linear inputs
are quantized: residual states, attention operations and other activations retain
the pipeline's floating representation. Dense weights and fake quantization do
not demonstrate packed resident-memory savings, low-bit execution speed or energy.
The compact radix serialization was not used in this run.

Next experiment: staged upstream-query image diagnostic with the same paired BF16 reference.

W&B upload verified finished status, all four images, diagnostic metrics and the
evidence artifact: [run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-shiftadd-006__attempt__20261102-full-activation). No model weights uploaded.
