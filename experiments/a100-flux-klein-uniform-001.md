---
type: experiment-record
id: a100-flux-klein-uniform-001
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-uniform-001__attempt__20261102-control]
updated: 2026-11-02
---

# a100-flux-klein-uniform-001: same-source radix versus conventional INT4 image control

## Setup

Week 5 execution variants, RQ1/RQ3. A100 SXM4 on gilbreth-n012, job 12133926.
The device slug follows the existing cluster registry; actual GPU is recorded in
`environment.json`. Distilled Klein revision e7b7dc27f91deacad38e78976d1f2b499d76a294,
512×512, four steps, guidance 1, batch one, seed 0, same cat/sign calibration prompt.
Text encoder and VAE remain BF16. Compare BF16, radix p4 and conventional symmetric
INT4 from the same source. Both compressed candidates use group size 128 and
4.25-bit weight payload. Their file sizes differ slightly in headers.
Each replaces all 109 transformer matrices and executes after dense BF16 reconstruction.
The new compact-base radix serialization is not the bundle used in this run.

## Results

[Config, status, environment, metrics and replacement audits](../results/runs/a100-flux-klein-uniform-001__attempt__20261102-control/).
Single image per case, without a steady timing protocol: no speed or energy claim.

| Case | Relative latent L2 vs paired BF16 |
|---|---:|
| BF16 | 0.0000000 |
| Radix p4 | 0.7684926 |
| Conventional INT4 | 0.6253031 |

All latents are finite. All images decode at 512×512. The paired validator verifies
all 109 replacements per candidate, rechecks latent metrics, and confirms the BF16
latent exactly matches the earlier radix diagnostic. Minor CPU norm-reduction
rounding differences are within the validator's declared tolerance.

Visual inspection: all cases show a recognizable cat holding a sign. BF16 clearly
renders “hello world”; both compressed cases distort the first word's lettering.
These unblinded observations on one calibration prompt are not a formal quality score.
Images and latent arrays remain local in `shiftadd/runs/a100-uniform-control-001/`.

## Interpretation and limits

Conventional INT4 is closer to the BF16 latent on this prompt at effectively equal
weight storage. This does not establish a general ranking or prompt-quality advantage.
The result does not support claiming the current radix fit is superior to conventional
INT4. All cases execute floating GEMMs; the recorded generation-to-latent times
are order/cold-state-confounded diagnostic timings, not quantized inference speed.
Packed runtime memory, throughput and energy remain unmeasured.

Next experiment: expand the same-source comparison to the existing calibration set.

W&B publishing verified finished status, all three images, diagnostic metrics
and the evidence artifact: [run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-uniform-001__attempt__20261102-control). No model weights uploaded.
