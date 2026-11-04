---
type: experiment-record
id: a100-flux-klein-upstream-all-dual-q3-001
status: complete
device: a100-sxm4-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-all-dual-q3-001__attempt__20261103-calibration]
updated: 2026-11-03
---

# a100-flux-klein-upstream-all-dual-q3-001: three-plane image queries across all dual-stream blocks

Status: **complete**, job 12140665, 49 seconds allocation wall time, not inference latency.

## Setup

Week 5 precision variants, RQ1/RQ3. Starting from the 30-matrix upstream
four-plane image/text Q/K/V reference, change only the five image-query
matrices to three-plane upstream fits. The other 25 selected matrices retain
four planes; all other weights and activations remain BF16. The
[five Q3 fits](m5-flux-klein-shiftadd-018.md) use the same capture and fitting
settings apart from plane count. The
[weight-only image reference](a100-flux-klein-upstream-all-dual-001.md) is
repeated within this run.

The selected mixed bundle occupies 147,118,920 bytes versus 153,385,800 for
all-four-plane files, a 4.09% reduction for this 30-matrix subset. These are
packed file sizes, not runtime memory. The image runner reconstructs dense BF16.

A100-SXM4-40GB, gilbreth-n009, Torch 2.5.1+cu121; pinned distilled Klein revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`, 512×512, four steps, guidance 1,
batch one. Placement was broadened from n012 to matching N-feature SXM4 nodes
while pending; the update receipt is retained. The same four calibration
prompts/seeds are used: teapot 17, lighthouse 23, three lemons 31, ECO AI robot
47. The first two supplied fitting inputs; the latter two are calibration
validation. No held-out prompts were used.

[Executed configuration, environment, audits and raw measurements](../results/runs/a100-flux-klein-upstream-all-dual-q3-001__attempt__20261103-calibration/).
[Metric definitions](../wiki/methods/baseline-metrics.md).
One image per case/prompt; no repeated timing protocol.

## Results

Relative latent L2 against paired BF16 (zero for the BF16 case):

| Prompt | All four-plane upstream | Five image queries at three planes |
|---|---:|---:|
| cal01 | 0.1238229 | 0.0836489 |
| cal02 | 0.1827682 | 0.1851288 |
| cal03 | 0.3757953 | 0.4311583 |
| cal04 | 0.2393647 | 0.2496140 |

All 12 outputs are finite and have the configured dimensions. Both reference
cases reproduce the prior latents exactly and their eight PNGs are byte-identical.
Each candidate audit lists the intended 30 replacements. Independent CPU norms
agree within rtol 1e-5 and atol 1e-8 for FP32 reduction variation.

## Observations

The four new mixed-plane images were inspected alongside previously inspected,
byte-identical references. All scenes remain recognizable. The robot sign reads
ECO AI, and the lemon image still shows two visible lemons rather than three.
There are changes to highlights, small geometry and rock/water details.

## Interpretation

Reducing all five image-query matrices to three planes saves a modest amount
of subset storage. BF16 latent distance improves for the teapot but increases
for the other three prompts, including both calibration-validation prompts.
This does not establish an admitted mixed-precision configuration or an image
quality advantage. It also does not establish GPU memory, latency or energy
savings because execution uses reconstructed dense BF16 weights.

## Next experiment
- Evaluate selective query-plane allocation using calibration quality criteria.

## Published evidence

[Verified W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-upstream-all-dual-q3-001__attempt__20261103-calibration) contains all 12 images, metrics and the evidence artifact. [Receipt](../results/runs/a100-flux-klein-upstream-all-dual-q3-001__attempt__20261103-calibration/wandb-verification.json).
