---
type: experiment-record
id: a100-flux-klein-upstream-lowplane-fused-001
status: complete
device: a100-sxm4-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-lowplane-fused-001__attempt__20261103-calibration]
updated: 2026-11-04
---

# a100-flux-klein-upstream-lowplane-fused-001: fixed-selection one-to-four-plane image diagnostic

## Setup

Week 5 precision variants, RQ1/RQ3. Extend the
[mixed-fused image reference](a100-flux-klein-upstream-mixed-fused-001.md)
with two-plane and one-plane fits for exactly the same ten fused matrices.
All other forty converted matrices stay at four planes; other transformer
weights and all activations stay BF16. No reselection or new prompts.

[Config](../shiftadd/configs/a100-upstream-lowplane-fused-001.json),
[Slurm launcher](../shiftadd/configs/a100-upstream-lowplane-fused-001.slurm),
[raw evidence](../results/runs/a100-flux-klein-upstream-lowplane-fused-001__attempt__20261103-calibration/),
[metrics](../wiki/methods/baseline-metrics.md).

Five cases: BF16, all-four-plane, mixed three/four, mixed two/four, mixed one/four.
Each uses the same four calibration prompts and seeds, 512×512, four steps,
guidance 1, batch one and pinned revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
One image per prompt/case; no repeat timing or formal held-out quality evaluation.
Weights reconstruct into dense BF16. This cannot establish packed inference
latency, runtime memory or energy gains.

## Results

Completed job **12163596**, exit 0:0, on gilbreth-n003 (A100 SXM4).
Allocation wall time was 578 seconds, not inference latency. Runtime Torch
2.5.1+cu121. All twenty images are 512×512 and all latents are finite.
Twelve BF16/four-plane/three-plane control images and latents exactly reproduce
the prior mixed-fused run. Each candidate replaces exactly the requested fifty
matrices. Independent CPU norms match at rtol 1e-5, atol 1e-8.
[Validation](../results/runs/a100-flux-klein-upstream-lowplane-fused-001__attempt__20261103-calibration/validation.json).

Relative latent L2 versus paired BF16 (lower means closer, not prompt accuracy):

| Prompt | Four planes | Mixed three/four | Mixed two/four | Mixed one/four |
|---|---:|---:|---:|---:|
| Teapot | 0.1956372 | 0.2772961 | 0.4529391 | 0.7189091 |
| Lighthouse | 0.3091596 | 0.3887144 | 0.5432779 | 0.8307384 |
| Lemons | 0.3365283 | 0.4750833 | 0.6945075 | 0.9773436 |
| Robot | 0.4791622 | 0.5953849 | 0.8158534 | 1.1030682 |

## Observations

Four- and three-plane cases retain the scenes with detail changes. Two-plane
substitution also retains all four scenes and readable ECO AI text, with larger
changes to the teapot, lighthouse, bottle and robot geometry. These cases all
show two visible lemons despite a three-lemon prompt.

One-plane substitution severely changes the outputs: the teapot loses its normal
handle/spout structure, the lighthouse composition changes substantially, the
lemon scene becomes a blurry yellow-green texture, and the robot becomes a
malformed object without a readable sign or clear face. This is visual inspection
of calibration images, not a formal blinded quality score.

## Interpretation

BF16 latent distance increases monotonically from four to one plane on each
prompt. Two-plane substitution retains recognizable prompt content on this small
set, but does not establish quality acceptance. The one-plane candidate shows
clear semantic failures in this fixed ten-matrix allocation. These findings do
not establish behavior for an entirely one-/two-plane transformer or for other
layer selections. All cases execute reconstructed BF16 weights; no packed GPU
latency, runtime-memory or energy benefit is measured.

## Next experiment
- Test a smaller two-plane allocation with predefined calibration quality criteria.

## Published evidence

[Verified W&B run](https://forge.coreweave.com/wandb/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-upstream-lowplane-fused-001__attempt__20261103-calibration) contains all twenty images, metrics and the evidence
artifact. [Receipt](../results/runs/a100-flux-klein-upstream-lowplane-fused-001__attempt__20261103-calibration/wandb-verification.json).
