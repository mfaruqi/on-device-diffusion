---
type: experiment-record
id: a100-flux-klein-upstream-mixed-fused-001
status: complete
device: a100-sxm4-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-mixed-fused-001__attempt__20261103-calibration]
updated: 2026-11-03
---

# a100-flux-klein-upstream-mixed-fused-001: mixed three/four-plane image diagnostic

Status: **complete**, job 12149223, gilbreth-n008, exit 0:0.
Allocation wall time was 355 seconds; this is not inference latency.

## Question

Does the [fit-selected mixed candidate](m5-flux-klein-shiftadd-022.md) preserve
calibration image behavior relative to the all-four-plane attention reference?
Week 5 precision variants, RQ1/RQ3. Only the plane allocation changes.

## Setup

[Configuration](../shiftadd/configs/a100-upstream-mixed-fused-001.json),
[Slurm launcher](../shiftadd/configs/a100-upstream-mixed-fused-001.slurm),
[raw evidence](../results/runs/a100-flux-klein-upstream-mixed-fused-001__attempt__20261103-calibration/),
[metric definitions](../wiki/methods/baseline-metrics.md).

Distilled Klein revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`,
512×512, four steps, guidance 1, batch one, BF16 activations.
Three cases: unchanged BF16, all 50 selected matrices at four planes, and the
saved candidate with ten fused matrices at three planes and forty matrices at four.
The other 59 matrix weights remain unchanged. The four prompts/seeds match the
[all-attention control](a100-flux-klein-upstream-all-attention-001.md): teapot 17,
lighthouse 23, lemons 31, robot 47. These are calibration prompts, not held-out
quality evaluation. One generation per case/prompt; no repeated timing protocol.

All selected weights reconstruct to dense BF16. This run cannot establish packed
GPU latency, resident-memory or energy benefits. The allocation is capped at ten
minutes and selects N-feature SXM4 nodes. All 100 selected source-file hashes
were verified on Gilbreth before submission.

## Results

Relative latent L2 against paired BF16 (lower means closer, not necessarily better prompt quality):

| Prompt | Four-plane reference | Mixed three/four planes |
|---|---:|---:|
| Teapot | 0.1956372 | 0.2772961 |
| Lighthouse | 0.3091596 | 0.3887144 |
| Lemons | 0.3365283 | 0.4750833 |
| Robot | 0.4791622 | 0.5953849 |

All 12 latents are finite and all images are 512×512. Independent CPU norms agree
with the recorded metrics within rtol 1e-5 and atol 1e-8. All eight reference
latents and PNG files exactly reproduce the earlier BF16/four-plane controls.
Each candidate replaces precisely the requested 50 matrices. See
[validation](../results/runs/a100-flux-klein-upstream-mixed-fused-001__attempt__20261103-calibration/validation.json) and
[summary](../results/runs/a100-flux-klein-upstream-mixed-fused-001__attempt__20261103-calibration/summary.json).

## Observations

All scenes remain recognizable. The mixed candidate changes teapot highlights
and geometry, lighthouse/rock details, bottle and lemon shapes, and robot
proportions. All three cases show two visible lemons although the prompt asks
for three; all robot signs read ECO AI. This is visual inspection of four
calibration prompts, not a blinded or held-out quality assessment.

## Interpretation

The mixed candidate has greater BF16 latent distance on all four prompts.
The local fitting-error selection therefore does not preserve the four-plane
reference's numerical fidelity at this allocation. The candidate remains
visually coherent; no formal quality acceptance or rejection is established.
Storage savings are documented in the source candidate record; this reconstructed
BF16 experiment establishes no packed GPU speed, memory or energy advantage.

## Next experiment
- Evaluate fewer three-plane substitutions with a declared calibration quality criterion before choosing a deployment candidate.

## Published evidence

[Verified W&B run](https://forge.coreweave.com/wandb/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-upstream-mixed-fused-001__attempt__20261103-calibration) contains all 12 images, diagnostic metrics and
the evidence artifact. [Verification receipt](../results/runs/a100-flux-klein-upstream-mixed-fused-001__attempt__20261103-calibration/wandb-verification.json).
