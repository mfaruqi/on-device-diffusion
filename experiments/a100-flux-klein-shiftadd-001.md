---
type: experiment-record
id: a100-flux-klein-shiftadd-001
status: complete
device: a100-sxm4-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-shiftadd-001__smoke__20261102]
updated: 2026-11-02
---

# a100-flux-klein-shiftadd-001: greedy four-plane reconstructed-weight smoke

## Question

Does the greedy four-plane compressed transformer preserve a recognizable
image on one calibration prompt? Week 5 precision variants, RQ1/RQ3.

## Setup

Slurm job 12133572, node gilbreth-n012. Distilled Klein 4B revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`, 512-square, four steps,
guidance 1, seed 0, batch one. Prompt: “A cat holding a sign that says hello world”.
[Config](../results/runs/a100-flux-klein-shiftadd-001__smoke__20261102/config.json).

Cases in one loaded pipeline: BF16; reconstruct one attention query matrix;
then reconstruct all 109 matrix weights from four sign planes with group-128
power-of-two scales. Weights are installed into dense BF16 parameters. No packed
GPU kernel runs. Text encoder and VAE are unchanged. This is a one-prompt
functional diagnostic, with no warmups or repeated timings.

## Results

[Summary](../results/runs/a100-flux-klein-shiftadd-001__smoke__20261102/summary.json):

| Case | Relative latent L2 vs BF16 | Finite |
|---|---:|---|
| BF16 | 0 | yes |
| One query projection | 0.288506 | yes |
| All matrix weights | 1.113309 | yes |

The decoded one-projection image retains a cat and readable “hello world” sign.
The all-matrix image is noise-like with no recognizable requested subject.
Images and latents are preserved locally in `shiftadd/runs/a100-greedy-paired-001/`
and remotely under `/scratch/gilbreth/mfaruqi/shiftadd-initial/runs/paired-001/`.

Generation-to-latent observations in the summary are cold/warm-confounded smoke
measurements, exclude decode and weight installation, and are not speedup evidence.
Packed runtime memory and energy were not measured. Relative latent L2 is a
diagnostic, not GenEval or a declared quality eligibility test.

## Interpretation

This greedy whole-transformer conversion fails the single-prompt visual smoke.
Finite tensors and successful conversion do not establish image quality. The
single-projection case only establishes bounded functional feasibility for that
matrix and prompt; it cannot validate a fully compressed transformer.

## Next experiment

- `shiftadd/configs/a100-radix-activation-smoke-001.json`: joint signed-radix fit, then activation diagnostics.
