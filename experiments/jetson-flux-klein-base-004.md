---
type: experiment-record
id: jetson-flux-klein-base-004
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-base-004__attempt__20261005-073439, jetson-flux-klein-base-004__baseline__20261005-074852]
updated: 2026-10-05
---

# Jetson Base: parameter prefetch disabled

Status: **complete**; correctness attempt and full repeated protocol independently validated.

## Question

Does disabling parameter prefetch preserve the fixed eager disk-backed Base execution, and how does it affect repeated latency and memory? Week2 image baseline evidence, RQ1. Reference configs/jetson-flux-klein-base-q4-512-disk.json; only prefetch changes.

## Setup

[Attempt config](../results/runs/jetson-flux-klein-base-004__attempt__20261005-073439/config.json), [full config](../configs/jetson-flux-klein-base-q4-512-no-prefetch.json), [environment](../results/runs/jetson-flux-klein-base-004__attempt__20261005-073439/environment.json). Same pinned Base Q4_0,Qwen3 Q4_K_M and VAE;512×512,fifty Euler steps,flux2 scheduler,CFG4,seed0,one cat/sign prompt. Jetson Orin Nano25W,four CPU threads,eager disk-backed parameters,segmented CUDA,flash attention,no cache or tiling. optimizations.prefetch=false maps to disable-prefetch1 and is audited as a requested setting. Separate runner snapshot6e67d68 preserves earlier campaigns and reuses the same unprofiled harness binary. [Metric definitions](../wiki/methods/baseline-metrics.md).

## Results

[Summary](../results/runs/jetson-flux-klein-base-004__attempt__20261005-073439/summary.json), [status](../results/runs/jetson-flux-klein-base-004__attempt__20261005-073439/status.json). Completed11:34:39–11:46:49UTC. One first and one additional observation,no warm-ups; these are functional observations, not baseline medians.

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
| wall_ms | 287564.840 | 329566.098 |
| text_encode_ms | 3276.197 | 24156.669 |
| denoise_ms | 280100.180 | 300480.011 |
| vae_decode_ms | 3738.345 | 4635.781 |
| other_ms | 450.118 | 293.637 |

Context creation 64.550034s. Whole-capture sampled system RAM peak 5.996094GiB, swap 0.308594–0.408203GiB across 679samples. This includes load and both generations, with no per-stage allocation measurement.

## Validation and quality

[Validation](../results/runs/jetson-flux-klein-base-004__attempt__20261005-073439/prefetch-validation.json): fifty steps and one hundred transformer passes per image; callback order, phase order, summary aggregates, dimensions and pixel hashes checked. Requested disable_prefetch=true,eager_load=true,cacheoff audited. Both images equal the no-cache reference pixels. [Paired diagnostics](../results/runs/jetson-flux-klein-base-004__attempt__20261005-073439/quality-diagnostics.json): MSE0,infinite PSNR,LPIPS0; one development prompt/seed, no formal quality eligibility. Physical storage traffic and continuous residency remain unverified.

## Interpretation

The option preserves deterministic output for this prompt and seed. Denoising dominates repeated generation time. Physical I/O overlap and continuous residency were not measured in this unprofiled run; latency alone cannot identify the prefetch mechanism.

## Next experiment

- Evaluate the separately labelled exact-conditioning reuse configuration.

[W&B attempt](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-base-004__attempt__20261005-073439).

## Full repeated protocol

Completed11:48:52–13:04:59UTC, one first, three discarded warm-ups, ten measured generations. [Summary](../results/runs/jetson-flux-klein-base-004__baseline__20261005-074852/summary.json), [status](../results/runs/jetson-flux-klein-base-004__baseline__20261005-074852/status.json), [environment](../results/runs/jetson-flux-klein-base-004__baseline__20261005-074852/environment.json).

| Stage | Median, ms | Min–max, ms |
|---|---:|---:|
| wall_ms | 320877.059 | 315421.228–328175.862 |
| text_encode_ms | 21929.209 | 16734.053–24684.082 |
| denoise_ms | 295679.244 | 289598.117–299968.072 |
| vae_decode_ms | 4367.119 | 4064.204–4589.003 |
| other_ms | 283.556 | 259.066–295.068 |

First generation 287351.453ms; context creation 62.948871s after weight verification warmed the filesystem cache. Whole-capture RAM peak 6.045898GiB; swap 0.313477–0.375000GiB, 4496 samples. These are system measurements, not per-stage allocations.

[Validation](../results/runs/jetson-flux-klein-base-004__baseline__20261005-074852/prefetch-validation.json) passed all14generations: fifty steps, one hundred CFG transformer passes each, callback boundaries, phase order, summary aggregates and RGB hashes. All images have the same decoded pixels. [Paired diagnostics](../results/runs/jetson-flux-klein-base-004__baseline__20261005-074852/quality-diagnostics.json): MSE0, infinite PSNR, LPIPS0; development prompt only, no formal quality eligibility. [Selection comparison](compare-jetson-base-prefetch.md).

[W&B full run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-base-004__baseline__20261005-074852).
