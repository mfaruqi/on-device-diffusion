---
type: experiment-record
id: jetson-flux-klein-base-002
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-base-002__baseline__20261005-024556, jetson-flux-klein-base-002__attempt__20261005-023028]
updated: 2026-10-05
---

# Jetson klein Base: EasyCache on the disk-backed Q4 workload

Status: **complete**; correctness attempt and unprofiled1+3+10 protocol independently validated.

## Question

Can EasyCache execute the pinned fifty-step Base workload with the existing Jetson memory policy? This serves Week 1–2 image baseline work and RQ1. The only execution change from the [no-cache reference](jetson-flux-klein-base-001.md) is approximate step reuse.

## Setup

[Attempt config](../configs/jetson-flux-klein-base-q4-512-easycache-smoke.json), [full config](../configs/jetson-flux-klein-base-q4-512-easycache.json). sd.cpp commit19bbbca1c736bbb9538679fc0ae690cb2b46b492, isolated runner snapshot92580bb. Base Q4_0 checkpoint revision d12671125306ca6b5f6db1b33ed4c80c8511a53f; unchanged pinned Qwen3 Q4_K_M and original VAE.512×512,50 Euler steps, flux2 schedule, guidance4, seed0, one cat/sign prompt. Jetson Orin Nano25W, four threads, disk parameters, segmentation and prefetch enabled, eager load, diffusion flash attention; no auto-fit, conditioning cache or tiling. EasyCache threshold0.2, start0.15, end0.95 explicitly requested and log-audited.

One first and one measured observation, no warm-ups. [Metric definitions](../wiki/methods/baseline-metrics.md). Times are host callback durations including on-demand loading.

## Results

[Run](../results/runs/jetson-flux-klein-base-002__attempt__20261005-023028/), [summary](../results/runs/jetson-flux-klein-base-002__attempt__20261005-023028/summary.json), [environment](../results/runs/jetson-flux-klein-base-002__attempt__20261005-023028/environment.json), [status](../results/runs/jetson-flux-klein-base-002__attempt__20261005-023028/status.json).

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
| wall_ms | 151837.054 | 184900.394 |
| text_encode_ms | 3235.541 | 22399.402 |
| denoise_ms | 144469.053 | 157859.932 |
| vae_decode_ms | 3693.760 | 4384.475 |
| other_ms | 438.700 | 256.585 |

Context creation:62.146s after weight hashing warmed file cache. Whole-capture one-second system RAM peak:5.988GiB; sampled swap:0.274–0.533GiB. Memory includes loading and both images; it is not per-stage allocator memory.

## Validation and observations

All fifty callback steps remain present. EasyCache initializes separately for each generation and reports25/50 skipped steps each time. Independent timestamp-filtered logs show fifty actual transformer executions per generation, consistent with two CFG passes on each of the twenty-five computed steps. Settings, phase sequence, dimensions, summary aggregates and saved PNG pixel hashes passed independent checks. Both outputs have the same RGB hash. [Validation receipt](../results/runs/jetson-flux-klein-base-002__attempt__20261005-023028/cache-validation.json).

The saved image was visually inspected: coherent tabby cat, wooden sign, legible hello world; no obvious gross corruption. This is functional evidence for one prompt/seed. Paired pixel diagnostics and comparison limits are in the [comparison record](compare-jetson-base-easycache.md). The subsequent [CPU diagnostic](../results/runs/jetson-flux-klein-base-002__attempt__20261005-023028/quality-diagnostics.json) measured LPIPS AlexNet v0.1=0.008252; the same-image sanity check was zero. Method, evaluator versions and weight hashes are recorded separately from generation measurements. No formal quality eligibility is claimed.

## Repeated protocol

[Full run](../results/runs/jetson-flux-klein-base-002__baseline__20261005-024556/), [summary](../results/runs/jetson-flux-klein-base-002__baseline__20261005-024556/summary.json), [validation](../results/runs/jetson-flux-klein-base-002__baseline__20261005-024556/cache-validation.json). One first generation, three discarded warm-ups, ten measured generations, one model context.

| Metric | Median, ms | Min–max, ms |
|---|---:|---:|
| wall_ms | 184192.778 | 181321.881–190636.573 |
| text_encode_ms | 22132.515 | 13932.880–25001.993 |
| denoise_ms | 158403.877 | 154372.333–164174.193 |
| vae_decode_ms | 4454.548 | 3859.684–4591.680 |
| other_ms | 281.340 | 257.282–292.182 |

First generation:151174.187ms. Context creation:60.720s. Sampled whole-capture system RAM peak:5.996GiB; swap:0.312–0.404GiB. These memory values include loading and do not isolate cache allocations.

All fourteen generations passed callback/CFG/settings/phase checks and recorded25 skipped steps plus50 actual transformer executions each. All saved RGB hashes match one another and the validated attempt image. Therefore the previously evaluated pixel pair is unchanged: the [PSNR/LPIPS receipt](../results/runs/jetson-flux-klein-base-002__attempt__20261005-023028/quality-diagnostics.json) applies to these same pixels. Summary medians/min/max reproduce the measured CSV rows. No formal quality eligibility is claimed.

[W&B repeated run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-base-002__baseline__20261005-024556).

## Interpretation

The configured EasyCache path runs successfully with this disk-backed workload, and its skip reports agree with actual transformer execution counts. Callback steps still include cache decision and reuse work. The full repeated protocol establishes latency for this configuration. Selection for a combined policy remains subject to the matching limits in the comparison record.

## Next experiment

- Run the separate no-cache configs/jetson-flux-klein-base-q4-512-disk-profile.json capture.

[W&B attempt](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-base-002__attempt__20261005-023028).
