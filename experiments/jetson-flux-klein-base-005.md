---
type: experiment-record
id: jetson-flux-klein-base-005
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-base-005__baseline__20261005-093239, jetson-flux-klein-base-005__attempt__20261005-091031]
updated: 2026-10-05
---

# Jetson Base: exact prompt-conditioning reuse

Status: **complete**; correctness test and full repeated protocol independently validated.

## Question

Can exact reuse of fixed positive and negative prompt conditioning remove repeated encoding while preserving Base output? Week2 image-engine baseline work, RQ1. The only execution change relative to configs/jetson-flux-klein-base-q4-512-disk.json is conditioning_cache_size=2; approximate step caching remains disabled.

## Setup

[Attempt config](../results/runs/jetson-flux-klein-base-005__attempt__20261005-091031/config.json), [full config](../configs/jetson-flux-klein-base-q4-512-conditioning-cache.json), [environment](../results/runs/jetson-flux-klein-base-005__attempt__20261005-091031/environment.json). Jetson Orin Nano25W, four CPU threads; pinned Base Q4_0 transformer, Qwen3 Q4_K_M encoder and BF16 VAE,512²,50Euler steps,flux2 scheduler,CFG4,seed0,one cat/sign prompt. Fixed eager disk-backed segmented CUDA, diffusion flash attention, prefetch enabled, no tiling. Runner snapshot f389548 preserves older checkouts and reuses the cache-capable harness binary without rebuilding.

Two cache entries retain positive and negative conditioning separately. First generation must have zero hits, second two; both keep100transformer passes. [Metric definitions](../wiki/methods/baseline-metrics.md#exact-sdcpp-conditioning-reuse). Cache-hit text_encode_ms means conditioning retrieval and setup through the existing completion callback, not a new encoder forward. [Pinned-source evidence](../raw/engine-evidence/conditioning-source-evidence.txt).

## Results

[Summary](../results/runs/jetson-flux-klein-base-005__attempt__20261005-091031/summary.json), [status](../results/runs/jetson-flux-klein-base-005__attempt__20261005-091031/status.json), [raw observations](../results/runs/jetson-flux-klein-base-005__attempt__20261005-091031/runs.csv). Completed13:10:31–13:22:08UTC. One first and one additional observation, no warm-ups; not a repeated baseline estimate.

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
| wall_ms | 287080.571 | 298084.940 |
| text_encode_ms | 3008.453 | 13.028 |
| denoise_ms | 279740.584 | 293819.316 |
| vae_decode_ms | 3913.790 | 4096.872 |
| other_ms | 417.744 | 155.724 |

Context creation 61.894095s after weight verification warmed file cache. Whole-capture system RAM peak 6.014648GiB, swap 0.315430–0.317383GiB across645samples. These are system measurements covering loading and both generations, not isolated cache memory.

## Validation and quality

[Independent validation](../results/runs/jetson-flux-klein-base-005__attempt__20261005-091031/conditioning-validation.json): callbacks,50steps/100CFGpasses each, dimensions, hashes, phases, summary aggregates and unchanged binary passed. Hit counts are[0,2], independently confirmed by engine logs; step cache remains off. Both saved images match the no-cache reference pixels. [Paired diagnostics](../results/runs/jetson-flux-klein-base-005__attempt__20261005-091031/quality-diagnostics.json): MSE0,infinite PSNR,LPIPS0, self-distance0. One development prompt/seed only; no held-out or formal quality eligibility.

## Interpretation

Exact conditioning reuse is functionally supported for this fixed-prompt CFG workload. The hit-stage duration includes retrieval/setup overhead. The complete repeated protocol is reported below; cache capacity and behavior under multiple prompts remain outside this run.

## Next experiment

- Test an explicitly labelled combination with EasyCache after the queued distilled tests.

[W&B attempt](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-base-005__attempt__20261005-091031).

## Full repeated protocol

[Summary](../results/runs/jetson-flux-klein-base-005__baseline__20261005-093239/summary.json), [status](../results/runs/jetson-flux-klein-base-005__baseline__20261005-093239/status.json), [environment](../results/runs/jetson-flux-klein-base-005__baseline__20261005-093239/environment.json). Completed13:32:39–14:40:33UTC; first + three discarded warm-ups + ten measured generations.

| Metric | Median, ms | Min–max, ms |
|---|---:|---:|
| wall_ms | 279605.293 | 278047.652–285551.188 |
| text_encode_ms | 13.342 | 13.032–15.467 |
| denoise_ms | 276089.382 | 274535.797–282047.868 |
| vae_decode_ms | 3334.923 | 3310.370–4295.152 |
| other_ms | 177.985 | 156.929–179.120 |

First generation 287010.931ms; context creation 61.443845s with warmed file cache. Whole-capture RAM peak 6.022461GiB, swap 0.315430–0.317383GiB,4003samples.

[Validation](../results/runs/jetson-flux-klein-base-005__baseline__20261005-093239/conditioning-validation.json) passed: all14generations retain50steps/100CFGpasses, cache hits0then2for every subsequent generation, all14RGBhashes agree. [Quality diagnostics](../results/runs/jetson-flux-klein-base-005__baseline__20261005-093239/quality-diagnostics.json): reference-identical pixels,MSE0,infinite PSNR,LPIPS0; no formal quality eligibility. The text-stage values measure cached conditioning retrieval, not encoder execution. [Matched comparison and selection](compare-jetson-base-conditioning-reuse.md).

[W&B full run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-base-005__baseline__20261005-093239).
