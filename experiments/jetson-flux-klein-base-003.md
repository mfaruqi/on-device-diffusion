---
type: experiment-record
id: jetson-flux-klein-base-003
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-base-003__attempt__20261005-054530, jetson-flux-klein-base-003__baseline__20261005-060028]
updated: 2026-10-05
---

# Jetson klein Base: lazy disk-backed loading

Status: **complete**; correctness attempt and full unprofiled repeated protocol validated.

## Question

Does disabling eager loading preserve execution and output for the pinned Base Q4 disk-backed workload? This single residency-policy variant serves Week 2 image baselines and RQ1. The full repeated protocol measures latency and whole-capture memory.

## Setup

[Attempt config](../configs/jetson-flux-klein-base-q4-512-disk-lazy-smoke.json), [full config](../configs/jetson-flux-klein-base-q4-512-disk-lazy.json). Reference: configs/jetson-flux-klein-base-q4-512-disk.json on the same cache-capable binary. Only eager loading changes to false. Base Q4_0, Qwen3 Q4_K_M and original VAE pins remain unchanged; fifty Euler steps, flux2 scheduler, guidance4,512×512,batch1,seed0 and fixed cat/sign prompt. Disk-backed parameters, segmented CUDA computation, prefetch and diffusion flash attention remain enabled. No step cache, conditioning reuse, automatic fitting or VAE tiling.

Jetson Orin Nano,25W, four CPU threads. sd.cpp19bbbca1c736bbb9538679fc0ae690cb2b46b492, binary SHA2562d66fa04053c02b441d61a84d24412537a16f9099d6cb31b80dcca3c8cc1f896. [Environment](../results/runs/jetson-flux-klein-base-003__attempt__20261005-054530/environment.json); [metric definitions](../wiki/methods/baseline-metrics.md). Protocol:one first plus one additional observation, no warm-ups. These are correctness observations, not warm latency estimates.

## Correctness attempt

[Run](../results/runs/jetson-flux-klein-base-003__attempt__20261005-054530/), [summary](../results/runs/jetson-flux-klein-base-003__attempt__20261005-054530/summary.json), [status](../results/runs/jetson-flux-klein-base-003__attempt__20261005-054530/status.json).

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
| wall_ms | 316025.111 | 307112.599 |
| text_encode_ms | 11907.905 | 5743.825 |
| denoise_ms | 299748.728 | 296874.026 |
| vae_decode_ms | 4081.500 | 4191.796 |
| other_ms | 286.978 | 302.952 |

Context creation:1.061s, with parameter loading deferred into generation. File hashing occurs before measurement and warms filesystem cache. Whole-capture system RAM peak:4.065GiB; swap range:0.303–0.463GiB from623one-second samples. These include loading and both generations; no stage-specific allocator peak is available.

## Validation and quality diagnostics

[Validation receipt](../results/runs/jetson-flux-klein-base-003__attempt__20261005-054530/lazy-validation.json). Each generation completed fifty callback steps and one hundred single-segment transformer passes. Requested eager_load=false was confirmed; cache audit records zero enabled generations. Settings, phase order, summary aggregates, dimensions and saved PNG hashes passed independent checks. Parameter-placement logs show preparation on CUDA0 during execution; continuous placement and physical disk traffic remain unverified.

Both saved512×512images are pixel-identical to the same-binary no-cache reference. [Paired diagnostic receipt](../results/runs/jetson-flux-klein-base-003__attempt__20261005-054530/quality-diagnostics.json): RGB MSE0, PSNR infinite because pixels are identical, calibrated AlexNet LPIPSv0.1=0 with same-image control0. Full RGB images were evaluated without crop/resize on CPU; versions, input hashes and weight hashes are retained. The reference image was visually inspected as a coherent cat holding a readable sign. This is one development prompt/seed, not formal quality eligibility.

## Full repeated protocol

[Run](../results/runs/jetson-flux-klein-base-003__baseline__20261005-060028/), [summary](../results/runs/jetson-flux-klein-base-003__baseline__20261005-060028/summary.json), [status](../results/runs/jetson-flux-klein-base-003__baseline__20261005-060028/status.json), [environment](../results/runs/jetson-flux-klein-base-003__baseline__20261005-060028/environment.json). Completed 10:00:28–11:18:14 UTC on October 5. One first generation, three discarded warm-ups and ten measured generations; one persistent model context.

| Metric | Median, ms | Measured min–max, ms |
|---|---:|---:|
| wall_ms | 330683.062 | 329500.221–331511.407 |
| text_encode_ms | 24536.232 | 24197.899–25267.986 |
| denoise_ms | 301083.412 | 300215.457–302285.861 |
| vae_decode_ms | 4643.861 | 4556.354–4701.574 |
| other_ms | 291.382 | 260.296–334.051 |

First generation: 336056.358 ms. Context creation: 0.922235 s. Whole-capture system RAM peak: 4.157227 GiB; swap 0.309570–0.600586 GiB across 4584 samples; maximum sampled GPU temperature 63.625°C. These are whole-system capture metrics, not stage-specific allocations.

All fourteen saved RGB images match. Every generation has fifty callback steps and one hundred transformer passes. Cache remains disabled; requested eager loading is false. Independent summary, phase, callback, settings and pixel checks passed ([validation](../results/runs/jetson-flux-klein-base-003__baseline__20261005-060028/lazy-validation.json)). Full-run paired diagnostic: MSE0, infinite PSNR and LPIPS0 ([receipt](../results/runs/jetson-flux-klein-base-003__baseline__20261005-060028/quality-diagnostics.json)); one development prompt/seed, no formal quality eligibility.

[W&B baseline](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-base-003__baseline__20261005-060028).

## Interpretation

The lazy-loading configuration passes the functional gate. A short context-creation duration excludes deferred parameter work and must not be interpreted as an end-to-end speedup. The full protocol confirms stable output while retaining parameter preparation during generation. Selection against the reference is documented in the [comparison](compare-jetson-base-eager-vs-lazy.md).

## Next experiment

- Evaluate a separately labelled prefetch-disabled configuration against the eager no-cache reference.

[W&B attempt](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-base-003__attempt__20261005-054530).
