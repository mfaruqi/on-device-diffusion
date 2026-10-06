---
type: experiment-record
id: jetson-flux-klein-005
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-005__baseline__20261005-001522, jetson-flux-klein-005__attempt__20261005-000734, jetson-flux-klein-005__attempt__20261005-000956]
updated: 2026-10-05
---

# jetson-flux-klein-005: lazy loading with fixed disk residency

Status: **complete** (2026-10-05); full unprofiled 1+3+10 protocol completed.

## Question

Does disabling eager loading change feasibility and generation costs for the established
quantized disk-backed workload? This is a single memory-policy change against
[003](jetson-flux-klein-003.md), serving the image-baseline/profiling work and RQ1.

## Setup

[Configuration](../configs/jetson-flux-klein-q4-512-disk-lazy.json) changes eager_load to false.
The pinned Q4_0 transformer, Q4_K_M text encoder and BF16 VAE, fixed disk parameters,
CUDA computation, segmentation, four Euler steps, flux2 scheduler, 512×512, guidance1,
seed0 and 25W power mode remain unchanged. No reuse is enabled.
The [attempt config](../configs/jetson-flux-klein-q4-512-disk-lazy-smoke.json) uses first plus
one measured generation, no warm-ups. These short attempts do not estimate a warm baseline.
[Metric definitions](../wiki/methods/baseline-metrics.md).

## Results: correctness attempt

[Complete run](../results/runs/jetson-flux-klein-005__attempt__20261005-000956/), [summary](../results/runs/jetson-flux-klein-005__attempt__20261005-000956/summary.json),
[environment](../results/runs/jetson-flux-klein-005__attempt__20261005-000956/environment.json) and [status](../results/runs/jetson-flux-klein-005__attempt__20261005-000956/status.json).

| Metric | First (ms) | Second observation (ms) |
|---|---:|---:|
| wall_ms | 52838.774 | 49404.561 |
| text_encode_ms | 16514.518 | 11577.956 |
| denoise_ms | 31269.050 | 34164.758 |
| vae_decode_ms | 4752.430 | 3360.169 |
| other_ms | 302.776 | 301.678 |

Model-context creation took 0.968 seconds after file hashing warmed the filesystem cache.
The monitor's one-second whole-capture system-RAM peak was 4.229 GiB; this includes
loading and both generations, not a per-stage allocation peak
([summary](../results/runs/jetson-flux-klein-005__attempt__20261005-000956/summary.json)).

Both outputs have the same RGB hash. The inspected 512×512 image shows a coherent cat
holding a sign with legible hello world text and no obvious gross corruption
([inspection](../results/runs/jetson-flux-klein-005__attempt__20261005-000956/image-inspection.json)). This is functional evidence,
not a formal quality evaluation. Callback counts, dimensions, phase sequence and requested
lazy-loading settings passed validation.

## Results: repeated baseline

[Run](../results/runs/jetson-flux-klein-005__baseline__20261005-001522/), [summary](../results/runs/jetson-flux-klein-005__baseline__20261005-001522/summary.json), [environment](../results/runs/jetson-flux-klein-005__baseline__20261005-001522/environment.json). One first generation, three discarded warm-ups and ten measured generations completed with one model context.

| Metric | Median (ms) | Min–max (ms) |
|---|---:|---:|
| wall_ms | 60922.093 | 56826.812–66477.466 |
| text_encode_ms | 21210.954 | 15096.791–22082.510 |
| denoise_ms | 35970.152 | 34026.075–40995.305 |
| denoise_step_0_ms | 27816.947 | 25891.699–32827.226 |
| vae_decode_ms | 4578.507 | 3822.426–4802.677 |
| other_ms | 292.862 | 256.683–313.925 |

First generation: 54054.979 ms. Model-context creation: 0.867 seconds; file hashing warmed the filesystem cache. Whole-capture system RAM peak: 4.186 GiB, sampled once per second and including load. This is not a per-stage allocator measurement.

All fourteen generation rows have the same RGB hash. The saved image is coherent with legible sign text; this is functional evidence, not a quality benchmark.

## Interrupted attempt

[Earlier attempt](../results/runs/jetson-flux-klein-005__attempt__20261005-000734/status.json)
ended with KeyboardInterrupt after an accidental terminal interruption. It is retained
separately and is not classified as an engine memory or compatibility failure.
The retry used the same configuration.

## Interpretation

The lazy-loading configuration completed the repeated protocol. The first denoising step accounts for most denoising latency; later steps each have a median near 2.7 seconds. Callback durations include on-demand loading and do not isolate GPU compute. A separate matched comparison is needed before attributing a performance change to loading policy.

## Next experiment

- Compare against the fixed eager-loading reference, retaining differences in capture date and cache/background conditions.

W&B copies: [complete attempt](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-005__attempt__20261005-000956), [interrupted attempt](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-005__attempt__20261005-000734).

W&B baseline: [full protocol](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-005__baseline__20261005-001522).
