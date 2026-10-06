---
type: experiment-record
id: jetson-flux-klein-base-006
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-base-006__baseline__20261005-131024, jetson-flux-klein-base-006__attempt__20261005-125535]
updated: 2026-10-05
---

# jetson-flux-klein-base-006: memory-mapped Base weight files

Status: **complete**; correctness gate and full repeated protocol independently validated.

## Question

Can memory-mapped weight-file I/O execute the unchanged Base workload with correct outputs? Week 2 image-engine baselines, RQ1: “Which diffusion optimization choices transfer across devices and workloads?” ([proposal overview](../wiki/project/overview.md)).

## Setup

[Config](../results/runs/jetson-flux-klein-base-006__attempt__20261005-125535/config.json), [environment](../results/runs/jetson-flux-klein-base-006__attempt__20261005-125535/environment.json), [status](../results/runs/jetson-flux-klein-base-006__attempt__20261005-125535/status.json). Jetson Orin Nano, 25W, four CPU threads, pinned Base Q4_0 transformer, Qwen3 Q4_K_M encoder and BF16 VAE. 512×512, 50 Euler/flux2 steps, CFG 4, seed 0, batch 1, fixed development prompt. Eager disk-backed segmented CUDA execution, prefetch enabled, conditioning and denoising caches disabled. The sole policy change is mmap enabled. No checkpoint, precision or residency fallback.

Isolated snapshot e614bb0cf8041b65cf4e327d70d5cb1079f3c33e; sd.cpp19bbbca and unchanged cache-capable binary. One context, first generation then one measured observation, no warm-ups. Completed 16:55:35–17:07:33 UTC. [Metric definitions](../wiki/methods/baseline-metrics.md).

## Results

[Summary](../results/runs/jetson-flux-klein-base-006__attempt__20261005-125535/summary.json). These are two individual observations, not a full repeated benchmark.

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
| wall_ms | 287678.530 | 322747.466 |
| text_encode_ms | 3296.551 | 24483.420 |
| denoise_ms | 280379.549 | 293392.445 |
| vae_decode_ms | 3625.135 | 4591.984 |
| other_ms | 377.295 | 279.617 |

Context creation 65.495819 s after file-cache-warming hash checks. Whole-capture RAM peak 5.938477 GiB; swap 0.342773–0.591797 GiB across 673 samples. These are whole-system measurements, not per-stage allocations.

## Validation and quality

[Independent validation](../results/runs/jetson-flux-klein-base-006__attempt__20261005-125535/validation.json) passed: two 512×512 outputs, 50 scheduler steps and 100 actual CFG transformer passes per image, zero conditioning hits and no denoising cache. Engine audit confirms all three component file mappings with no fallback. Both reported pixel hashes agree, and retained PNG hashes match their rows. The binary hash matches the no-cache reference.

[Paired diagnostic](../results/runs/jetson-flux-klein-base-006__attempt__20261005-125535/quality-diagnostics.json) records identical RGB pixels, PSNR infinity and LPIPS 0 against the no-cache reference for the retained measured image. One development prompt/seed only; no formal quality eligibility.

## Interpretation

The mmap policy passes the functional gate for this configuration. A file mapping does not establish continuous GPU residency, zero-copy execution or physical storage traffic. One measured observation cannot establish a speed benefit. Swap grew during the capture; the full protocol retains this metric rather than treating successful generation as proof of no memory pressure.

## Next experiment

None scheduled; the authorized Jetson batch is complete.

## Full repeated protocol

[Summary](../results/runs/jetson-flux-klein-base-006__baseline__20261005-131024/summary.json), [config](../results/runs/jetson-flux-klein-base-006__baseline__20261005-131024/config.json), [environment](../results/runs/jetson-flux-klein-base-006__baseline__20261005-131024/environment.json), [status](../results/runs/jetson-flux-klein-base-006__baseline__20261005-131024/status.json). Completed 17:10:24–18:27:35 UTC in the same isolated snapshot as the smoke test. One context; one first generation, three discarded warm-ups, ten measured images; no profiler.

| Metric | Median, ms | Min–max, ms |
|---|---:|---:|
| wall_ms | 329425.282 | 328578.649–330363.177 |
| text_encode_ms | 24306.888 | 24049.472–24551.362 |
| denoise_ms | 300190.007 | 299476.021–301087.027 |
| vae_decode_ms | 4624.989 | 4577.930–4683.022 |
| other_ms | 307.835 | 275.818–355.469 |

First generation 287840.433 ms; context creation 58.348417 s after file-cache-warming hash checks. Whole-capture RAM peak 5.993164 GiB, swap 0.342773–0.961914 GiB, 4561 samples. Swap grew during this capture; no per-stage memory attribution is available.

[Validation](../results/runs/jetson-flux-klein-base-006__baseline__20261005-131024/validation.json) passed all 14 generations: 50 scheduler steps and 100 actual CFG transformer passes each, no denoising cache and zero conditioning hits. Three component file mappings confirmed without fallback. All 14 reported RGB hashes agree; the two retained PNGs match their recorded hashes. [Paired diagnostic](../results/runs/jetson-flux-klein-base-006__baseline__20261005-131024/quality-diagnostics.json): retained measured pixels match no-cache, PSNR infinity and LPIPS 0. One development prompt/seed, no formal quality eligibility.

[Matched mmap comparison](compare-jetson-base-mmap.md) records the overlapping latency ranges and failed diagnostic speed-selection rule. This configuration is not evidence for a speed-selected mmap combination.
