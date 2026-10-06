---
type: experiment-record
id: jetson-flux-klein-009
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-009__baseline__20261005-115227, jetson-flux-klein-009__attempt__20261005-105344]
updated: 2026-10-05
---

# jetson-flux-klein-009: distilled klein memory-mapped weight-file I/O

Status: **complete**; functional attempt and full repeated protocol validated.

## Question

Does memory-mapped weight-file I/O execute correctly for the original four-step workload? Week2 image baselines, RQ1: “Which diffusion optimization choices transfer across devices and workloads?” ([proposal overview](../wiki/project/overview.md)). The reference is the separately recorded no-cache control; this record does not establish a comparative performance result.

## Two-generation correctness attempt: jetson-flux-klein-009__attempt__20261005-105344

[Config](../results/runs/jetson-flux-klein-009__attempt__20261005-105344/config.json), [summary](../results/runs/jetson-flux-klein-009__attempt__20261005-105344/summary.json), [status](../results/runs/jetson-flux-klein-009__attempt__20261005-105344/status.json), [environment](../results/runs/jetson-flux-klein-009__attempt__20261005-105344/environment.json), [validation](../results/runs/jetson-flux-klein-009__attempt__20261005-105344/validation.json).

Pinned sd.cpp19bbbca; Jetson Orin Nano25W; distilled klein transformer Q4_0, Qwen3 Q4_K_M and BF16 VAE;512×512,4Euler/flux2steps,CFG1,seed0,batch1. Fixed prompt: “A cat holding a sign that says hello world”. One context, first generation then one measured observation; no discarded warm-up. The single option is **memory-mapped weight-file I/O**. Engine snapshot31f3be71d07adda2fb122de0ef75a056c27ec7b1; four CPU threads, eager disk-backed segmented CUDA execution. Requested settings and engine log audit are preserved in the summary; they do not establish continuous residency or physical I/O.

[Metric definitions](../wiki/methods/baseline-metrics.md): host callback times include on-demand loading; no CUDA-event span or allocator peak is available.

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
| wall_ms | 26068.969 | 58230.722 |
| text_encode_ms | 2390.184 | 21095.813 |
| denoise_ms | 19638.322 | 32288.230 |
| vae_decode_ms | 3643.950 | 4580.582 |
| other_ms | 396.513 | 266.097 |

Context creation 63.695277s after hash checks warmed the file cache. Whole-capture sampled RAM peak 5.962891GiB, swap 0.325195–0.499023GiB, 149samples; includes model loading and both generations.

Callback, settings, phase, aggregate and image-hash validation passed. Conditioning hits per generation: [0, 0]; actual transformer passes: [4, 4]. Both saved images are512×512RGB. Reference image equality is recorded in validation; no formal quality eligibility or held-out evaluation.

Engine-reported mapped component files: ['Qwen3-4B-Q4_K_M.gguf', 'flux-2-klein-4b-Q4_0.gguf', 'flux2-vae.safetensors']. These are weight-file mappings, not evidence of GPU zero-copy or fully resident weights.

[Paired image diagnostic](../results/runs/jetson-flux-klein-009__attempt__20261005-105344/quality-diagnostics.json): MSE0,infinite PSNR,LPIPS0 against the saved control image, one development prompt/seed only.

Interpretation: functional validation passed; one measured observation cannot qualify an option under the full-protocol speed-range rule.

Next experiment: unchanged full1+3+10protocol using `configs/jetson-flux-klein-q4-512-mmap.json`.

## Full repeated protocol

[Summary](../results/runs/jetson-flux-klein-009__baseline__20261005-115227/summary.json), [config](../results/runs/jetson-flux-klein-009__baseline__20261005-115227/config.json), [environment](../results/runs/jetson-flux-klein-009__baseline__20261005-115227/environment.json), [status](../results/runs/jetson-flux-klein-009__baseline__20261005-115227/status.json). Completed 15:52:27–16:09:17 UTC. Same pinned workload, engine and snapshot as the attempt above. One context; first generation separate, three warm-ups discarded, ten measured generations.

| Metric | Median, ms | Min–max, ms |
|---|---:|---:|
| wall_ms | 67758.838 | 66997.369–68557.088 |
| text_encode_ms | 23245.910 | 22718.246–23439.282 |
| denoise_ms | 39497.698 | 39095.780–40407.290 |
| vae_decode_ms | 4641.656 | 4562.725–4678.250 |
| other_ms | 287.717 | 258.120–340.494 |

First generation 26010.361 ms; context creation 65.420274 s after file-cache-warming hash checks. Whole-capture system RAM peak 5.953125 GiB, swap 0.329102–0.949219 GiB, 955 samples.

[Validation](../results/runs/jetson-flux-klein-009__baseline__20261005-115227/validation.json) passed: all 14 generations executed four transformer passes, with zero conditioning hits and approximate step caching off. Weight-file mappings for all three component files were confirmed in the engine log, with no fallback detected. All reported RGB hashes agree; retained first and measured PNG hashes were checked independently. [Image diagnostic](../results/runs/jetson-flux-klein-009__baseline__20261005-115227/quality-diagnostics.json): reference-identical pixels, MSE 0, infinite PSNR, LPIPS 0 for one development prompt/seed; no formal quality eligibility.

Interpretation: the explicit mmap policy completes the repeated protocol without changing this workload's output. The [matched comparison](compare-jetson-distilled-mmap.md) reports whether its timing distribution improves on the reference. Physical I/O and continuous memory residency are not measured by this run.
