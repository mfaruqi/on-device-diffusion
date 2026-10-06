---
type: experiment-record
id: jetson-flux-klein-008
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-008__baseline__20261005-113721, jetson-flux-klein-008__attempt__20261005-105027]
updated: 2026-10-05
---

# jetson-flux-klein-008: distilled klein prefetch disabled

Status: **complete**; functional attempt and full repeated protocol validated.

## Question

Does prefetch disabled execute correctly for the original four-step workload? Week2 image baselines, RQ1: “Which diffusion optimization choices transfer across devices and workloads?” ([proposal overview](../wiki/project/overview.md)). The reference is the separately recorded no-cache control; this record does not establish a comparative performance result.

## Two-generation correctness attempt: jetson-flux-klein-008__attempt__20261005-105027

[Config](../results/runs/jetson-flux-klein-008__attempt__20261005-105027/config.json), [summary](../results/runs/jetson-flux-klein-008__attempt__20261005-105027/summary.json), [status](../results/runs/jetson-flux-klein-008__attempt__20261005-105027/status.json), [environment](../results/runs/jetson-flux-klein-008__attempt__20261005-105027/environment.json), [validation](../results/runs/jetson-flux-klein-008__attempt__20261005-105027/validation.json).

Pinned sd.cpp19bbbca; Jetson Orin Nano25W; distilled klein transformer Q4_0, Qwen3 Q4_K_M and BF16 VAE;512×512,4Euler/flux2steps,CFG1,seed0,batch1. Fixed prompt: “A cat holding a sign that says hello world”. One context, first generation then one measured observation; no discarded warm-up. The single option is **prefetch disabled**. Engine snapshot31f3be71d07adda2fb122de0ef75a056c27ec7b1; four CPU threads, eager disk-backed segmented CUDA execution. Requested settings and engine log audit are preserved in the summary; they do not establish continuous residency or physical I/O.

[Metric definitions](../wiki/methods/baseline-metrics.md): host callback times include on-demand loading; no CUDA-event span or allocator peak is available.

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
| wall_ms | 25755.634 | 60505.340 |
| text_encode_ms | 2258.493 | 23034.537 |
| denoise_ms | 19625.576 | 32628.154 |
| vae_decode_ms | 3479.185 | 4545.999 |
| other_ms | 392.380 | 296.650 |

Context creation 61.742414s after hash checks warmed the file cache. Whole-capture sampled RAM peak 5.990234GiB, swap 0.324219–0.362305GiB, 148samples; includes model loading and both generations.

Callback, settings, phase, aggregate and image-hash validation passed. Conditioning hits per generation: [0, 0]; actual transformer passes: [4, 4]. Both saved images are512×512RGB. Reference image equality is recorded in validation; no formal quality eligibility or held-out evaluation.

Only engine prefetch is disabled; eager loading, parameter storage, segmented computation and caching settings remain explicit in the saved config.

[Paired image diagnostic](../results/runs/jetson-flux-klein-008__attempt__20261005-105027/quality-diagnostics.json): MSE0,infinite PSNR,LPIPS0 against the saved control image, one development prompt/seed only.

Interpretation: functional validation passed; one measured observation cannot qualify an option under the full-protocol speed-range rule.

Next experiment: unchanged full1+3+10protocol using `configs/jetson-flux-klein-q4-512-no-prefetch.json`.

## Full repeated protocol

[Summary](../results/runs/jetson-flux-klein-008__baseline__20261005-113721/summary.json), [config](../results/runs/jetson-flux-klein-008__baseline__20261005-113721/config.json), [environment](../results/runs/jetson-flux-klein-008__baseline__20261005-113721/environment.json), [status](../results/runs/jetson-flux-klein-008__baseline__20261005-113721/status.json). Completed 15:37:21–15:52:26 UTC. Same pinned workload, engine and snapshot as the attempt above. One context; first generation separate, three warm-ups discarded, ten measured generations.

| Metric | Median, ms | Min–max, ms |
|---|---:|---:|
| wall_ms | 60503.164 | 50312.039–65011.058 |
| text_encode_ms | 21274.854 | 13631.980–23862.450 |
| denoise_ms | 33938.654 | 30341.032–40776.559 |
| vae_decode_ms | 4488.260 | 4275.987–4802.754 |
| other_ms | 294.502 | 275.104–310.610 |

First generation 25053.133 ms; context creation 62.206236 s after file-cache-warming hash checks. Whole-capture system RAM peak 5.997070 GiB, swap 0.329102–0.330078 GiB, 854 samples.

[Validation](../results/runs/jetson-flux-klein-008__baseline__20261005-113721/validation.json) passed: all 14 generations executed four transformer passes, with zero conditioning hits and approximate step caching off. Prefetch-disabled settings were audited against the engine log. All reported RGB hashes agree; retained first and measured PNG hashes were checked independently. [Image diagnostic](../results/runs/jetson-flux-klein-008__baseline__20261005-113721/quality-diagnostics.json): reference-identical pixels, MSE 0, infinite PSNR, LPIPS 0 for one development prompt/seed; no formal quality eligibility.

Interpretation: the explicit prefetch-disabled policy completes the repeated protocol without changing this workload's output. The [matched comparison](compare-jetson-distilled-prefetch.md) reports whether its timing distribution improves on the reference. Physical I/O and continuous memory residency are not measured by this run.
