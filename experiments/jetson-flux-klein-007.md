---
type: experiment-record
id: jetson-flux-klein-007
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-007__baseline__20261005-113106, jetson-flux-klein-007__attempt__20261005-104736]
updated: 2026-10-05
---

# jetson-flux-klein-007: distilled klein exact conditioning reuse

Status: **complete**; functional attempt and full repeated protocol validated.

## Question

Does exact conditioning reuse execute correctly for the original four-step workload? Week2 image baselines, RQ1: “Which diffusion optimization choices transfer across devices and workloads?” ([proposal overview](../wiki/project/overview.md)). The reference is the separately recorded no-cache control; this record does not establish a comparative performance result.

## Two-generation correctness attempt: jetson-flux-klein-007__attempt__20261005-104736

[Config](../results/runs/jetson-flux-klein-007__attempt__20261005-104736/config.json), [summary](../results/runs/jetson-flux-klein-007__attempt__20261005-104736/summary.json), [status](../results/runs/jetson-flux-klein-007__attempt__20261005-104736/status.json), [environment](../results/runs/jetson-flux-klein-007__attempt__20261005-104736/environment.json), [validation](../results/runs/jetson-flux-klein-007__attempt__20261005-104736/validation.json).

Pinned sd.cpp19bbbca; Jetson Orin Nano25W; distilled klein transformer Q4_0, Qwen3 Q4_K_M and BF16 VAE;512×512,4Euler/flux2steps,CFG1,seed0,batch1. Fixed prompt: “A cat holding a sign that says hello world”. One context, first generation then one measured observation; no discarded warm-up. The single option is **exact conditioning reuse**. Engine snapshot31f3be71d07adda2fb122de0ef75a056c27ec7b1; four CPU threads, eager disk-backed segmented CUDA execution. Requested settings and engine log audit are preserved in the summary; they do not establish continuous residency or physical I/O.

[Metric definitions](../wiki/methods/baseline-metrics.md): host callback times include on-demand loading; no CUDA-event span or allocator peak is available.

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
| wall_ms | 25917.423 | 36491.045 |
| text_encode_ms | 2266.852 | 11.140 |
| denoise_ms | 19658.575 | 31985.233 |
| vae_decode_ms | 3584.906 | 4314.512 |
| other_ms | 407.090 | 180.160 |

Context creation 63.373436s after hash checks warmed the file cache. Whole-capture sampled RAM peak 5.996094GiB, swap 0.322266–0.329102GiB, 126samples; includes model loading and both generations.

Callback, settings, phase, aggregate and image-hash validation passed. Conditioning hits per generation: [0, 1]; actual transformer passes: [4, 4]. Both saved images are512×512RGB. Reference image equality is recorded in validation; no formal quality eligibility or held-out evaluation.

Conditioning cache capacity1 retains the fixed prompt between generations; the text stage after a hit measures retrieval/setup, not a fresh encoder pass. Approximate step caching remains off.

[Paired image diagnostic](../results/runs/jetson-flux-klein-007__attempt__20261005-104736/quality-diagnostics.json): MSE0,infinite PSNR,LPIPS0 against the saved control image, one development prompt/seed only.

Interpretation: functional validation passed; one measured observation cannot qualify an option under the full-protocol speed-range rule.

Next experiment: unchanged full1+3+10protocol using `configs/jetson-flux-klein-q4-512-conditioning-cache.json`.

## Full repeated protocol

[Summary](../results/runs/jetson-flux-klein-007__baseline__20261005-113106/summary.json), [config](../results/runs/jetson-flux-klein-007__baseline__20261005-113106/config.json), [environment](../results/runs/jetson-flux-klein-007__baseline__20261005-113106/environment.json), [status](../results/runs/jetson-flux-klein-007__baseline__20261005-113106/status.json). Completed 15:31:06–15:37:21 UTC. Same pinned workload, engine and snapshot as the attempt above; one first generation, three discarded warm-ups and ten measured generations in one context.

| Metric | Median, ms | Min–max, ms |
|---|---:|---:|
| wall_ms | 15734.541 | 15589.200–20335.525 |
| text_encode_ms | 11.550 | 11.239–14.953 |
| denoise_ms | 12186.432 | 12096.185–16846.371 |
| vae_decode_ms | 3336.363 | 3313.352–3859.701 |
| other_ms | 180.117 | 158.056–183.012 |

First generation 25872.131 ms; context creation 63.078454 s after file-cache-warming hash checks. Whole-capture system RAM peak 6.010742 GiB, swap 0.329102–0.330078 GiB, 328 samples.

[Independent validation](../results/runs/jetson-flux-klein-007__baseline__20261005-113106/validation.json) passed: all 14 generations executed four transformer passes; conditioning hits were zero initially and one for every later generation. Approximate step caching stayed off. All 14 reported RGB hashes agree; retained first and measured PNG hashes were checked. [Paired diagnostic](../results/runs/jetson-flux-klein-007__baseline__20261005-113106/quality-diagnostics.json): reference-identical pixels, MSE 0, infinite PSNR, LPIPS 0 for the fixed development prompt/seed. No formal quality eligibility.

The measured text stage covers cached conditioning retrieval/setup, not encoder execution. This policy reuses the identical prompt between images; it does not skip denoising steps. The [matched comparison](compare-jetson-distilled-conditioning-reuse.md) reports stage differences and selection. No claim about memory residency or physical storage traffic follows from these timing records alone.
