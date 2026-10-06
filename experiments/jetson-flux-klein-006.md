---
type: experiment-record
id: jetson-flux-klein-006
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-006__baseline__20261005-111428, jetson-flux-klein-006__attempt__20261005-104417]
updated: 2026-10-05
---

# jetson-flux-klein-006: distilled klein EasyCache

Status: **complete**; functional attempt and full repeated protocol validated.

## Question

Does EasyCache execute correctly for the original four-step workload? Week2 image baselines, RQ1: “Which diffusion optimization choices transfer across devices and workloads?” ([proposal overview](../wiki/project/overview.md)). The reference is the separately recorded no-cache control; this record does not establish a comparative performance result.

## Two-generation correctness attempt: jetson-flux-klein-006__attempt__20261005-104417

[Config](../results/runs/jetson-flux-klein-006__attempt__20261005-104417/config.json), [summary](../results/runs/jetson-flux-klein-006__attempt__20261005-104417/summary.json), [status](../results/runs/jetson-flux-klein-006__attempt__20261005-104417/status.json), [environment](../results/runs/jetson-flux-klein-006__attempt__20261005-104417/environment.json), [validation](../results/runs/jetson-flux-klein-006__attempt__20261005-104417/validation.json).

Pinned sd.cpp19bbbca; Jetson Orin Nano25W; distilled klein transformer Q4_0, Qwen3 Q4_K_M and BF16 VAE;512×512,4Euler/flux2steps,CFG1,seed0,batch1. Fixed prompt: “A cat holding a sign that says hello world”. One context, first generation then one measured observation; no discarded warm-up. The single option is **EasyCache**. Engine snapshot31f3be71d07adda2fb122de0ef75a056c27ec7b1; four CPU threads, eager disk-backed segmented CUDA execution. Requested settings and engine log audit are preserved in the summary; they do not establish continuous residency or physical I/O.

[Metric definitions](../wiki/methods/baseline-metrics.md): host callback times include on-demand loading; no CUDA-event span or allocator peak is available.

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
| wall_ms | 25803.655 | 66367.282 |
| text_encode_ms | 2316.958 | 21408.376 |
| denoise_ms | 19590.432 | 40034.498 |
| vae_decode_ms | 3484.819 | 4650.133 |
| other_ms | 411.446 | 274.275 |

Context creation 60.511824s after hash checks warmed the file cache. Whole-capture sampled RAM peak 5.995117GiB, swap 0.318359–0.363281GiB, 151samples; includes model loading and both generations.

Callback, settings, phase, aggregate and image-hash validation passed. Conditioning hits per generation: [0, 0]; actual transformer passes: [4, 4]. Both saved images are512×512RGB. Reference image equality is recorded in validation; no formal quality eligibility or held-out evaluation.

EasyCache parameters are threshold0.2,start0.15,end0.95. Enabled for both generations, skipped steps[0, 0]. Thus this attempt demonstrates enabled execution without step reuse. It does not establish sustained performance.

[Paired image diagnostic](../results/runs/jetson-flux-klein-006__attempt__20261005-104417/quality-diagnostics.json): MSE0,infinite PSNR,LPIPS0 against the saved control image, one development prompt/seed only.

Interpretation: functional validation passed; one measured observation cannot qualify an option under the full-protocol speed-range rule.

Next experiment: unchanged full1+3+10protocol using `configs/jetson-flux-klein-q4-512-easycache.json`.

## Full repeated protocol

[Summary](../results/runs/jetson-flux-klein-006__baseline__20261005-111428/summary.json), [config](../results/runs/jetson-flux-klein-006__baseline__20261005-111428/config.json), [environment](../results/runs/jetson-flux-klein-006__baseline__20261005-111428/environment.json), [status](../results/runs/jetson-flux-klein-006__baseline__20261005-111428/status.json). Completed 15:14:28–15:31:05 UTC. Same pinned workload, engine and snapshot as the attempt above; one first generation, three discarded warm-ups and ten measured generations in one context.

| Metric | Median, ms | Min–max, ms |
|---|---:|---:|
| wall_ms | 68241.927 | 57236.082–68913.162 |
| text_encode_ms | 23499.517 | 14753.634–24047.600 |
| denoise_ms | 39763.747 | 37647.882–40316.180 |
| vae_decode_ms | 4587.891 | 4541.062–4668.941 |
| other_ms | 285.858 | 262.056–306.810 |

First generation 26123.794 ms; context creation 61.468242 s after file-cache-warming hash checks. Whole-capture system RAM peak 5.991211 GiB, swap 0.327148–0.387695 GiB, 939 samples.

[Validation](../results/runs/jetson-flux-klein-006__baseline__20261005-111428/validation.json) passed: all 14 generations completed four transformer passes, zero conditioning hits, and **zero skipped steps despite EasyCache being enabled**. All 14 reported RGB hashes agree; saved first and measured PNG hashes were checked independently. [Image diagnostic](../results/runs/jetson-flux-klein-006__baseline__20261005-111428/quality-diagnostics.json): identical pixels, MSE 0, infinite PSNR and LPIPS 0 against the full control, for one development prompt/seed only. No formal quality eligibility.

Interpretation: at these settings, the four-step run did not reuse any denoising steps. Enabled caching alone therefore does not demonstrate avoided transformer work. The [matched comparison](compare-jetson-distilled-easycache.md) reports performance and selection separately.
