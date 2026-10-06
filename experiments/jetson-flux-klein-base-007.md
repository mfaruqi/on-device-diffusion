---
type: experiment-record
id: jetson-flux-klein-base-007
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-base-007__baseline__20261005-121955, jetson-flux-klein-base-007__attempt__20261005-120925]
updated: 2026-10-05
---

# jetson-flux-klein-base-007: EasyCache with exact conditioning reuse

Status: **complete**; correctness gate and full repeated protocol independently validated.

## Question

Can the two individually qualified reuse policies execute together? Week 2 image baseline work, RQ1 and preparation for RQ3 joint-policy evaluation ([proposal overview](../wiki/project/overview.md)). The added change relative to the EasyCache-only configuration is conditioning cache capacity 2; the combination is explicitly labelled.

## Setup

[Config](../results/runs/jetson-flux-klein-base-007__attempt__20261005-120925/config.json), [environment](../results/runs/jetson-flux-klein-base-007__attempt__20261005-120925/environment.json), [status](../results/runs/jetson-flux-klein-base-007__attempt__20261005-120925/status.json). Jetson Orin Nano, 25W, four CPU threads; pinned Base Q4_0 transformer, Qwen3 Q4_K_M and BF16 VAE; 512², 50 Euler/flux2 steps, CFG 4, seed 0, batch 1 and the fixed development prompt. Eager disk-backed segmented CUDA execution, prefetch enabled, mmap off. EasyCache threshold 0.2, start 0.15, end 0.95, plus exact positive/negative conditioning capacity 2. No precision, checkpoint or residency fallback.

Pinned sd.cpp19bbbca and unchanged cache-capable harness. Isolated campaign snapshot e614bb0cf8041b65cf4e327d70d5cb1079f3c33e. First generation plus one measured observation, no discarded warm-ups; one model context. Completed 16:09:25–16:16:40 UTC. [Metric definitions](../wiki/methods/baseline-metrics.md).

## Results

[Summary](../results/runs/jetson-flux-klein-base-007__attempt__20261005-120925/summary.json).

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
| wall_ms | 152333.410 | 160576.708 |
| text_encode_ms | 2869.596 | 18.672 |
| denoise_ms | 144310.147 | 157115.287 |
| vae_decode_ms | 4687.222 | 3282.724 |
| other_ms | 466.445 | 160.025 |

Context creation 55.527124 s after file-cache-warming hash checks. Whole-capture RAM peak 6.000000 GiB, swap 0.342773–0.348633 GiB, 366 samples.

## Validation and quality

[Independent execution validation](../results/runs/jetson-flux-klein-base-007__attempt__20261005-120925/validation.json) and [combination gate](../results/runs/jetson-flux-klein-base-007__attempt__20261005-120925/combination-gate.json) passed. Conditioning hits [0,2]; EasyCache enabled in both generations and skipped [25,25] steps. Each image executed 50 actual transformer passes (two CFG passes per non-skipped step), with 50 scheduler progress steps. Both saved PNGs are 512×512; hashes match the generated rows. Informal inspection shows a coherent cat with a legible hello world sign.

[Paired diagnostic](../results/runs/jetson-flux-klein-base-007__attempt__20261005-120925/quality-diagnostics.json) against the no-cache Base reference: PSNR 31.204132360 dB, LPIPS 0.008252321. The measured RGB pixels are identical to the EasyCache-only output in the [gate receipt](../results/runs/jetson-flux-klein-base-007__attempt__20261005-120925/combination-gate.json). One development prompt/seed; approximate reuse remains approximate, with no formal quality eligibility.

## Interpretation

Exact conditioning reuse and approximate denoising reuse both activate in this configuration. The cached text stage measures retrieval/setup after the first generation. The smoke observation alone does not establish sustained performance; the completed full protocol below supplies repeated timing.

## Next experiment

None scheduled for this configuration.

## Full repeated protocol

[Summary](../results/runs/jetson-flux-klein-base-007__baseline__20261005-121955/summary.json), [config](../results/runs/jetson-flux-klein-base-007__baseline__20261005-121955/config.json), [environment](../results/runs/jetson-flux-klein-base-007__baseline__20261005-121955/environment.json), [status](../results/runs/jetson-flux-klein-base-007__baseline__20261005-121955/status.json). Completed 16:19:55–16:55:35 UTC in the same isolated snapshot and with the same policies as the smoke test. One context; one first generation, three discarded warm-ups, ten measured generations.

| Metric | Median, ms | Min–max, ms |
|---|---:|---:|
| wall_ms | 141012.347 | 140666.258–141232.225 |
| text_encode_ms | 13.349 | 13.275–13.493 |
| denoise_ms | 137497.409 | 137150.352–137719.217 |
| vae_decode_ms | 3323.317 | 3316.910–3345.552 |
| other_ms | 178.887 | 159.479–182.290 |

First generation 151398.382 ms; context creation 59.406272 s after file-cache-warming hash checks. Whole-capture RAM peak 6.002930 GiB, swap 0.339844–0.358398 GiB, 2075 samples.

[Validation](../results/runs/jetson-flux-klein-base-007__baseline__20261005-121955/validation.json) passed: 50 scheduler steps each, 25 skipped denoising steps and 50 actual CFG transformer passes per image; conditioning hits 0 then 2 for all later generations. All 14 reported RGB hashes agree. [Paired diagnostic](../results/runs/jetson-flux-klein-base-007__baseline__20261005-121955/quality-diagnostics.json): PSNR 31.204132360 dB and LPIPS 0.008252321 against no-cache. The retained measured image matches the EasyCache-only image; approximate execution has not become exact relative to no-cache. No formal quality eligibility.

[Matched comparison](compare-jetson-base-combined-reuse.md) evaluates the combination against the reference and both individual policies. This fixed-prompt protocol measures exact conditioning reuse across repeated images, not fresh prompts.
