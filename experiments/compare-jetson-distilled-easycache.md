---
type: experiment-record
id: compare-jetson-distilled-easycache
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-003__baseline__20261005-105911, jetson-flux-klein-006__baseline__20261005-111428]
updated: 2026-10-05
---

# Compare: Jetson four-step klein EasyCache

## Question

Does EasyCache help the original four-step workload? Week 2 image baselines; RQ1 asks “Which diffusion optimization choices transfer across devices and workloads?” ([proposal overview](../wiki/project/overview.md)).

## Runs

| Role | Config and summary | Record |
|---|---|---|
| No-cache control | [config](../results/runs/jetson-flux-klein-003__baseline__20261005-105911/config.json), [summary](../results/runs/jetson-flux-klein-003__baseline__20261005-105911/summary.json) | [003](jetson-flux-klein-003.md) |
| EasyCache | [config](../results/runs/jetson-flux-klein-006__baseline__20261005-111428/config.json), [summary](../results/runs/jetson-flux-klein-006__baseline__20261005-111428/summary.json) | [006](jetson-flux-klein-006.md) |

## Matching

| Held fixed | Value |
|---|---|
| Device | Jetson Orin Nano, 25W, four CPU threads |
| Model | Same pinned distilled klein Q4_0 transformer, Qwen3 Q4_K_M and BF16 VAE |
| Workload | Same prompt, seed 0, 512², four Euler/flux2 steps, CFG 1, batch 1 |
| Engine | Same sd.cpp commit and harness binary; snapshot 31f3be71d07adda2fb122de0ef75a056c27ec7b1 |
| Execution | Eager disk-backed segmented CUDA, prefetch on, mmap off, conditioning cache 0 |
| Protocol | One context; 1 first + 3 discarded warm-ups + 10 measured; no profiler |
| Changed policy | EasyCache enabled with threshold 0.2, start 0.15, end 0.95 |

[Matching and selection receipt](../results/runs/jetson-flux-klein-006__baseline__20261005-111428/comparison.json). Sequential runs do not control filesystem-cache, swap, background or thermal history. Both hash checks warmed files. Initial noise uses the same engine and seed, not a cross-engine RNG assumption.

## Results

| Metric | No-cache median (range), ms | EasyCache median (range), ms |
|---|---:|---:|
| wall_ms | 60935.979 (46863.060–70348.506) | 68241.927 (57236.082–68913.162) |
| text_encode_ms | 22139.018 (12379.430–24546.671) | 23499.517 (14753.634–24047.600) |
| denoise_ms | 36899.651 (30073.790–41110.784) | 39763.747 (37647.882–40316.180) |
| vae_decode_ms | 4563.792 (3610.729–4658.949) | 4587.891 (4541.062–4668.941) |
| other_ms | 289.341 (264.250–311.416) | 285.858 (262.056–306.810) |

Observed median difference: +7305.948 ms, or +11.990%; variant/reference ratio 1.119895. Whole-capture sampled system RAM peaks: 5.997070 GiB control and 5.991211 GiB EasyCache; these are not cache-allocation estimates.

## Findings and selection

EasyCache was enabled in all 14 generations but skipped zero steps throughout; every generation still executed all four transformer passes. Saved output pixels match the full no-cache control, with MSE 0 and LPIPS 0. Observed median latency was higher, and the measured ranges overlap. The declared speed rule fails: variant maximum 68913.162 ms is not below reference minimum 46863.060 ms. This four-step option does not qualify for a speed-selected combination.

Interpretation: the configuration provided no observed transformer-call reduction. This comparison does not isolate cache-management overhead from the sequential-run confounds. It does not establish that every cache threshold, prompt or schedule would behave identically.

## Limits

One fixed development prompt/seed, no held-out set or formal quality approval. Retained first/measured images and per-generation hashes provide repeatability evidence; all raw generation images are not retained. No physical-I/O or SM-utilization attribution is made.
