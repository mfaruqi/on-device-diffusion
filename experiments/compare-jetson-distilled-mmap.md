---
type: experiment-record
id: compare-jetson-distilled-mmap
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-003__baseline__20261005-105911, jetson-flux-klein-009__baseline__20261005-115227]
updated: 2026-10-05
---

# Compare: Jetson four-step ordinary weight-file reads versus mmap

## Question

Does enabling weight-file mmap improve repeated four-step generation? Week 2 image baselines, RQ1 ([proposal overview](../wiki/project/overview.md)).

## Runs

| Role | Config and summary | Record |
|---|---|---|
| Ordinary-read control | [config](../results/runs/jetson-flux-klein-003__baseline__20261005-105911/config.json), [summary](../results/runs/jetson-flux-klein-003__baseline__20261005-105911/summary.json) | [003](jetson-flux-klein-003.md) |
| Mmap enabled | [config](../results/runs/jetson-flux-klein-009__baseline__20261005-115227/config.json), [summary](../results/runs/jetson-flux-klein-009__baseline__20261005-115227/summary.json) | [008](jetson-flux-klein-009.md) |

## Matching

| Held fixed | Value |
|---|---|
| Device | Jetson Orin Nano, 25W, four CPU threads |
| Model | Same pinned distilled klein Q4_0 transformer, Qwen3 Q4_K_M and BF16 VAE |
| Workload | Same prompt, seed 0, 512², four Euler/flux2 steps, CFG 1, batch 1 |
| Engine | Same pinned sd.cpp and harness binary; snapshot 31f3be71d07adda2fb122de0ef75a056c27ec7b1 |
| Execution | Eager disk-backed segmented CUDA, prefetch on, conditioning and approximate step caching off |
| Protocol | One context; 1 first + 3 discarded warm-ups + 10 measured; no profiler |
| Changed policy | Weight-file mmap disabled→enabled |

[Matching receipt](../results/runs/jetson-flux-klein-009__baseline__20261005-115227/comparison.json). Sequential captures leave filesystem-cache, swap, background and thermal history uncontrolled; the observed comparison is descriptive, not an isolated causal estimate.

## Results

| Metric | Ordinary reads median (range), ms | Mmap median (range), ms |
|---|---:|---:|
| wall_ms | 60935.979 (46863.060–70348.506) | 67758.838 (66997.369–68557.088) |
| text_encode_ms | 22139.018 (12379.430–24546.671) | 23245.910 (22718.246–23439.282) |
| denoise_ms | 36899.651 (30073.790–41110.784) | 39497.698 (39095.780–40407.290) |
| vae_decode_ms | 4563.792 (3610.729–4658.949) | 4641.656 (4562.725–4678.250) |
| other_ms | 289.341 (264.250–311.416) | 287.717 (258.120–340.494) |

Observed median difference 6822.860 ms (11.197%); variant/reference ratio 1.111968. Whole-capture sampled system RAM peak 5.997070→5.953125 GiB. These peaks include loading and are not per-stage memory estimates.

## Findings and selection

The saved images are identical, with MSE 0 and LPIPS 0, and all four transformer passes remain. The timing ranges overlap. Variant maximum 68557.088 ms is not below reference minimum 46863.060 ms, so the declared speed-selection rule fails. No speed-selected combination is justified by this result.

Interpretation: the observed higher median does not support a latency improvement under this protocol. These measurements do not identify physical I/O or GPU zero-copy residency.

## Limits

One fixed development prompt/seed and sequential captures. No formal quality or held-out evaluation. System memory samples are not stage-aligned. First/measured PNGs and all generation hashes are retained; other raw images were deleted after hashing.
