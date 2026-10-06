---
type: experiment-record
id: compare-jetson-distilled-prefetch
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-003__baseline__20261005-105911, jetson-flux-klein-008__baseline__20261005-113721]
updated: 2026-10-05
---

# Compare: Jetson four-step prefetch enabled versus disabled

## Question

Does disabling engine prefetch improve repeated four-step generation? Week 2 image baselines, RQ1 ([proposal overview](../wiki/project/overview.md)).

## Runs

| Role | Config and summary | Record |
|---|---|---|
| Prefetch-enabled control | [config](../results/runs/jetson-flux-klein-003__baseline__20261005-105911/config.json), [summary](../results/runs/jetson-flux-klein-003__baseline__20261005-105911/summary.json) | [003](jetson-flux-klein-003.md) |
| Prefetch disabled | [config](../results/runs/jetson-flux-klein-008__baseline__20261005-113721/config.json), [summary](../results/runs/jetson-flux-klein-008__baseline__20261005-113721/summary.json) | [008](jetson-flux-klein-008.md) |

## Matching

| Held fixed | Value |
|---|---|
| Device | Jetson Orin Nano, 25W, four CPU threads |
| Model | Same pinned distilled klein Q4_0 transformer, Qwen3 Q4_K_M and BF16 VAE |
| Workload | Same prompt, seed 0, 512², four Euler/flux2 steps, CFG 1, batch 1 |
| Engine | Same pinned sd.cpp and harness binary; snapshot 31f3be71d07adda2fb122de0ef75a056c27ec7b1 |
| Execution | Eager disk-backed segmented CUDA, mmap off, conditioning and approximate step caching off |
| Protocol | One context; 1 first + 3 discarded warm-ups + 10 measured; no profiler |
| Changed policy | Engine prefetch enabled→disabled |

[Matching receipt](../results/runs/jetson-flux-klein-008__baseline__20261005-113721/comparison.json). Sequential captures leave filesystem-cache, swap, background and thermal history uncontrolled; the observed comparison is descriptive, not an isolated causal estimate.

## Results

| Metric | Enabled median (range), ms | Disabled median (range), ms |
|---|---:|---:|
| wall_ms | 60935.979 (46863.060–70348.506) | 60503.164 (50312.039–65011.058) |
| text_encode_ms | 22139.018 (12379.430–24546.671) | 21274.854 (13631.980–23862.450) |
| denoise_ms | 36899.651 (30073.790–41110.784) | 33938.654 (30341.032–40776.559) |
| vae_decode_ms | 4563.792 (3610.729–4658.949) | 4488.260 (4275.987–4802.754) |
| other_ms | 289.341 (264.250–311.416) | 294.502 (275.104–310.610) |

Observed median difference -432.815 ms (-0.710%); variant/reference ratio 0.992897. Whole-capture sampled system RAM peak 5.997070→5.997070 GiB. These peaks include loading and are not per-stage memory estimates.

## Findings and selection

The saved images are identical, with MSE 0 and LPIPS 0, and all four transformer passes remain. The timing ranges overlap. Variant maximum 65011.058 ms is not below reference minimum 46863.060 ms, so the declared speed-selection rule fails. No speed-selected combination is justified by this result.

Interpretation: the small observed median difference does not establish a sustained latency improvement under this protocol. These measurements do not identify physical I/O overlap or prefetch overhead.

## Limits

One fixed development prompt/seed and sequential captures. No formal quality or held-out evaluation. System memory samples are not stage-aligned. First/measured PNGs and all generation hashes are retained; other raw images were deleted after hashing.
