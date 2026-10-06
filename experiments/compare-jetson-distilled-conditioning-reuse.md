---
type: experiment-record
id: compare-jetson-distilled-conditioning-reuse
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-003__baseline__20261005-105911, jetson-flux-klein-007__baseline__20261005-113106]
updated: 2026-10-05
---

# Compare: Jetson four-step exact conditioning reuse

## Question

Does retaining identical prompt conditioning improve repeated four-step generation? Week 2 image baselines, RQ1 ([proposal overview](../wiki/project/overview.md)).

## Runs

| Role | Config and summary | Record |
|---|---|---|
| No-cache control | [config](../results/runs/jetson-flux-klein-003__baseline__20261005-105911/config.json), [summary](../results/runs/jetson-flux-klein-003__baseline__20261005-105911/summary.json) | [003](jetson-flux-klein-003.md) |
| Conditioning reuse | [config](../results/runs/jetson-flux-klein-007__baseline__20261005-113106/config.json), [summary](../results/runs/jetson-flux-klein-007__baseline__20261005-113106/summary.json) | [007](jetson-flux-klein-007.md) |

## Matching

| Held fixed | Value |
|---|---|
| Device | Jetson Orin Nano, 25W, four CPU threads |
| Model | Same pinned distilled klein Q4_0 transformer, Qwen3 Q4_K_M and BF16 VAE |
| Workload | Same prompt, seed 0, 512², four Euler/flux2 steps, CFG 1, batch 1 |
| Engine | Same pinned sd.cpp and harness binary, snapshot 31f3be71d07adda2fb122de0ef75a056c27ec7b1 |
| Execution | Eager disk-backed segmented CUDA, prefetch on, mmap off, approximate step cache off |
| Protocol | One context; 1 first + 3 discarded warm-ups + 10 measured; no profiler |
| Changed policy | Conditioning cache capacity 0→1 |

[Matching receipt](../results/runs/jetson-flux-klein-007__baseline__20261005-113106/comparison.json). Sequential captures leave filesystem-cache, swap, background and thermal history uncontrolled; observed differences are descriptive, not isolated causal estimates.

## Results

| Metric | No-cache median (range), ms | Conditioning reuse median (range), ms |
|---|---:|---:|
| wall_ms | 60935.979 (46863.060–70348.506) | 15734.541 (15589.200–20335.525) |
| text_encode_ms | 22139.018 (12379.430–24546.671) | 11.550 (11.239–14.953) |
| denoise_ms | 36899.651 (30073.790–41110.784) | 12186.432 (12096.185–16846.371) |
| vae_decode_ms | 4563.792 (3610.729–4658.949) | 3336.363 (3313.352–3859.701) |
| other_ms | 289.341 (264.250–311.416) | 180.117 (158.056–183.012) |

Observed median reduction 45201.438 ms (74.179%); reference/variant ratio 3.872752. Whole-capture sampled system RAM peak 5.997070→6.010742 GiB; these are not per-cache allocation estimates.

## Findings and selection

The first generation computes conditioning, then all 13 later generations report one hit. All four transformer passes remain; there is no approximate denoising reuse. Measured text-stage time becomes retrieval/setup time. Denoising time also decreases, so the end-to-end difference cannot be explained solely by subtracting encoder time. A change in loading/residency behavior is a hypothesis, not directly measured evidence here.

Saved pixels match the no-cache reference, with MSE 0 and LPIPS 0. Variant maximum 20335.525 ms is below reference minimum 46863.060 ms, so the predeclared speed and exact-fidelity diagnostic screens pass. No additional four-step combination is scheduled under the campaign stopping boundary.

## Limits

One repeated development prompt/seed, not diverse prompts or cache eviction. The result measures reuse across repeated images with identical conditioning, not general per-image latency for new prompts. No formal quality or held-out evaluation. First/measured PNGs and all generation hashes are retained; physical I/O and continuous parameter residency were not measured.
