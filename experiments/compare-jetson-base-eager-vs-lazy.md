---
type: experiment-record
id: compare-jetson-base-eager-vs-lazy
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-base-001__repeat__20261005-042014, jetson-flux-klein-base-003__baseline__20261005-060028]
updated: 2026-10-05
---

# Compare: Jetson Base eager and lazy disk-backed loading

## Question

Does disabling eager loading reduce generation latency or whole-capture memory on the Base workload? Week 2 image baseline evidence, RQ1; candidate screening for later residency/reuse combinations.

## Runs

| Role | Run and configuration | Record |
|---|---|---|
| Eager no-cache reference | [jetson-flux-klein-base-001__repeat__20261005-042014](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/config.json) | [Base001](jetson-flux-klein-base-001.md) |
| Lazy no-cache variant | [jetson-flux-klein-base-003__baseline__20261005-060028](../results/runs/jetson-flux-klein-base-003__baseline__20261005-060028/config.json) | [Base003](jetson-flux-klein-base-003.md) |

## Matching

| Field | Both runs |
|---|---|
| Device | Jetson Orin Nano,25W, four CPU threads |
| Model and precision | Same pinned Base Q4_0 transformer, Qwen3 Q4_K_M, original VAE |
| Workload | 512×512,50 Euler steps,flux2 scheduler,CFG4,batch1,seed0, same cat/sign prompt |
| Protocol | One first,three warm-ups,ten measured,one model context,no profiler |
| Engine | Same sd.cpp commit and identical unprofiled harness SHA256 |
| Other settings | Disk-backed parameters,segmented CUDA,flash attention,prefetch on; no reuse or tiling |
| Changed setting | eager-load1 → 0 |

Model/workload/protocol/engine and non-eager harness arguments were independently matched ([receipt](../results/runs/jetson-flux-klein-base-003__baseline__20261005-060028/lazy-comparison.json)). Sequential captures leave filesystem-cache, swap and background-state confounds. Comparisons are descriptive/qualitative, not isolated causal effects.

## Results

Sources: [eager summary](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/summary.json), [lazy summary](../results/runs/jetson-flux-klein-base-003__baseline__20261005-060028/summary.json).

| Metric | Eager median (min–max), ms | Lazy median (min–max), ms |
|---|---:|---:|
| wall_ms | 328719.445 (313962.972–331504.639) | 330683.062 (329500.221–331511.407) |
| text_encode_ms | 24192.235 (16156.323–24600.976) | 24536.232 (24197.899–25267.986) |
| denoise_ms | 299599.784 (289221.908–301971.807) | 301083.412 (300215.457–302285.861) |
| vae_decode_ms | 4598.830 (3588.229–4890.237) | 4643.861 (4556.354–4701.574) |
| other_ms | 295.655 (280.565–306.708) | 291.382 (260.296–334.051) |

Lazy-minus-eager median wall difference: 1963.617 ms (+0.597%). Reference/variant ratio: 0.994062.

| Whole-capture metric | Eager | Lazy |
|---|---:|---:|
| Peak sampled system RAM, GiB | 6.014648 | 4.157227 |
| Sampled swap min–max, GiB | 0.302734–0.316406 | 0.309570–0.600586 |
| Context creation, s | 62.844562 | 0.922235 |

Lazy loading defers parameter work into generation, so context creation alone is not an end-to-end speedup. Both captures include load and all generations; lower sampled RAM accompanies more swap in the lazy capture. The RAM difference cannot establish a smaller total working set or absence of memory pressure.

All fourteen lazy images are pixel-identical to the reference. Paired diagnostic: MSE0, infinite PSNR, LPIPS0 ([quality receipt](../results/runs/jetson-flux-klein-base-003__baseline__20261005-060028/quality-diagnostics.json)).

## Findings

The fidelity diagnostic passes, but the pre-registered range rule fails: lazy maximum 331511.407 ms is not below eager minimum 313962.972 ms. Lazy loading does not qualify for speed-selected combinations. Retain it as residency evidence; do not infer a benefit from the earlier two-image attempt.

## Limits

One development prompt/seed and sequential captures; no held-out or formal quality evaluation. Whole-system memory sampling does not isolate model allocations. Requested placement plus preparation logs do not prove continuous residency or physical storage traffic. A memory-oriented selection rule would be a separately declared extension, not a retroactive change to the speed rule.
