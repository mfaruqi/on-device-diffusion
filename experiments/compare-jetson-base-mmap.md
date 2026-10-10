---
type: experiment-record
id: compare-jetson-base-mmap
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-base-001__repeat__20261005-042014, jetson-flux-klein-base-006__baseline__20261005-131024]
updated: 2026-10-05
---

# Compare: Jetson Base ordinary weight-file reads and mmap

## Question

Does enabling memory-mapped weight-file I/O improve the fixed Base workload? Week 2 image-engine baseline evidence for RQ1 ([proposal overview](../wiki/project/overview.md)).

## Runs

| Policy | Config and summary | Record |
|---|---|---|
| Ordinary reads | [config](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/config.json), [summary](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/summary.json) | [Base001](jetson-flux-klein-base-001.md) |
| Mmap | [config](../results/runs/jetson-flux-klein-base-006__baseline__20261005-131024/config.json), [summary](../results/runs/jetson-flux-klein-base-006__baseline__20261005-131024/summary.json) | [Base006](jetson-flux-klein-base-006.md) |

## Matching

| Held fixed | Value |
|---|---|
| Device | Jetson Orin Nano, 25W, four CPU threads |
| Model | Identical pinned Base Q4_0 transformer, Qwen3 Q4_K_M encoder, BF16 VAE |
| Workload | Same prompt, seed 0, 512², 50 Euler/flux2 steps, CFG 4, batch 1 |
| Engine | Same pinned sd.cpp and exact harness binary hash |
| Execution | Eager disk-backed segmented CUDA, prefetch enabled, no denoising or conditioning reuse |
| Protocol | One context; 1 first + 3 discarded warm-ups + 10 measured; no profiler |
| Variable | mmap disabled versus enabled; sole differing harness argument |

Sequential captures leave filesystem-cache, background, swap and thermal history uncontrolled. Ratios describe these captures, not an isolated causal estimate. Hash checks warm file cache before context creation.

## Results

| Metric | Ordinary reads | Mmap |
|---|---:|---:|
| wall_ms, median (min–max) | 328719.445 (313962.972–331504.639) | 329425.282 (328578.649–330363.177) |
| text_encode_ms, median (min–max) | 24192.235 (16156.323–24600.976) | 24306.888 (24049.472–24551.362) |
| denoise_ms, median (min–max) | 299599.784 (289221.908–301971.807) | 300190.007 (299476.021–301087.027) |
| vae_decode_ms, median (min–max) | 4598.830 (3588.229–4890.237) | 4624.989 (4577.930–4683.022) |
| Whole-capture RAM peak, GiB | 6.014648 | 5.993164 |
| Whole-capture swap min–max, GiB | 0.302734–0.316406 | 0.342773–0.961914 |

Mmap's observed median differs by +705.837 ms (+0.215%). Reference/mmap median ratio 0.997857.

## Findings and selection

Latency ranges overlap: mmap's maximum 330363.177 ms is not below reference's minimum 313962.972 ms. The predeclared speed-range screen fails ([receipt](../results/runs/jetson-flux-klein-base-006__baseline__20261005-131024/comparison.json)). No speed-selected mmap combination follows.

Both execute 100 CFG transformer passes per image. Mmap confirms mappings for three component files and retains reference-identical measured RGB pixels; [paired diagnostic](../results/runs/jetson-flux-klein-base-006__baseline__20261005-131024/quality-diagnostics.json) reports PSNR infinity and LPIPS 0. Formal quality eligibility remains false.

Interpretation: this capture does not establish a latency benefit from mmap for the Base workload. The sampled memory values describe the whole system and do not identify which mappings or stages caused swap growth.

## Limits

One fixed development prompt/seed, no held-out quality test. File mappings do not prove physical storage traffic, GPU zero-copy or continuous weight residency. Callback timings include loading and synchronization rather than isolated GPU kernel time. Memory samples include context loading and all generations; no allocator/stage alignment. Sequential-run confounds prevent a causal claim from the small median difference.
