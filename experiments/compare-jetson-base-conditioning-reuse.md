---
type: experiment-record
id: compare-jetson-base-conditioning-reuse
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-base-001__repeat__20261005-042014, jetson-flux-klein-base-005__baseline__20261005-093239]
updated: 2026-10-05
---

# Compare: Jetson Base exact conditioning reuse

## Question

Does retaining fixed prompt conditioning reduce repeated generation latency? Week2 image baselines,RQ1; candidate selection for combined reuse policies.

## Runs and matching

| Role | Config and summary | Record |
|---|---|---|
| No-cache reference | [config](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/config.json),[summary](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/summary.json) | [Base001](jetson-flux-klein-base-001.md) |
| Exact conditioning reuse | [config](../results/runs/jetson-flux-klein-base-005__baseline__20261005-093239/config.json),[summary](../results/runs/jetson-flux-klein-base-005__baseline__20261005-093239/summary.json) | [Base005](jetson-flux-klein-base-005.md) |

| Held fixed | Value |
|---|---|
| Device/engine | Jetson Orin Nano25W,four CPU threads,same pinned sd.cpp and harness SHA256 |
| Model/workload | Identical Base Q4_0,Qwen3 Q4_K_M,BF16VAE pins;512²,50Euler/flux2steps,CFG4,seed0,batch1,same prompt |
| Protocol | One first,three warm-ups,ten measured; one context,one-second tegrastats,no profiler |
| Residency | Fixed eager disk-backed segmented CUDA,prefetch enabled,flash attention,no approximate step cache |
| Changed option | Conditioning cache capacity0→2 |

[Matching receipt](../results/runs/jetson-flux-klein-base-005__baseline__20261005-093239/conditioning-comparison.json). Sequential captures leave filesystem-cache,background,swap and thermal history uncontrolled; observed ratios are descriptive, not isolated causal effects.

## Results

| Metric | No-cache median (range), ms | Reuse median (range), ms |
|---|---:|---:|
| wall_ms | 328719.445 (313962.972–331504.639) | 279605.293 (278047.652–285551.188) |
| text_encode_ms | 24192.235 (16156.323–24600.976) | 13.342 (13.032–15.467) |
| denoise_ms | 299599.784 (289221.908–301971.807) | 276089.382 (274535.797–282047.868) |
| vae_decode_ms | 4598.830 (3588.229–4890.237) | 3334.923 (3310.370–4295.152) |
| other_ms | 295.655 (280.565–306.708) | 177.985 (156.929–179.120) |

Observed median reduction 14.941%; reference/variant ratio 1.175655. Whole-capture sampled RAM 6.014648→6.022461GiB; no isolated cache-allocation estimate. Cached text stage includes retrieval/setup. Denoising also changed despite unchanged transformer call count; this comparison cannot attribute that difference to a specific residency or I/O mechanism.

## Findings and selection

All14images match reference pixels; [diagnostics](../results/runs/jetson-flux-klein-base-005__baseline__20261005-093239/quality-diagnostics.json) give MSE0,LPIPS0. The exact-option screen passes. Variant maximum 285551.188ms is below reference minimum 313962.972ms, so the predeclared speed rule also passes. This option qualifies for a separately labelled combination with the independently qualified EasyCache policy. No combination result exists yet.

## Limits

One fixed development prompt/seed; prompt diversity,eviction behavior and formal/held-out quality are untested. No physical-I/O or continuous-residency measurement here. Saved exact pixels establish equality for this workload only.
