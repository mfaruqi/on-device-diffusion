---
type: experiment-record
id: compare-jetson-base-combined-reuse
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-base-001__repeat__20261005-042014, jetson-flux-klein-base-002__baseline__20261005-024556, jetson-flux-klein-base-005__baseline__20261005-093239, jetson-flux-klein-base-007__baseline__20261005-121955]
updated: 2026-10-05
---

# Compare: Jetson Base individual and combined reuse

## Question

Does combining exact conditioning reuse with EasyCache improve on each separately? Week 2 baseline evidence for RQ1 and preparation for RQ3's joint-policy evaluation ([proposal overview](../wiki/project/overview.md)); this is a fixed engine configuration comparison, not a planner evaluation.

## Runs

| Policy | Config and summary | Record |
|---|---|---|
| No cache | [config](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/config.json), [summary](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/summary.json) | [jetson-flux-klein-base-001](../experiments/jetson-flux-klein-base-001.md) |
| EasyCache | [config](../results/runs/jetson-flux-klein-base-002__baseline__20261005-024556/config.json), [summary](../results/runs/jetson-flux-klein-base-002__baseline__20261005-024556/summary.json) | [jetson-flux-klein-base-002](../experiments/jetson-flux-klein-base-002.md) |
| Conditioning reuse | [config](../results/runs/jetson-flux-klein-base-005__baseline__20261005-093239/config.json), [summary](../results/runs/jetson-flux-klein-base-005__baseline__20261005-093239/summary.json) | [jetson-flux-klein-base-005](../experiments/jetson-flux-klein-base-005.md) |
| Combined | [config](../results/runs/jetson-flux-klein-base-007__baseline__20261005-121955/config.json), [summary](../results/runs/jetson-flux-klein-base-007__baseline__20261005-121955/summary.json) | [jetson-flux-klein-base-007](../experiments/jetson-flux-klein-base-007.md) |

## Matching

| Held fixed | Value |
|---|---|
| Device | Jetson Orin Nano, 25W, four CPU threads |
| Model | Same pinned Base Q4_0 transformer, Qwen3 Q4_K_M encoder and BF16 VAE |
| Workload | Identical prompt and seed 0, 512², 50 Euler/flux2 steps, CFG 4, batch 1 |
| Engine | Same pinned sd.cpp and harness binary |
| Residency | Eager disk-backed segmented CUDA, prefetch enabled, mmap disabled |
| Protocol | One context; 1 first + 3 discarded warm-ups + 10 measured; no profiler |
| Policies | Conditioning capacity 0 or 2; EasyCache off or threshold0.2/start0.15/end0.95 |

Sequential captures leave filesystem-cache, background, swap and thermal history uncontrolled. New validator snapshots explicitly support the labelled options; the inference binary is identical. Ratios are descriptive rather than isolated causal estimates.

## Results

| Policy | Wall median (min–max), ms | Text median, ms | Denoise median, ms | VAE median, ms | Whole-capture RAM peak, GiB |
|---|---:|---:|---:|---:|---:|
| No cache | 328719.445 (313962.972–331504.639) | 24192.235 | 299599.784 | 4598.830 | 6.014648 |
| EasyCache | 184192.778 (181321.881–190636.573) | 22132.515 | 158403.877 | 4454.548 | 5.996094 |
| Conditioning reuse | 279605.293 (278047.652–285551.188) | 13.342 | 276089.382 | 3334.923 | 6.022461 |
| Combined | 141012.347 (140666.258–141232.225) | 13.349 | 137497.409 | 3323.317 | 6.002930 |

| Compared with | Observed median reduction | Reference/combined ratio |
|---|---:|---:|
| No cache | 57.103% | 2.331139 |
| EasyCache | 23.443% | 1.306217 |
| Conditioning reuse | 49.567% | 1.982843 |

## Findings and selection

The combined maximum 141232.225 ms is below each comparator's minimum, passing the predeclared speed-range screen against both individual policies and no-cache. [Receipt](../results/runs/jetson-flux-klein-base-007__baseline__20261005-121955/comparison.json). All generations retain both policies' expected behavior: positive/negative conditioning reused after the first image; 25 of 50 denoising steps skipped, 50 actual CFG transformer passes instead of 100.

The saved combined output matches EasyCache-only pixels. Against no-cache, [diagnostics](../results/runs/jetson-flux-klein-base-007__baseline__20261005-121955/quality-diagnostics.json) remain PSNR 31.204132360 dB and LPIPS 0.008252321; exact conditioning adds no observed pixel change to this approximate output. Formal quality eligibility remains false.

Interpretation: the combination improves observed latency on this repeated-prompt workload. This does not prove a particular I/O/residency mechanism or a planner advantage, and separate median savings must not be assumed additive.

## Limits

One fixed development prompt/seed, no held-out quality test. Cache-hit behavior depends on repeated identical conditioning. Memory is whole-capture sampled RAM, not per-cache allocation or stage-aligned residency. The campaign stopping boundary permits no further combinations.
