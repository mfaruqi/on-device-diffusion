---
type: finding
summary: With disk-backed parameters on Jetson, sd.cpp reloads and releases each component every generation; engine-reported loading is most of a 4-step image.
status: supported
confidence: high
rq: [RQ1, RQ3]
sources: [../../experiments/jetson-flux-klein-003.md]
updated: 2026-10-06
---

# Weight reloading dominates disk-backed Jetson generations

**Claim.** In the repeated 4-step baseline, engine-reported tensor loading sums to a median of **49.2 s**
([`measured_engine_loading_median_s`](../../results/runs/jetson-flux-klein-003__baseline__20260930-193208/baseline-review.json))
per measured generation, out of a **66117.0 ms**
([`measured.wall_ms.median`](../../results/runs/jetson-flux-klein-003__baseline__20260930-193208/summary.json)) generation.

**Evidence.** [Record](../../experiments/jetson-flux-klein-003.md#observation-repeated-component-loading-dominates-the-slow-generations):
the text encoder, transformer and VAE are prepared before use and released after it in every generation;
denoising steps after the first stay near 2.7 s. These are engine-reported intervals, not measured
physical storage I/O.

**Interpretation.** On this device, for few-step workloads, residency decisions matter more than kernel
speed. For the 50-step Base workload the same reloading is a much smaller share of the generation.

Related: [stage residency](../concepts/stage-residency.md), [Jetson](../systems/jetson-orin-nano.md).
