---
type: experiment-record
id: compare-jetson-eager-vs-lazy-flux-klein
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-003__baseline__20260930-193208, jetson-flux-klein-005__baseline__20261005-001522]
updated: 2026-10-05
---

# Compare: eager and lazy disk-backed loading on Jetson

## Question

How do recorded generation latency and system RAM differ when eager loading is disabled? This supports Week 1–2 image baselines and RQ1. This is a **qualitative comparison**, because the captures were not interleaved and the harness builds differ.

## Runs

| Role | Run | Config | Record |
|---|---|---|---|
| Eager reference | [jetson-flux-klein-003__baseline__20260930-193208](../results/runs/jetson-flux-klein-003__baseline__20260930-193208/) | [saved config](../results/runs/jetson-flux-klein-003__baseline__20260930-193208/config.json) | [jetson-flux-klein-003](jetson-flux-klein-003.md) |
| Lazy variant | [jetson-flux-klein-005__baseline__20261005-001522](../results/runs/jetson-flux-klein-005__baseline__20261005-001522/) | [saved config](../results/runs/jetson-flux-klein-005__baseline__20261005-001522/config.json) | [jetson-flux-klein-005](jetson-flux-klein-005.md) |

## Matching

| Setting | Both runs |
|---|---|
| Hardware | Same Jetson Orin Nano, 25W |
| Engine | sd.cpp commit 19bbbca1c736bbb9538679fc0ae690cb2b46b492 |
| Weights | Same pinned transformer Q4_0, Qwen3 Q4_K_M and original VAE; component SHA256 pins identical |
| Workload | 512×512, four Euler steps, flux2 schedule, guidance 1, batch 1, seed 0, identical cat/sign prompt |
| Protocol | First separately, three discarded warm-ups, ten measured; one-second system RAM sampling |
| Residency | Fixed disk parameters, segmentation allowed, prefetch enabled, no conditioning or step reuse |
| Intended change | eager_load true → false |

Confounds: September 30 versus October 5 captures; filesystem-cache, swap and background-process state were not controlled as paired conditions. The recorded harness binary hashes differ after instrumentation/option work. There is no matched no-option repeat on the newer binary in this comparison. Hashing warmed model files before each run, but does not guarantee identical subsequent residency. Neither capture measures physical storage traffic directly.

## Results

Medians and min–max come directly from each linked summary. Delta is lazy minus eager; ratio is lazy/eager, descriptive rather than a causal speedup estimate.

| Metric | Eager median [min–max], ms | Lazy median [min–max], ms | Delta, ms | Ratio |
|---|---:|---:|---:|---:|
| wall_ms | 66117.050 [54829.892–69043.193] | 60922.093 [56826.812–66477.466] | -5194.957 | 0.921 |
| text_encode_ms | 23109.280 [14616.058–24467.919] | 21210.954 [15096.791–22082.510] | -1898.326 | 0.918 |
| denoise_ms | 38773.048 [31372.669–40612.817] | 35970.152 [34026.075–40995.305] | -2802.896 | 0.928 |
| denoise_step_0_ms | 30621.590 [23244.624–32487.371] | 27816.947 [25891.699–32827.226] | -2804.643 | 0.908 |
| denoise_step_1_ms | 2714.191 [2702.933–2722.158] | 2707.348 [2700.039–2724.822] | -6.843 | 0.997 |
| denoise_step_2_ms | 2708.292 [2701.621–2719.159] | 2711.744 [2704.471–2726.776] | 3.452 | 1.001 |
| denoise_step_3_ms | 2715.521 [2705.656–2726.209] | 2717.666 [2709.590–2727.728] | 2.145 | 1.001 |
| vae_decode_ms | 4547.142 [3609.804–4644.565] | 4578.507 [3822.426–4802.677] | 31.365 | 1.007 |

| Additional metric | Eager | Lazy |
|---|---:|---:|
| First generation, ms | 25785.179 | 54054.979 |
| Model-context creation, s | 58.069 | 0.867 |
| Whole-capture system RAM peak, GiB | 6.078 | 4.186 |

RAM is the same whole-system, whole-capture one-second metric in both runs, including model loading. It is not per-stage allocation or a component-residency measurement. Context-creation time alone is not end-to-end cold start: lazy loading can defer work into generation.

## Findings

- The recorded lazy median is lower, but measured generation ranges overlap. It does not meet the campaign's non-overlapping-range selection rule; do not promote it into a speed-selected combination on this evidence.
- Both runs' later denoising steps have medians near 2.7 seconds; the first step is much longer. These host callback durations include on-demand loading, not just GPU work.
- All generation-row RGB hashes match within and across both runs. This supports exact output agreement for this prompt and seed, not a general quality guarantee.
- The lazy run records a lower whole-capture system RAM peak. Loading policy may contribute, but the confounds prevent a causal attribution.

## Limits

No matched repeat on the new binary, randomized ordering, physical I/O measurement, or multi-prompt quality evaluation. This comparison does not establish a generally faster policy or formal quality eligibility. A paired eager/lazy repeat is needed before using this difference to select a combined policy.
