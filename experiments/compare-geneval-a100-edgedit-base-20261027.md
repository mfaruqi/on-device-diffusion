---
type: experiment-record
id: compare-geneval-a100-edgedit-base-20261027
status: complete
device: a100-pcie-40gb
engine: edge-dit-cpp
runs: []
updated: 2026-10-27
---

# Compare: GenEval for A100 edge-dit.cpp Base step caches (GGUF Q4_0, 1024², 50 steps, CFG 4)

## Question

Do edge-dit.cpp's step caches at engine defaults (TaylorSeer, DiCache, EasyCache, MagCache) change GenEval correctness relative to the edge-dit Base reference ([a100-edgedit-flux-klein-base-001](a100-edgedit-flux-klein-base-001.md))? RQ2.

## Runs

[geneval_generate_edgedit.py](../scripts/geneval_generate_edgedit.py) reran each benchmark run's exact engine command (`command.json`) with only the prompt file, output folder, repeat count and seed base changed: ed-sample gives image n seed n, each prompt is written four times, so prompt i uses seeds 4i…4i+3 in every configuration (edge-dit RNG; not comparable with PyTorch seeds). Each image's engine metadata (prompt, seed, index) was checked. Generation Slurm 12021915 (gilbreth-g010, lab A100-40GB, 16 h 57 min); scoring 12024260/61 and 12027742–44 (standby). Reference `a100-edgedit-flux-klein-base-001__geneval-calibration_subset__20261026-045452`; variants `a100-edgedit-flux-klein-base-005__geneval-calibration_subset__20261026-101028`, `a100-edgedit-flux-klein-base-008__geneval-calibration_subset__20261026-133147`, `a100-edgedit-flux-klein-base-002__geneval-calibration_subset__20261026-161831`, `a100-edgedit-flux-klein-base-007__geneval-calibration_subset__20261026-200541`. Scores and receipts in `results/geneval/<run>/`.

## Matching

Same GGUF Q4_0 Base transformer, Q4_K_M encoder, BF16 VAE, edge-dit commit and binary, 1024², 50 steps, guidance and CFG 4.0, prompts and seeds; one cache flag differs. Compressed-weight workload, separate from the BF16 PyTorch comparisons. Method: [GenEval](../wiki/methods/baseline-metrics.md#geneval-formal-image-quality-metric).

## Results

Reference GenEval **0.7318** (single 0.984, two 0.828, counting 0.766, colours 0.828, position 0.469, colour attr. 0.516). Passes reused and wall change from the benchmark records.

| Cache | Passes reused | Wall change | GenEval | Δ (95% bootstrap) | Discordant (ref-only / variant-only) | McNemar p | Task Δ, points (single / two / counting / colours / position / colour attr.) |
|---|---|---:|---:|---|---|---:|---|
| [TaylorSeer](a100-edgedit-flux-klein-base-005.md) | 38/100 | −37.0% | 0.7318 | +0.0000 (-0.0146 … +0.0146) | 4 / 4 | 1.000 | +0.0 / -1.6 / +3.1 / +0.0 / -1.6 / +0.0 |
| [DiCache](a100-edgedit-flux-klein-base-008.md) | 54/100 | −49.1% | 0.7266 | -0.0052 (-0.0202 … +0.0094) | 5 / 3 | 0.727 | +0.0 / -1.6 / -1.6 / +1.6 / -1.6 / +0.0 |
| [EasyCache](a100-edgedit-flux-klein-base-002.md) | 29/100 | −28.3% | 0.7318 | +0.0000 (-0.0311 … +0.0300) | 16 / 16 | 1.000 | -1.6 / +0.0 / +0.0 / -1.6 / -4.7 / +7.8 |
| [MagCache](a100-edgedit-flux-klein-base-007.md) | 70/100 | −68.2% | 0.7240 | -0.0078 (-0.0360 … +0.0194) | 16 / 13 | 0.711 | +0.0 / +0.0 / -3.1 / +1.6 / -4.7 / +1.6 |

## Findings

- No edge-dit Base cache shows a GenEval change distinguishable from zero at this sample size; TaylorSeer (8 discordant images) and DiCache (8) barely change correctness, EasyCache and MagCache flip more images (32 and 29) in both directions.
- MagCache removes 68% of wall time with a GenEval point estimate of −0.8 points, although its benchmark image shows a deformed sign ([base-007](a100-edgedit-flux-klein-base-007.md)): GenEval does not test text rendering, so this result does not clear MagCache for text prompts.
- edge-dit's TaylorSeer (feature-level, 38/100 passes reused) shows no GenEval cost here, unlike the diffusers TaylorSeer configurations ([A100 Base](compare-geneval-a100-base-20261025.md)); the implementations and reuse settings differ, so this is not a contradiction about one method.

## Limits

- 96-prompt calibration subset; intervals of ±1.5–3.5 points. EasyCache and MagCache, with the most flips, would need the full calibration half to bound a 2-point degradation.
- GenEval omits text rendering; the MagCache sign deformation is outside it.
- A100 compressed-weight configuration only; Jetson edge-dit (Q8 embedding, 512², tiling) not covered.
