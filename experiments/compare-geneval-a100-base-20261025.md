---
type: experiment-record
id: compare-geneval-a100-base-20261025
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: []
updated: 2026-10-25
---

# Compare: GenEval (calibration subset) for the A100 Base BF16 reference, torch.compile and TaylorSeer; and against the distilled result

## Question

Do torch.compile and the diffusers TaylorSeer step cache change GenEval correctness on the 50-step CFG Base workload, relative to the A100 torch 2.8 BF16 Base reference? And does each lever's GenEval effect transfer from the 4-step distilled workload ([distilled comparison](compare-geneval-a100-distilled-20261024.md))? RQ2.

## Runs

Scoring outputs and generation receipts in `results/geneval/<run>/`; images on Gilbreth scratch. Slurm 12016165 (generation, gilbreth-g008, A100-PCIE-40GB, torch 2.8.0+cu126; 2-prompt smoke per configuration first) and 12017527 (scoring). 384 images per configuration.

| Run | Configuration | Benchmark record |
|---|---|---|
| `a100-flux-klein-base-torch28-001__geneval-calibration_subset__20261024-205729` | [a100-flux-klein-base-torch28-bf16.json](../configs/a100-flux-klein-base-torch28-bf16.json) | [a100-flux-klein-base-torch28-001](a100-flux-klein-base-torch28-001.md) |
| `a100-flux-klein-base-torch28-004__geneval-calibration_subset__20261025-011144` | [a100-flux-klein-base-torch28-bf16-compile.json](../configs/a100-flux-klein-base-torch28-bf16-compile.json) | [a100-flux-klein-base-torch28-004](a100-flux-klein-base-torch28-004.md) |
| `a100-flux-klein-base-torch28-003__geneval-calibration_subset__20261024-235542` | [a100-flux-klein-base-torch28-bf16-taylorseer.json](../configs/a100-flux-klein-base-torch28-bf16-taylorseer.json) | [a100-flux-klein-base-torch28-003](a100-flux-klein-base-torch28-003.md) |

## Matching

Pinned FLUX.2 klein Base 4B checkpoint, BF16, 1024², 50 steps, guidance 4.0 (empty negative prompt), 512-token text, the same 96 calibration prompts and seeds 0–3 as the distilled comparison, same software; configurations differ only in the labelled option. The distilled runs use a different checkpoint, step count and guidance, so distilled-vs-Base absolute scores are a workload comparison, not a lever effect. Method: [GenEval](../wiki/methods/baseline-metrics.md#geneval-formal-image-quality-metric).

## Results

Base reference GenEval **0.7474**; per task: single object 0.984, two objects 0.953, counting 0.719, colours 0.719, position 0.453, colour attribution 0.656.

| Variant | GenEval | Δ vs reference (95% bootstrap) | Discordant images (ref-only / variant-only correct) | McNemar p | Task Δ, points (single / two / counting / colours / position / colour attr.) |
|---|---:|---|---|---:|---|
| [a100-flux-klein-base-torch28-004](a100-flux-klein-base-torch28-004.md) torch.compile | 0.7422 | -0.0052 (-0.0238 … +0.0129) | 7 / 5 | 0.774 | +0.0 / +0.0 / +0.0 / +4.7 / -1.6 / -6.2 |
| [a100-flux-klein-base-torch28-003](a100-flux-klein-base-torch28-003.md) TaylorSeer (interval 5 from step 3, order 1) | 0.7318 | -0.0156 (-0.0412 … +0.0086) | 15 / 9 | 0.307 | +0.0 / +0.0 / +0.0 / -1.6 / +0.0 / -7.8 |

Lever effects side by side (Δ GenEval vs each workload's own reference; per-image speed from the benchmark records):

| Lever | Distilled Δ (95% CI) | Base Δ (95% CI) | Distilled wall change | Base wall change |
|---|---|---|---|---|
| torch.compile | −0.003 (−0.014 … +0.009) | −0.005 (−0.024 … +0.013) | −21% | −26% |
| TaylorSeer | −0.023 (−0.070 … +0.020) | −0.016 (−0.041 … +0.009) | −34% | −62% |

## Findings

- torch.compile is GenEval-neutral on both workloads (|Δ| ≤ 0.005; 5 and 12 images flip out of 384).
- TaylorSeer's GenEval cost is small and not significant on either workload at this sample size (−1.6 points on Base, −2.3 on distilled), concentrated in colour attribution (−7.8 and −9.4 points). On Base it removes 62% of wall time for that cost; on distilled it removes 34% and visibly changes images.
- The distilled reference scores higher than the Base reference on this subset (0.800 vs 0.747), mainly on counting, colours and position; this is a checkpoint/schedule difference, not a lever effect.

## Limits

- 96 of 277 calibration prompts; intervals of ±1–4 points. A declared maximum degradation needs the full calibration half.
- A100 BF16 at 1024² only; Q4/NF4 (Jetson weights), 512², sd.cpp and edge-dit caches not covered yet.
- GenEval does not score text rendering, so the deformed-sign cases seen with MagCache are outside it.
