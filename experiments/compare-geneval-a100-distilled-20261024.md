---
type: experiment-record
id: compare-geneval-a100-distilled-20261024
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: []
updated: 2026-10-24
---

# Compare: GenEval (calibration subset) for the A100 distilled BF16 reference, torch.compile and TaylorSeer

## Question

Do torch.compile and the diffusers TaylorSeer step cache change prompt-content correctness, measured with the proposal's formal image metric (GenEval), relative to the A100 torch 2.8 BF16 distilled reference? RQ2 (reuse levers under a quality requirement).

## Runs

Image runs live outside `results/runs/` (GenEval layout); scoring outputs and generation receipts are in `results/geneval/<run>/` (`generations.json`, `geneval-results.jsonl`, `geneval-summary.txt`); images on Gilbreth scratch (`/scratch/gilbreth/mfaruqi/geneval/images/`).

| Run | Configuration | Benchmark record |
|---|---|---|
| `a100-flux-klein-torch28-001__geneval-calibration_subset__20261024-192745` | [a100-flux-klein-torch28-bf16.json](../configs/a100-flux-klein-torch28-bf16.json) | [a100-flux-klein-torch28-001](a100-flux-klein-torch28-001.md) |
| `a100-flux-klein-compile-001__geneval-calibration_subset__20261024-194449` | [a100-flux-klein-torch28-bf16-compile.json](../configs/a100-flux-klein-torch28-bf16-compile.json) | [a100-flux-klein-compile-001](a100-flux-klein-compile-001.md) |
| `a100-flux-klein-torch28-004__geneval-calibration_subset__20261024-195927` | [a100-flux-klein-torch28-bf16-taylorseer.json](../configs/a100-flux-klein-torch28-bf16-taylorseer.json) | [a100-flux-klein-torch28-004](a100-flux-klein-torch28-004.md) |

Slurm 12015083 (generation, gilbreth-g000, A100-PCIE-40GB, torch 2.8.0+cu126; each configuration passed a 2-prompt smoke first) and 12015962 (scoring). 384 images per configuration.

## Matching

Same pinned FLUX.2 klein 4B distilled checkpoint, BF16, 1024², 4 steps, guidance 1.0, 512-token text, identical 96 prompts and seeds 0–3, same node and software; the configurations differ only in the labelled option. Method and definitions: [GenEval](../wiki/methods/baseline-metrics.md#geneval-formal-image-quality-metric).

## Results

Reference GenEval **0.7995**; per task: single object 1.000, two objects 0.938, counting 0.828, colours 0.859, position 0.562, colour attribution 0.609.

| Variant | GenEval | Δ vs reference (95% bootstrap) | Discordant images (ref-only / variant-only correct) | McNemar p | Task Δ, points (single / two / counting / colours / position / colour attr.) |
|---|---:|---|---|---:|---|
| [a100-flux-klein-compile-001](a100-flux-klein-compile-001.md) torch.compile | 0.7969 | -0.0026 (-0.0143 … +0.0092) | 3 / 2 | 1.000 | +0.0 / +0.0 / +0.0 / -1.6 / +1.6 / -1.6 |
| [a100-flux-klein-torch28-004](a100-flux-klein-torch28-004.md) TaylorSeer (interval 5 from step 3, order 1) | 0.7760 | -0.0234 (-0.0695 … +0.0202) | 26 / 17 | 0.222 | +0.0 / +1.6 / +0.0 / -7.8 / +1.6 / -9.4 |

## Findings

- torch.compile leaves GenEval unchanged within this subset's resolution: Δ −0.003 with an interval of about ±0.012, and only 5 of 384 images flip.
- TaylorSeer's point estimate is −0.023 (−2.3 points), concentrated in colour tasks (colour attribution −9.4, colours −7.8 points); 43 images flip (26 against, 17 for). With 96 prompts the interval (−0.070 … +0.020) includes zero, so the degradation is suggestive, not established.
- The reference itself is weakest on position (0.562) and colour attribution (0.609), the tasks where small changes are most visible.

## Limits

- Calibration subset only (96 of 277 calibration prompts); intervals of a few points. The full calibration half (277 prompts) would narrow them.
- Distilled workload on the A100 at 1024²; device-specific variants (Q4/NF4 at 512², sd.cpp, edge-dit) are not covered yet.
- GenEval checks object presence, count, colour and position with a detector; it does not measure text rendering or aesthetics.
