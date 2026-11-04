---
type: experiment-record
id: compare-geneval-jetson-matched-20261026
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: []
updated: 2026-10-26
---

# Compare: GenEval for the Jetson PyTorch configurations (Q4_0 GGUF + NF4 encoder, 512²), evaluated on the A100

## Question

Which Jetson PyTorch options keep GenEval correctness, relative to the Jetson's own reference configuration (Q4_0 GGUF transformer, NF4 Qwen3 encoder, 512², distilled and Base)? RQ2. Image quality depends on weights, precision, workload and options rather than on the device, so the Jetson configurations were run on the A100 to make 384-image evaluations feasible.

## Runs

Configurations: `configs/geneval/a100-geneval-torch-flux-klein-*` from [geneval_matched_configs.py](../scripts/geneval_matched_configs.py). Each is a copy of the Jetson config (e.g. [jetson-torch-flux-klein-q4-nf4-512.json](../configs/jetson-torch-flux-klein-q4-nf4-512.json)) with only backend, device label, file paths, stage residency (off) and conditioning reuse (off) changed; residency and reuse only move weights or skip identical encodings, and the Jetson records show them bit-identical ([jetson-torch-flux-klein-004](jetson-torch-flux-klein-004.md), [base-002](jetson-torch-flux-klein-base-002.md)).

- **Weights identical to the Jetson's (sha256):** distilled transformer `d1023499…` (leejet/FLUX.2-klein-4B-GGUF@3b1f5a9d), Base transformer `c3a28545…`, NF4 encoder `eaa2048d…`, re-quantized on Gilbreth by [save_quantized_text_encoder.py](../scripts/save_quantized_text_encoder.py) with bitsandbytes 0.50.2 (the Jetson used 0.48.0.dev0) and byte-identical to the Jetson copy.
- **Software:** torch 2.8.0+cu126, diffusers 0.40.0, transformers 5.16.1, the same stack as the Jetson (env `torch28-geneval` = `torch28-trt` + bitsandbytes). GPU arithmetic differs from the Jetson's, so images are close, not identical.
- **Jobs:** generation Slurm 12019688 (gilbreth-g010) and, for the regenerated SDPA-math run, 12023348 (standby, a100-40gb partition); scoring 12023321–12023327 and 12023349 (standby; A30, A10 and A100 nodes; scoring is GPU-independent). 96 calibration prompts × 4 seeds.
- **Invalid run (kept, excluded):** the first SDPA-math run (`…sdpa-math__geneval-calibration_subset__20261026-004540`, marked `INVALID.json`) was generated before [geneval_generate.py](../scripts/geneval_generate.py) applied `attention_backend`; all 384 images were pixel-identical to the reference. The generator now applies the backend through the runner's own `restrict_attention`, refuses any option it does not apply, and checks that VAE tiling ran; the regenerated run (`…20261026-092207`) has 0 of 384 images identical to the reference.

## Matching

Within each workload: same weights, 512², steps (4 / 50), guidance (1.0 / 4.0), prompts, seeds and software; one labelled option differs. The Jetson's TaylorSeer configuration (cache interval 2 from step 1) is more aggressive than the A100 BF16 configuration's (interval 5 from step 3), so its effect is not the same lever setting as in the [A100 comparisons](compare-geneval-a100-distilled-20261024.md). Method: [GenEval](../wiki/methods/baseline-metrics.md#geneval-formal-image-quality-metric).

## Results

Distilled reference `a100-geneval-torch-flux-klein-q4-nf4-512__geneval-calibration_subset__20261026-002058`: GenEval **0.7734** (single 1.000, two 0.938, counting 0.781, colours 0.797, position 0.609, colour attr. 0.516).

| Option | Run | GenEval | Δ (95% bootstrap) | Discordant (ref-only / variant-only) | McNemar p | Task Δ, points (single / two / counting / colours / position / colour attr.) |
|---|---|---:|---|---|---:|---|
| torch.compile | `a100-geneval-torch-flux-klein-q4-nf4-512-condreuse-compile__geneval-calibration_subset__20261026-002809` | 0.7708 | -0.0026 (-0.0190 … +0.0150) | 5 / 4 | 1.000 | +0.0 / -1.6 / -1.6 / +1.6 / +1.6 / -1.6 |
| TaylorSeer (interval 2 from step 1, order 1) | `a100-geneval-torch-flux-klein-q4-nf4-512-taylorseer__geneval-calibration_subset__20261026-003222` | 0.7240 | -0.0495 (-0.0951 … -0.0040) | 36 / 17 | 0.013 | +0.0 / -6.2 / +0.0 / +0.0 / +1.6 / -25.0 |
| VAE tiling 128 px / 0.25 | `a100-geneval-torch-flux-klein-q4-nf4-512-vaetile__geneval-calibration_subset__20261026-003715` | 0.7708 | -0.0026 (-0.0175 … +0.0126) | 6 / 5 | 1.000 | +0.0 / +0.0 / -1.6 / +0.0 / +3.1 / -3.1 |
| SDPA math backend | `a100-geneval-torch-flux-klein-q4-nf4-512-sdpa-math__geneval-calibration_subset__20261026-092207` | 0.7708 | -0.0026 (-0.0260 … +0.0212) | 9 / 8 | 1.000 | +0.0 / -1.6 / +1.6 / +0.0 / +3.1 / -4.7 |

Base reference `a100-geneval-torch-flux-klein-base-q4-nf4-512__geneval-calibration_subset__20261026-005416`: GenEval **0.7370** (single 1.000, two 0.891, counting 0.750, colours 0.828, position 0.438, colour attr. 0.516).

| Option | Run | GenEval | Δ (95% bootstrap) | Discordant (ref-only / variant-only) | McNemar p | Task Δ, points (single / two / counting / colours / position / colour attr.) |
|---|---|---:|---|---|---:|---|
| torch.compile | `a100-geneval-torch-flux-klein-base-q4-nf4-512-condreuse-compile__geneval-calibration_subset__20261026-024726` | 0.7266 | -0.0104 (-0.0216 … -0.0021) | 5 / 1 | 0.219 | +0.0 / +0.0 / -3.1 / -1.6 / +0.0 / -1.6 |

## Findings

- torch.compile, VAE tiling and the SDPA math backend are GenEval-neutral on the Jetson's distilled configuration (|Δ| ≤ 0.003; at most 17 of 384 images flip).
- The Jetson TaylorSeer configuration costs about 5 points (Δ −0.050, interval excludes zero, McNemar p = 0.013), almost entirely colour attribution (−25 points); it is more aggressive than the A100 configuration that cost about 2 points.
- On Base, torch.compile's point estimate is −1.0 point with an interval that just excludes zero, but only 6 images flip (5 / 1; McNemar p = 0.22); treat as unresolved at 96 prompts.

## Limits

- 96-prompt calibration subset; the full calibration half would narrow the Base compile and the colour-attribution estimates.
- Evaluated on the A100 with the Jetson's weights and software; Jetson GPU arithmetic is not reproduced bit-for-bit.
- GenEval does not score text rendering.
