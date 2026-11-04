---
type: experiment-record
id: compare-paired-image-diagnostics-20261024
status: complete
device: [jetson-orin-nano, a100-pcie-40gb]
engine: [stable-diffusion-cpp, edge-dit-cpp, pytorch-diffusers]
runs: [a100-edgedit-flux-klein-002__attempt__20261015-152019, a100-edgedit-flux-klein-007__baseline__20261015-152117, a100-edgedit-flux-klein-base-001__baseline__20261015-165052, a100-edgedit-flux-klein-base-002__baseline__20261015-170356, a100-edgedit-flux-klein-base-003__baseline__20261015-171344, a100-edgedit-flux-klein-base-004__baseline__20261015-172354, a100-edgedit-flux-klein-base-005__baseline__20261015-173406, a100-edgedit-flux-klein-base-006__baseline__20261015-174257, a100-edgedit-flux-klein-base-007__baseline__20261015-175238, a100-edgedit-flux-klein-base-008__baseline__20261015-175727, a100-flux-klein-base-torch28-001__baseline__20261015-155338, a100-flux-klein-base-torch28-002__baseline__20261015-160117, a100-flux-klein-base-torch28-003__baseline__20261015-160821, a100-flux-klein-base-torch28-004__baseline__20261015-161253, a100-flux-klein-compile-001__baseline__20261015-151740, a100-flux-klein-torch28-001__baseline__20261015-151548, a100-flux-klein-torch28-002__baseline__20261015-151840, a100-flux-klein-torch28-004__baseline__20261015-151949, jetson-edgedit-flux-klein-006__baseline__20261014-170149, jetson-edgedit-flux-klein-011__baseline__20261015-133915, jetson-edgedit-flux-klein-013__baseline__20261015-135118, jetson-edgedit-flux-klein-014__baseline__20261015-135841, jetson-edgedit-flux-klein-base-001__baseline__20261015-193452, jetson-edgedit-flux-klein-base-002__baseline__20261015-204021, jetson-edgedit-flux-klein-base-003__baseline__20261015-213038, jetson-edgedit-flux-klein-base-004__baseline__20261015-222249, jetson-edgedit-flux-klein-base-005__baseline__20261015-231521, jetson-edgedit-flux-klein-base-006__baseline__20261016-000110, jetson-edgedit-flux-klein-base-007__baseline__20261016-005119, jetson-flux-klein-003__repeat__20261014-163411, jetson-flux-klein-010__baseline__20261014-183501, jetson-flux-klein-011__baseline__20261014-185639, jetson-torch-flux-klein-001__baseline__20261014-171410, jetson-torch-flux-klein-002__baseline__20261014-174108, jetson-torch-flux-klein-003__baseline__20261014-180843, jetson-torch-flux-klein-004__baseline__20261015-141028, jetson-torch-flux-klein-006__baseline__20261015-142456, jetson-torch-flux-klein-007__baseline__20261015-145234, jetson-torch-flux-klein-base-001__baseline__20261016-050136, jetson-torch-flux-klein-base-002__baseline__20261016-015207, jetson-torch-flux-klein-base-004__baseline__20261016-033850]
updated: 2026-10-24
---

# Compare: paired image diagnostics (PSNR, LPIPS) for pixel-changing options on Jetson and A100

## Question

How far does each pixel-changing option move the saved image away from its own engine's reference configuration, on the same device, prompt and seed? This is a diagnostic for RQ2 (reuse and precision levers under a quality requirement); it does not establish prompt alignment or the formal quality requirement (GenEval, [overview](../wiki/project/overview.md)).

## Runs

32 variant–reference pairs in nine groups (table below), each variant against the reference configuration of the same engine on the same device: one completed full-protocol run per configuration. Image: `images/measured-0.png` (first measured generation; PyTorch, sd.cpp) or `output.png` (final pass; edge-dit). The A100 edge-dit distilled reference PNG is not retained in its run directory; the zero-reuse screen [a100-edgedit-flux-klein-002 smoke](../results/runs/a100-edgedit-flux-klein-002__attempt__20261015-152019/) is used instead, and its RGB hash equals the reference's recorded final hash (`c476fc32…`, [validation.json](../results/runs/a100-edgedit-flux-klein-001__baseline__20261005-233237/validation.json)).

## Matching

Same engine, device, checkpoint files, resolution, steps, guidance, seed and prompt within each pair; the only difference is the labelled option. Pairs are never formed across engines or devices (initial noise and precision differ). Per-pair files: `quality-diagnostics.json` in each variant's run directory; summary: `output/quality-diagnostics-20261024/quality-diagnostics.json` (git-ignored). Method: [paired saved-image diagnostics](../wiki/methods/baseline-metrics.md#paired-saved-image-diagnostics), implemented in [quality_diagnostics.py](../scripts/quality_diagnostics.py) (LPIPS AlexNet v0.1 via `lpips` 0.1.4, torch 2.14.1 CPU; same-image distance 0 in every pair).

## Results

PSNR in dB (higher = closer); LPIPS lower = closer; mean absolute RGB difference in 8-bit levels. Visual notes are copied from the variant's record where one exists.

| Group | Variant | Option | PSNR | LPIPS | Mean abs. diff. | Visual (record) |
|---|---|---|---:|---:|---:|---|
| Jetson sd.cpp distilled | [jetson-flux-klein-010](jetson-flux-klein-010.md) | flash attention off | 35.69 | 0.0028 | 1.26 | – |
| Jetson sd.cpp distilled | [jetson-flux-klein-011](jetson-flux-klein-011.md) | VAE tiling | 25.98 | 0.0545 | 10.84 | – |
| Jetson edge-dit distilled | [jetson-edgedit-flux-klein-011](jetson-edgedit-flux-klein-011.md) | TaylorSeer (0/4 reused) | identical | 0.0000 | 0.00 | – |
| Jetson edge-dit distilled | [jetson-edgedit-flux-klein-013](jetson-edgedit-flux-klein-013.md) | MagCache (3/4 reused) | 13.27 | 0.3546 | 33.33 | degraded |
| Jetson edge-dit distilled | [jetson-edgedit-flux-klein-014](jetson-edgedit-flux-klein-014.md) | DiCache (0/4 reused) | identical | 0.0000 | 0.00 | – |
| Jetson edge-dit Base | [jetson-edgedit-flux-klein-base-002](jetson-edgedit-flux-klein-base-002.md) | EasyCache (29/100) | 19.80 | 0.0596 | 11.87 | legible; stray stroke |
| Jetson edge-dit Base | [jetson-edgedit-flux-klein-base-003](jetson-edgedit-flux-klein-base-003.md) | UCache (26/100) | 24.09 | 0.0336 | 7.77 | legible |
| Jetson edge-dit Base | [jetson-edgedit-flux-klein-base-004](jetson-edgedit-flux-klein-base-004.md) | DBCache (23/100) | 21.25 | 0.0495 | 9.24 | legible; stray stroke |
| Jetson edge-dit Base | [jetson-edgedit-flux-klein-base-005](jetson-edgedit-flux-klein-base-005.md) | TaylorSeer (38/100) | 35.94 | 0.0041 | 1.58 | near-identical |
| Jetson edge-dit Base | [jetson-edgedit-flux-klein-base-006](jetson-edgedit-flux-klein-base-006.md) | cache-dit (23/100) | 21.25 | 0.0495 | 9.24 | = DBCache output |
| Jetson edge-dit Base | [jetson-edgedit-flux-klein-base-007](jetson-edgedit-flux-klein-base-007.md) | MagCache (70/100) | 21.84 | 0.0279 | 6.50 | "world" deformed |
| Jetson PyTorch distilled | [jetson-torch-flux-klein-002](jetson-torch-flux-klein-002.md) | SDPA math backend | 20.17 | 0.0462 | 8.55 | – |
| Jetson PyTorch distilled | [jetson-torch-flux-klein-003](jetson-torch-flux-klein-003.md) | VAE tiling | 25.37 | 0.0506 | 10.97 | – |
| Jetson PyTorch distilled | [jetson-torch-flux-klein-004](jetson-torch-flux-klein-004.md) | conditioning reuse | identical | 0.0000 | 0.00 | – |
| Jetson PyTorch distilled | [jetson-torch-flux-klein-006](jetson-torch-flux-klein-006.md) | TaylorSeer | 14.97 | 0.2490 | 28.12 | visibly changed |
| Jetson PyTorch distilled | [jetson-torch-flux-klein-007](jetson-torch-flux-klein-007.md) | reuse + torch.compile | 23.77 | 0.0160 | 3.52 | – |
| Jetson PyTorch Base | [jetson-torch-flux-klein-base-002](jetson-torch-flux-klein-base-002.md) | conditioning reuse | identical | 0.0000 | 0.00 | – |
| Jetson PyTorch Base | [jetson-torch-flux-klein-base-004](jetson-torch-flux-klein-base-004.md) | reuse + torch.compile | 33.25 | 0.0125 | 2.53 | same text, visually same |
| A100 PyTorch torch 2.8 distilled | [a100-flux-klein-compile-001](a100-flux-klein-compile-001.md) | torch.compile | 32.54 | 0.0122 | 2.30 | – |
| A100 PyTorch torch 2.8 distilled | [a100-flux-klein-torch28-002](a100-flux-klein-torch28-002.md) | conditioning reuse | identical | 0.0000 | 0.00 | – |
| A100 PyTorch torch 2.8 distilled | [a100-flux-klein-torch28-004](a100-flux-klein-torch28-004.md) | TaylorSeer | 13.47 | 0.4391 | 39.29 | visibly changed |
| A100 PyTorch torch 2.8 Base | [a100-flux-klein-base-torch28-002](a100-flux-klein-base-torch28-002.md) | conditioning reuse | identical | 0.0000 | 0.00 | – |
| A100 PyTorch torch 2.8 Base | [a100-flux-klein-base-torch28-003](a100-flux-klein-base-torch28-003.md) | TaylorSeer | 14.59 | 0.2550 | 25.11 | changed; sign legible |
| A100 PyTorch torch 2.8 Base | [a100-flux-klein-base-torch28-004](a100-flux-klein-base-torch28-004.md) | torch.compile | 22.42 | 0.0915 | 9.79 | – |
| A100 edge-dit distilled (ref. image via 002 screen) | [a100-edgedit-flux-klein-007](a100-edgedit-flux-klein-007.md) | MagCache (3/4 reused) | 14.36 | 0.4372 | 34.46 | degraded |
| A100 edge-dit Base (Q4) | [a100-edgedit-flux-klein-base-002](a100-edgedit-flux-klein-base-002.md) | EasyCache (29/100) | 21.15 | 0.1187 | 12.24 | – |
| A100 edge-dit Base (Q4) | [a100-edgedit-flux-klein-base-003](a100-edgedit-flux-klein-base-003.md) | UCache (26/100) | 23.62 | 0.0705 | 8.99 | – |
| A100 edge-dit Base (Q4) | [a100-edgedit-flux-klein-base-004](a100-edgedit-flux-klein-base-004.md) | DBCache (24/100) | 22.47 | 0.0882 | 10.63 | – |
| A100 edge-dit Base (Q4) | [a100-edgedit-flux-klein-base-005](a100-edgedit-flux-klein-base-005.md) | TaylorSeer (38/100) | 32.07 | 0.0152 | 3.02 | indistinguishable |
| A100 edge-dit Base (Q4) | [a100-edgedit-flux-klein-base-006](a100-edgedit-flux-klein-base-006.md) | cache-dit (24/100) | 22.47 | 0.0882 | 10.63 | – |
| A100 edge-dit Base (Q4) | [a100-edgedit-flux-klein-base-007](a100-edgedit-flux-klein-base-007.md) | MagCache (70/100) | 19.00 | 0.1216 | 10.78 | sign misspelled |
| A100 edge-dit Base (Q4) | [a100-edgedit-flux-klein-base-008](a100-edgedit-flux-klein-base-008.md) | DiCache (54/100) | 26.53 | 0.0263 | 4.06 | – |

## Findings

- **Exact options are exact.** Conditioning reuse (PyTorch, Jetson and A100, both workloads) and the edge-dit caches that reused nothing at 4 steps (TaylorSeer, DiCache) give pixel-identical images.
- **Execution changes that alter arithmetic move pixels modestly:** VAE tiling (LPIPS 0.051–0.055, sd.cpp and PyTorch), SDPA math backend (0.046), sd.cpp flash attention off (0.003), torch.compile (0.012–0.092; A100 Base 0.092, others ≤ 0.016).
- **Step caches span a wide range.** Base, edge-dit: TaylorSeer is closest on both devices (LPIPS 0.004 Jetson, 0.015 A100), then DiCache (A100 0.026) and UCache; EasyCache, DBCache and cache-dit sit at 0.05–0.12. Distilled: MagCache (0.35–0.44) and PyTorch TaylorSeer (0.25–0.44) move the image far.
- **LPIPS misses text errors.** Jetson Base MagCache has LPIPS 0.028 (lower than EasyCache's 0.060) although its sign text is deformed; A100 Base MagCache (0.122, sign misspelled) is similar to EasyCache (0.119, legible). Paired perceptual distance cannot stand in for an alignment metric that checks content.

## Limits

- One development prompt and seed; no held-out data; no alignment or formal quality metric.
- Distances are relative to each engine's own reference, not to a BF16 or full-precision ground truth; Jetson references are themselves Q4.
- Compiled outputs are not bit-identical across processes on the Jetson distilled run, so its distance includes run-to-run variation.
