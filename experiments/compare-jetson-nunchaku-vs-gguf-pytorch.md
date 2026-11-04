---
type: experiment-record
id: compare-jetson-nunchaku-vs-gguf-pytorch
status: complete
device: jetson-orin-nano
engine: nunchaku
runs: [jetson-torch-flux-klein-001__baseline__20261014-171410, jetson-nunchaku-flux-klein-001__baseline__20261102-120036, jetson-torch-flux-klein-004__baseline__20261015-141028, jetson-nunchaku-flux-klein-002__baseline__20261102-123151]
updated: 2026-11-02
---

# Compare: Jetson klein 512², GGUF Q4_0 transformer vs Nunchaku SVDQuant W4A4 transformer

## Question

On the Jetson Orin Nano, with the encoder, VAE, workload and stage residency fixed, how do stage latency and memory
change when the GGUF Q4_0 transformer (4-bit weights, BF16 compute) is replaced by the community SVDQuant INT4
W4A4/r32 checkpoint run by Nunchaku's fused kernels? RQ1 (precision and engine options on the edge device); week 5
quantized variant.

## Runs

| Role | Run and configuration | Record |
|---|---|---|
| Reference: GGUF Q4_0 | [jetson-torch-flux-klein-001__baseline__20261014-171410](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/config.json) | [jetson-torch-flux-klein-001](jetson-torch-flux-klein-001.md) |
| Variant: SVDQuant W4A4 | [jetson-nunchaku-flux-klein-001__baseline__20261102-120036](../results/runs/jetson-nunchaku-flux-klein-001__baseline__20261102-120036/config.json) | [jetson-nunchaku-flux-klein-001](jetson-nunchaku-flux-klein-001.md) |

## Matching

| Field | Both runs |
|---|---|
| Device | Jetson Orin Nano 8 GB, 25W, R36.4.4, CUDA 12.6 |
| Software | torch 2.8.0, diffusers 0.40.0, transformers 5.16.1, bitsandbytes 0.48.0.dev0+ff389db |
| Workload | Distilled klein 512×512, 4 steps, guidance 1, batch 1, seed 0, same prompt, max_sequence_length 512 |
| Text encoder, VAE | Same pre-quantized NF4 Qwen3 encoder (sha256 `eaa2048d…`), BF16 VAE from the pinned snapshot |
| Residency | Encoder released after each encoding; transformer released during encoding and reloaded |
| Protocol | 1 first + 3 warm-up + 10 measured; memory guard 1536 MB swap growth |
| **Changed** | Transformer: GGUF Q4_0 file (BF16 compute in diffusers `GGUFLinear`) → community SVDQuant INT4/r32 checkpoint (Nunchaku W4A4 kernels, sm_87 source build) |

Confounds:
- The change is two different compressed checkpoints plus two different kernel implementations, not one execution option.
  The SVDQuant checkpoint's ancestry to the pinned BF16 transformer is not established.
- The runs are 19 days apart.
- The runner version differs (sha256 `7d8b7d39…` vs `e7337b57…`); the later version adds options that neither
  configuration sets.
- Quality is not measured on either side.

The comparison is descriptive.

## Results

Sources: [reference summary](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/summary.json),
[variant summary](../results/runs/jetson-nunchaku-flux-klein-001__baseline__20261102-120036/summary.json),
the two `stage-residency.json` files and the two `load.json` files.

| Metric (ms) | GGUF Q4_0, median (min–max) | SVDQuant W4A4, median (min–max) | Variant / reference |
|---|---:|---:|---:|
| wall_ms | 80314.9 (78169.8–82599.2) | 64598.3 (58458.7–72655.9) | 0.804 |
| text_encode_ms | 1241.3 (1187.0–2167.5) | 1104.1 (1063.7–2205.5) | 0.889 |
| denoise_ms | 14795.5 (14705.3–14817.0) | 2953.5 (2940.7–3249.9) | 0.200 |
| denoise_step_0_ms | 4015.3 (3955.8–4038.4) | 761.0 (755.0–1055.9) | 0.190 |
| denoise_step_1_ms | 3591.2 (3580.2–3603.7) | 728.9 (726.4–730.0) | 0.203 |
| vae_decode_ms | 491.4 (489.6–540.6) | 486.6 (485.3–487.2) | 0.990 |
| other_ms (reloads) | 63343.9 (61584.6–65354.9) | 59910.9 (53910.5–66815.0) | 0.946 |
| First run wall_ms | 94586.1 | 77784.4 | 0.822 |

| Reload, median over 14 generations (s) | GGUF Q4_0 | SVDQuant W4A4 |
|---|---:|---:|
| Text encoder load | 33.48 | 33.11 |
| Transformer load | 30.23 | 27.88 |

| Memory | GGUF Q4_0 | SVDQuant W4A4 |
|---|---:|---:|
| Transformer weights (GiB) | 2.291 | 2.280 |
| Peak allocated per generation (GiB) | 3.081 | 3.118 |
| Peak reserved per generation (GiB) | 3.422 | 3.650 |
| Whole-capture system RAM peak, tegrastats (GiB) | 6.11 | 5.53 |
| Whole-capture swap max (GiB) | 1.23 | 0.96 |
| Host peak RSS (GiB) | 5.48 | 3.65 |
| Memory guard MemAvailable minimum (MB) | 1206 | 1771 |

Output: the reference is bitwise deterministic across its measured generations (`deterministic_output` true); the
variant is not (14 different hashes). Neither side has a quality score.

## Findings

- At the same resident weight size, the W4A4 transformer denoises 5.0× faster (2953.5 vs 14795.5 ms per
  generation). In a single-layer check on the same board, the W4A4 layers ran at 0.35–0.57 of BF16 `F.linear` time,
  while the GGUF path dequantizes to BF16 and computes in BF16
  ([kernel check](jetson-nunchaku-flux-klein-001.md#kernel-check-gate)).
- End-to-end wall time improves by only 19.6%. Under sequential residency, reloads are 79% (reference) and 93%
  (variant) of median wall time, and their median sum (encoder + transformer, about 61–64 s) is unchanged by the
  transformer's kernels.
- Per-generation allocator peaks are within 0.23 GiB; the transformer's packed size is the same.
- The whole-capture RAM, swap and RSS are lower for the variant. These metrics include load and are sampled across
  the whole capture, so this comparison cannot attribute the difference to the transformer format.
- Determinism does not transfer: the fused W4A4 path gives run-to-run pixel differences at a fixed seed.

## With exact conditioning reuse

The same transformer change, with `conditioning_reuse: true` on both sides. Everything else is matched as above.

| Role | Run | Record |
|---|---|---|
| Reference: GGUF Q4_0 + reuse | [jetson-torch-flux-klein-004__baseline__20261015-141028](../results/runs/jetson-torch-flux-klein-004__baseline__20261015-141028/config.json) | [jetson-torch-flux-klein-004](jetson-torch-flux-klein-004.md) |
| Variant: SVDQuant W4A4 + reuse | [jetson-nunchaku-flux-klein-002__baseline__20261102-123151](../results/runs/jetson-nunchaku-flux-klein-002__baseline__20261102-123151/config.json) | [jetson-nunchaku-flux-klein-002](jetson-nunchaku-flux-klein-002.md) |

The runner versions differ (`44e44608…` vs `e7337b57…`), and the runs are 18 days apart.

| Metric (ms) | GGUF Q4_0 + reuse, median (min–max) | SVDQuant W4A4 + reuse, median (min–max) | Variant / reference |
|---|---:|---:|---:|
| wall_ms | 14911.4 (14880.6–14961.3) | 3444.6 (3435.3–3445.8) | 0.231 |
| denoise_ms | 14377.3 (14350.5–14426.8) | 2927.7 (2919.6–2929.1) | 0.204 |
| denoise_step_1_ms | 3592.4 (3585.9–3606.2) | 730.8 (728.6–730.9) | 0.203 |
| vae_decode_ms | 485.7 (485.4–486.2) | 486.1 (485.0–486.6) | 1.001 |
| First run wall_ms | 96501.7 | 75460.9 | 0.782 |
| Peak allocated (GiB) | 3.083 | 3.119 | – |
| Peak reserved (GiB) | 3.426 | 3.664 | – |

Sources: [reference summary](../results/runs/jetson-torch-flux-klein-004__baseline__20261015-141028/summary.json),
[variant summary](../results/runs/jetson-nunchaku-flux-klein-002__baseline__20261102-123151/summary.json).

- Without reloads, the transformer's denoise ratio (0.204) carries through to wall time: 4.3× faster per image
  (14.91 → 3.44 s). This is consistent with the first finding: residency, not the kernels, limited the wall ratio in
  the no-reuse pair.
- The reference stays bitwise deterministic; the variant does not.

## Limits

- This comparison doesn't measure quality. It can't say whether W4A4 or Q4_0 preserves the BF16 model better.
- The 5× denoise ratio is specific to this pair of implementations: diffusers GGUF dequantize-then-BF16 vs Nunchaku
  fused W4A4. It doesn't isolate 4-bit activations from kernel quality.
- The wall-time ratio depends on the residency policy: 0.80 with per-generation reloads, 0.23 with exact conditioning
  reuse. Reuse applies only to repeated identical prompts.
