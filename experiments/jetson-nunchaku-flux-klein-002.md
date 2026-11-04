---
type: experiment-record
id: jetson-nunchaku-flux-klein-002
status: complete
device: jetson-orin-nano
engine: nunchaku
runs: [jetson-nunchaku-flux-klein-002__baseline__20261102-123151, jetson-nunchaku-flux-klein-002__attempt__20261102-122834]
updated: 2026-11-02
---

# jetson-nunchaku-flux-klein-002: exact conditioning reuse on the Nunchaku SVDQuant W4A4 configuration, 512×512

Status: **complete** (2026-11-02, run `__baseline__20261102-123151`, 17:31:51–17:35:18 UTC; run directly on the
device, no Slurm job). The smoke gate and the full 1+3+10 protocol completed.

## Question

How much of [jetson-nunchaku-flux-klein-001](jetson-nunchaku-flux-klein-001.md)'s per-image latency remains when
an identical prompt's encoding is reused? With reuse, the encoder load and the transformer release and reload happen
only in the first generation. RQ1 and the reuse dimension of the planner (RQ2); week 5 quantized variant.

## Setup

- [Configuration](../configs/jetson-nunchaku-flux-klein-int4-r32-512-condreuse.json) and
  [smoke configuration](../configs/jetson-nunchaku-flux-klein-int4-r32-512-condreuse-smoke.json).
- One change from [jetson-nunchaku-flux-klein-001](jetson-nunchaku-flux-klein-001.md):
  `optimizations.conditioning_reuse: true`. This is exact prompt-encoding reuse, a one-entry cache wrapped outside
  the stage-residency swaps. A hit skips the encoder load, the encoding and the transformer release/reload
  ([runner docstring](../scripts/run_flux_portable.py)).
- Everything else is as in 001:
  - checkpoint and kernels: community SVDQuant INT4/r32 checkpoint, Nunchaku `0c8379d` sm_87 in-place build with the
    two-line allowlist patch, hash verified at load;
  - NF4 text encoder, BF16 VAE, same stage-residency policy;
  - workload: distilled klein 512², 4 steps, guidance 1, seed 0, same prompt;
  - protocol 1+3+10; runner sha256 `e7337b57…`; torch 2.8.0, diffusers 0.40.0, R36.4.4, 25W.
- Launched by [queue O](../scripts/jetson_nunchaku_condreuse_queue_20261102.sh) under the memory guard
  (`MAX_SWAP_GROWTH_MB=1536`).
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md).

## Results

[Run](../results/runs/jetson-nunchaku-flux-klein-002__baseline__20261102-123151/),
[summary.json](../results/runs/jetson-nunchaku-flux-klein-002__baseline__20261102-123151/summary.json),
[conditioning-cache.json](../results/runs/jetson-nunchaku-flux-klein-002__baseline__20261102-123151/conditioning-cache.json)
(1 miss, 13 hits),
[stage-residency.json](../results/runs/jetson-nunchaku-flux-klein-002__baseline__20261102-123151/stage-residency.json)
(one swap cycle, in the first generation). Guard not triggered.

### Latency

| Metric | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall_ms | 3444.6 | 3435.3–3445.8 | – |
| text_encode_ms (cache hit) | 0.5 | 0.5–0.5 | 0.0% |
| denoise_ms | 2927.7 | 2919.6–2929.1 | 85.0% |
| denoise_step_0_ms | 735.4 | 733.7–736.9 | – |
| denoise_step_1_ms | 730.8 | 728.6–730.9 | – |
| vae_decode_ms | 486.1 | 485.0–486.6 | 14.1% |
| postprocess_ms | 10.2 | 9.7–10.4 | 0.3% |
| other_ms | 20.2 | 19.1–21.4 | 0.6% |

The first generation (cache miss) takes 75460.9 ms. Of that, text_encode_ms is 66846.5: under reuse, the encoder
load (29.5 s) and the transformer reload (31.8 s) fall inside the timed encode, because the cache wraps the
residency swaps. The first run's vae_decode_ms is 3779.0. Cold load is 67.5 s (cached read).

### Memory

Peak allocated 3.119 GiB and peak reserved 3.664 GiB in every measured generation. Whole-capture sampled system RAM
peak is 5.55 GiB; swap stays at 0.28–0.29 GiB; host peak RSS is 3.30 GiB; minimum MemAvailable is 1770 MB
([memguard.json](../results/runs/jetson-nunchaku-flux-klein-002__baseline__20261102-123151/memguard.json)).

### Delta vs reference (jetson-nunchaku-flux-klein-001)

| Metric | 001 median (ms) | 002 median (ms) | 002 / 001 |
|---|---:|---:|---:|
| wall_ms | 64598.3 | 3444.6 | 0.053 |
| denoise_ms | 2953.5 | 2927.7 | 0.991 |
| vae_decode_ms | 486.6 | 486.1 | 0.999 |
| other_ms | 59910.9 | 20.2 | – |
| peak_alloc_gib | 3.118 | 3.119 | – |

### Repeatability and output

- Wall time spread across the 10 measured generations is 10.6 ms.
- `deterministic_output` is false: 14 different pixel hashes for 14 generations.
- `first.png` and `measured-0.png` are nonflat 512×512 images. Their mean absolute pixel difference is 12.88 on a
  0–255 scale (max 217), a diagnostic only. Quality is not measured.
- Smoke ([summary](../results/runs/jetson-nunchaku-flux-klein-002__attempt__20261102-122834/summary.json)): complete;
  first 76330.9 ms, one measured 3431.3 ms.

### W&B

Uploaded and verified 2026-11-02 (state finished, medians match `summary.json`, two images, run-directory artifact): [full](https://forge.coreweave.com/wandb/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-nunchaku-flux-klein-002__baseline__20261102-123151), [smoke](https://forge.coreweave.com/wandb/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-nunchaku-flux-klein-002__attempt__20261102-122834).

## Observations

- With the encoding reused, a generation is 85% denoising and 14% VAE decode.
- Denoise and VAE times are within 1% of 001; reuse removes only the reload cost.
- Measured generations hold memory steady: swap does not grow, and the allocator peak equals 001's.
- The first generation still pays the full swap cycle and is 22× slower than a measured generation.

## Interpretation

- In steady state with a repeated prompt, this configuration's latency is set by the W4A4 transformer (4 × ~730 ms)
  plus the BF16 VAE decode. Further speedup has to come from the denoiser or the decoder, not from residency.
- The result applies only to repeated identical prompts. A new prompt pays the first-generation cost (about 75 s) as
  long as the encoder and transformer cannot be resident together.

## Next experiment

- `jetson-nunchaku-flux-klein-int4-r32-512-condreuse` with the text encoder kept resident instead of released, to test whether the 2.28 GiB W4A4 transformer, NF4 encoder and VAE fit together without swaps.
- Quality: paired LPIPS/CLIP against the Jetson GGUF images on the calibration prompt set.
