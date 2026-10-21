---
type: experiment-record
id: a100-flux-klein-torch28-004
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-torch28-004__baseline__20261015-151949, a100-flux-klein-torch28-004__attempt__20261015-151930]
updated: 2026-10-15
---

# a100-flux-klein-torch28-004: diffusers TaylorSeer cache (interval 2, from step 1, order 1), BF16 1024² klein, torch 2.8.0; denoise −42.9%, output visibly changed

Status: **complete** (2026-10-15, job 11959806 on gilbreth-g008; full run 19:19:49–19:20:18 UTC, smoke
19:19:30–19:19:48 UTC). The output image changes visibly (different sign text and tone).

## Question

How much denoising time does diffusers' TaylorSeer cache remove from the 4-step BF16 reference workload, and how
does it change memory and the output? Reference: [a100-flux-klein-torch28-001](a100-flux-klein-torch28-001.md),
baseline [`__baseline__20261015-151548`](../results/runs/a100-flux-klein-torch28-001__baseline__20261015-151548/).
Step-reuse lever under a quality requirement (RQ2).

## Setup

- Config [a100-flux-klein-torch28-bf16-taylorseer.json](../configs/a100-flux-klein-torch28-bf16-taylorseer.json);
  smoke [a100-flux-klein-torch28-bf16-taylorseer-smoke.json](../configs/a100-flux-klein-torch28-bf16-taylorseer-smoke.json).
  The run-directory `config.json` copies equal these files.
- **One change vs [a100-flux-klein-torch28-bf16.json](../configs/a100-flux-klein-torch28-bf16.json):**
  `optimizations.step_cache: {"method": "taylorseer", "cache_interval": 2, "disable_cache_before_step": 1,
  "max_order": 1}`.
- How it is applied ([scripts/run_flux_portable.py](../scripts/run_flux_portable.py) `apply_step_cache`):
  `pipe.transformer.enable_cache(TaylorSeerCacheConfig(cache_interval=2, disable_cache_before_step=1, max_order=1))`
  once after load. Other `TaylorSeerCacheConfig` fields are at their diffusers 0.40.0 defaults.
- **Receipt limitation.** [step-cache.json](../results/runs/a100-flux-klein-torch28-004__baseline__20261015-151949/step-cache.json)
  field `full_transformer_computations_per_generation` counts forward-pre-hook calls on the **last single-stream
  block** (`count_full_computations`). TaylorSeer predicts the outputs of sub-modules inside the blocks, so the
  blocks are still entered on predicted steps. The counter reads 4 for all 14 generations and **does not measure
  TaylorSeer's prediction**. Per-step timings (`denoise_step_i_ms`) are used below instead.
- Workload, protocol (1 + 3 + 10; smoke 1 + 1), device and environment as in
  [torch28-001 setup](a100-flux-klein-torch28-001.md#setup). Runner sha256 `44e44608…`.
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md). With a step cache, the mean of
  steps 1–3 mixes computed and predicted steps; per-step medians are reported alongside it.

## Results

[Run](../results/runs/a100-flux-klein-torch28-004__baseline__20261015-151949/),
[summary.json](../results/runs/a100-flux-klein-torch28-004__baseline__20261015-151949/summary.json),
[runs.csv](../results/runs/a100-flux-klein-torch28-004__baseline__20261015-151949/runs.csv),
[stages.csv](../results/runs/a100-flux-klein-torch28-004__baseline__20261015-151949/stages.csv).
`status.json` `complete`.

### Latency

| Stage | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| text_encode_ms | 46.4 | 46.1–49.0 | 5.4% |
| denoise_ms | 592.8 | 589.3–594.0 | 69.4% |
| denoise_step_0_ms (computed) | 255.4 | 254.4–257.3 | – |
| denoise_step_1_ms (predicted) | 36.7 | 36.6–37.1 | – |
| denoise_step_2_ms (computed) | 260.0 | 258.2–263.0 | – |
| denoise_step_3_ms (predicted) | 39.6 | 39.1–39.9 | – |
| Mean of steps 1–3 per generation | 112.2 | 111.3–113.1 | – |
| vae_decode_ms | 168.1 | 166.2–169.0 | 19.7% |
| postprocess_ms | 38.4 | 30.3–43.7 | 4.5% |
| other_ms | 8.1 | 7.8–8.6 | 1.0% |
| **wall_ms** | **854.2** | 844.1–859.0 (stdev 4.9) | 100% |
| First generation: wall_ms | 1437.1 | – | text 384.8 / denoise 676.0 (steps 340.7 / 38.0 / 258.2 / 39.1) / VAE 306.0 / post 49.4 / other 20.8 |
| Warm-ups (3, discarded): wall_ms | – | 842.1–849.7 | – |
| Load (cached, s) | 8.79 | – | `from_pretrained` 2.79 + `.to("cuda")` 6.00 |

"Computed"/"predicted" labels follow from the step times (steps 1 and 3 take 37–40 ms, steps 0 and 2 take
254–263 ms) and from the configured interval; the runner does not record which steps were predicted.

### Memory (GiB)

| Quantity | Value |
|---|---:|
| peak_alloc_gib (all measured equal) | **18.64**, set by `vae_decode` |
| Allocated at stage start (steps 0 / 1 / 2 / 3 / VAE decode) | 14.94 / 15.60 / 15.60 / 16.26 / 16.26 |
| Stage peaks (steps 0 / 1 / 2 / 3 / VAE decode) | 16.15 / 16.02 / 16.80 / 16.68 / 18.64 |
| peak_reserved_gib | 20.80 |
| device_used_peak_gib (NVML, 20 ms) | 21.81 |
| host_rss_gib, measured median | 1.69 (1.65–1.69) |
| host_peak_rss_gib (includes load) | 8.07 |

Allocated memory rises by 0.66 GiB after step 0 and again after step 2 and is held through VAE decode and
postprocess; the next generation's text encode starts at 14.94 GiB again (from `stages.csv`).

### Repeatability and functional check (visual, not a quality metric)

All 14 generations have `image_sha256` `ce5caa2fe8a902d28630eb3efa73623e825251fcd3a2d69462f0f00ddb8f01e4`
(`deterministic_output: true`); the smoke run has the same hash. It differs from torch28-001 (`f11220f6…`): mean
absolute RGB difference 39.3 levels, 95.0% of pixels differ by more than 8 levels in any channel (computed from the
two `images/measured-0.png` files).

Visual check of `images/measured-0.png` against torch28-001's: same scene type (tabby cat behind a sign held in
both paws), but the sign is beige/cardboard-coloured instead of white, the overall tone is warmer and darker, and
the background changes (indoor wall and floor instead of a window). The sign text reads "molon" in thick brush
strokes instead of "hellword". The cat's face and fur remain coherent.

### Smoke gate

[`__attempt__20261015-151930`](../results/runs/a100-flux-klein-torch28-004__attempt__20261015-151930/):
complete. First 1477.4 ms (steps 324.6 / 94.4 / 258.1 / 39.4 ms); one measured 850.0 ms (steps 256.6 / 36.6 /
261.5 / 39.1 ms; VAE 167.9 ms). Load 8.84 s. Memory 18.64 / 20.80 / 21.81 GiB. Not a baseline (n = 1).

### Delta vs reference (torch28-001 baseline)

| Metric | torch28-001 median | torch28-004 median | Δ |
|---|---:|---:|---:|
| wall_ms | 1290.5 | 854.2 | −436.3 (−33.8%) |
| denoise_ms | 1037.8 | 592.8 | −444.9 (−42.9%) |
| denoise_step_0 / 1 / 2 / 3_ms | 255.6 / 257.1 / 258.4 / 266.0 | 255.4 / 36.7 / 260.0 / 39.6 | −0.1 / −220.5 / +1.7 / −226.4 |
| Mean of steps 1–3 | 260.3 | 112.2 | −148.1 (−56.9%) |
| text_encode_ms | 46.2 | 46.4 | +0.2 |
| vae_decode_ms | 168.2 | 168.1 | −0.1 |
| First generation wall_ms | 1830.9 | 1437.1 | −393.8 (−21.5%) |
| peak_alloc / peak_reserved / device_used_peak (GiB) | 17.32 / 19.61 / 20.62 | 18.64 / 20.80 / 21.81 | +1.32 / +1.19 / +1.19 |

## Observations

- Steps 1 and 3 take 36.7 and 39.6 ms against about 255–260 ms for the computed steps; denoising falls by
  444.9 ms (−42.9%) and wall by 436.3 ms (−33.8%).
- Computed steps 0 and 2 take the same time as in the reference (within 2 ms).
- The step-cache receipt reports 4 full computations in every generation, which contradicts the step timings;
  the counter does not observe TaylorSeer's predictions (see Setup).
- The allocator peak rises by 1.32 GiB to 18.64 GiB, in two 0.66 GiB increments after each computed step, and
  the device-wide peak by 1.19 GiB to 21.81 GiB.
- The output is deterministic but visibly different from the reference, with different sign text, sign colour,
  tone and background.

## Interpretation

- At 4 steps with interval 2, TaylorSeer predicts two of four steps; this removes about 43% of transformer time
  but extrapolates from one or two computed points, and the changed image (text, colour, background) shows the
  approximation error at this step count. Under a quality requirement that includes matching the reference
  content or legible text, this setting does not qualify without a quality measurement.
- Hypothesis: the 37–40 ms residual on predicted steps is the work the cache does not replace (embeddings,
  modulation, norms, final projection and non-cached modules) plus hook overhead; a per-module profile would
  separate them.
- Hypothesis: the +0.66 GiB per computed step is the TaylorSeer state (stored outputs and first-order differences
  per cached sub-module), released at the end of each generation.

## Next experiment

- Runner: count predicted steps from the TaylorSeer hook state instead of the last-block pre-hook.
- Copy of `a100-flux-klein-torch28-bf16-taylorseer.json` with a paired saved-image quality metric against torch28-001 (calibration prompts only).
