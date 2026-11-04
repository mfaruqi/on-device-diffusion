---
type: experiment-record
id: a100-edgedit-flux-klein-base-007
status: complete
device: a100-pcie-40gb
engine: edge-dit-cpp
runs: [a100-edgedit-flux-klein-base-007__baseline__20261015-175238, a100-edgedit-flux-klein-base-007__attempt__20261015-175159]
updated: 2026-10-15
---

# a100-edgedit-flux-klein-base-007: edge-dit.cpp step cache `magcache` (engine defaults), klein Base GGUF Q4_0 1024², 50 steps CFG 4; 70 of 100 transformer passes reused, denoise −69.5%

Status: **complete** (2026-10-15, job 11960173 on gilbreth-g002; full run 21:52:38–21:56:28 UTC, smoke 21:51:59–21:52:38 UTC).
Full 1+3+10 protocol; the engine reused 70 transformer passes in every generation.

## Question

Does edge-dit.cpp's `magcache` step cache, at its default parameters, skip or reuse transformer work in the 50-step
CFG klein Base configuration [a100-edgedit-flux-klein-base-001](a100-edgedit-flux-klein-base-001.md), and what does it
change in latency, device memory and output? Single change against
[a100-edgedit-flux-klein-base-q4-1024.json](../configs/a100-edgedit-flux-klein-base-q4-1024.json). Step-reuse lever under
a quality requirement (RQ2).

## Setup

- **Config**: [a100-edgedit-flux-klein-base-q4-1024-magcache.json](../configs/a100-edgedit-flux-klein-base-q4-1024-magcache.json);
  smoke [a100-edgedit-flux-klein-base-q4-1024-magcache-smoke.json](../configs/a100-edgedit-flux-klein-base-q4-1024-magcache-smoke.json)
  (first + one measured). The run-directory `config.json` copies equal these files; `reference_config` is
  `configs/a100-edgedit-flux-klein-base-q4-1024.json`.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "magcache"}`, mapped by
  [scripts/run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) to `--cache_method magcache` with no tuning flags, so
  the engine's per-method defaults apply ([command.json](../results/runs/a100-edgedit-flux-klein-base-007__baseline__20261015-175238/command.json)). The engine log does not
  print the default parameter values.
- Everything else as in the [reference setup](a100-edgedit-flux-klein-base-001.md#setup): edge-dit.cpp `97cc585`,
  `ed-sample` sha256 `fc6b14a7…`; GGUF Q4_0 transformer, Q4_K_M Qwen3 encoder, BF16 VAE, FP32 activations; flash
  attention on; VAE tiling off; 1024×1024, 50 Euler steps, guidance 4.0, CFG 4.0, seed 0, one prompt; four threads;
  runner sha256 `153d3ea9…`. Same job and node as the reference.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`.
  [scripts/gilbreth-gap-base-edgedit-20261015.slurm](../scripts/gilbreth-gap-base-edgedit-20261015.slurm) runs the
  full protocol only when the smoke's `step-cache.json` reports skipped/reused work (here [70, 70]).
- **Cache counts are per transformer pass.** With CFG each step runs two transformer passes (conditional and
  unconditional; the engine reports `variants=2`), 100 per image. The engine's per-generation line
  `<Method> reused k/50 steps` prints the step count as denominator, but k counts passes: this run prints
  k = 70 > 50 (as does DiCache, [a100-edgedit-flux-klein-base-008](a100-edgedit-flux-klein-base-008.md)), and the measured denoise time scales with (100 − k)/100 (Results).
  `step-cache.json → skipped_steps_per_generation` copies k unchanged.
- Metric derivation: [Jetson edge-dit runner](../wiki/methods/baseline-metrics.md#jetson-edge-dit-runner) (wall from
  `ed-sample` stdout at 1 ms rounding; text / sampling / VAE / other from the engine breakdown at 10 ms rounding; no
  per-step times; device memory is one NVML peak over the whole child process including load; only the final pass's
  image retained).

## Results

Full run [a100-edgedit-flux-klein-base-007__baseline__20261015-175238](../results/runs/a100-edgedit-flux-klein-base-007__baseline__20261015-175238/): [status.json](../results/runs/a100-edgedit-flux-klein-base-007__baseline__20261015-175238/status.json) `complete`.

Cache receipt ([engine.log](../results/runs/a100-edgedit-flux-klein-base-007__baseline__20261015-175238/engine.log),
[step-cache.json](../results/runs/a100-edgedit-flux-klein-base-007__baseline__20261015-175238/step-cache.json)): every pass logs `cache enabled: mode=MagCache model=flux2-klein granularity=feature matched_sites=1 variants=2 slots=1` and
`MagCache reused 70/50 steps` (14 of 14 passes; smoke 2 of 2; source `magcache_policy.cpp`). `skipped_steps_per_generation` is 70 in all 14
generations. The log does not say which passes were reused. The engine still
printed 50 `flux step` lines per pass (700 in all).

### Latency (measured n = 10; [summary.json](../results/runs/a100-edgedit-flux-klein-base-007__baseline__20261015-175238/summary.json), [runs.csv](../results/runs/a100-edgedit-flux-klein-base-007__baseline__20261015-175238/runs.csv), [stages.csv](../results/runs/a100-edgedit-flux-klein-base-007__baseline__20261015-175238/stages.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall (1 ms rounding) | **15707.5** | 15538.0–15843.0 (stdev 84.6) | 100% |
| text_encode (engine breakdown) | 120 | 120–340 | 0.8% |
| denoise (engine `sampling`) | 14890 | 14840–15010 | 94.8% |
| vae_decode | 280 | 270–470 | 1.8% |
| other | 310 | 310–450 | 2.0% |
| `[[phase]]` markers: encode_setup / denoise / vae_decode | 421.5 / 14890.1 / 275.0 | 420.9–639.6 / 14837.1–15010.1 / 273.7–474.1 | – |
| First generation: wall | 15934.0 | – | text 190 / denoise 14900 / vae 530 / other 320 |
| Warm-ups (3, discarded): wall | – | 15632.0–15927.0 | – |
| Model load (s) | 2.26 | – | page cache warm |

Denoise per computed pass (derived): 14890 / (100 − 70) = 496.3 ms; reference 48740 / 100 = 487.4 ms.

### Memory (GiB; whole child process, includes load)

| Quantity | Value |
|---|---:|
| device_used_peak_gib (NVML, 20 ms; whole capture) | 12.019 (reference 11.732) |

Per-generation device peaks, allocator metrics and host RSS are unavailable for this runner.

### Smoke run

[a100-edgedit-flux-klein-base-007__attempt__20261015-175159](../results/runs/a100-edgedit-flux-klein-base-007__attempt__20261015-175159/): complete; reused 70 in both passes. First 16070.0 ms (text
340 / denoise 15050 / vae 370 / other 310); one
measured 15619.0 ms (230 / 14820 / 270 / 300). Load
2.09 s. Device peak 12.019 GiB. Not a baseline (n = 1).

### Delta vs reference ([a100-edgedit-flux-klein-base-001 baseline](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/), same job and node)

| Metric | base-001 median | 007 median | Δ |
|---|---:|---:|---:|
| wall_ms | 49428.5 | 15707.5 | −33721.0 (−68.2%) |
| denoise | 48740 | 14890 | −33850 (−69.5%) |
| denoise / reference denoise | 1 | 0.305 | computed-pass fraction (100 − k)/100 = 0.30 |
| text_encode / vae_decode / other | 120 / 280 / 290 | 120 / 280 / 310 | 0 / 0 / +20 |
| First generation wall | 49476.0 | 15934.0 | −33542.0 (−67.8%) |
| device_used_peak_gib (whole capture) | 11.732 | 12.019 | +0.287 |

### Repeatability and functional check (visual, not a quality metric)

Final-pass image (run index 13) RGB SHA256 `7f02244b123ad0f8…`; the smoke's final image has the same hash, so the output
reproduces across two processes. It differs from the reference final image (`dd6b02c1…`): mean absolute RGB difference
10.8 levels; 58.0% of pixels differ by more than 8 levels in any channel (computed from the two `output.png` files).

Visual check of `output.png` against the reference's: same composition, cat, cardboard sign and background as the reference (background slightly paler), with no blur, but the first word is misspelled: the sign reads "hdlloo world". Text content is visibly changed.

## Observations

- The engine reused 70 of 100 transformer passes in every generation; denoise fell from 48740 to 14890 ms
  (−69.5%) and wall from 49428.5 to 15707.5 ms (−68.2%).
- The measured denoise ratio (0.305) matches the computed-pass fraction ((100 − 70)/100 = 0.30); time per computed pass is 496.3 ms against 487.4 ms in the reference.
- Text encode, VAE decode and other medians are within 20 ms of the reference.
- Whole-process device peak 12.019 GiB (+0.287 GiB vs the reference capture).
- Output deterministic across the smoke and full processes (final-image hash); 10.8 mean RGB levels from the reference image.
- The measured wall range is 305.0 ms; denoise varies by 170 ms, and text encode reaches 340 ms (median 120), VAE decode reaches 470 ms (median 280), other reaches 450 ms (median 310) in at least one measured generation.

## Interpretation

- At engine defaults, MagCache reuses 70 of 100 transformer passes; latency falls almost in proportion (denoise −69.5%), but the sign text is misspelled ("hdlloo"), so this setting fails any quality requirement that includes the prompt's text, even though composition and sharpness are preserved.
- Hypothesis: MagCache's default magnitude-ratio threshold allows long reuse runs at this schedule; 70 of 100 passes reused leaves too few computed passes to preserve fine text structure, while coarse content survives.

## Next experiment

- Copy of `a100-edgedit-flux-klein-base-q4-1024-magcache.json` with a stricter threshold (needs a runner option; separately labelled) and a saved-image comparison against base-001.
