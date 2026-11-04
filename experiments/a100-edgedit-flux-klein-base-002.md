---
type: experiment-record
id: a100-edgedit-flux-klein-base-002
status: complete
device: a100-pcie-40gb
engine: edge-dit-cpp
runs: [a100-edgedit-flux-klein-base-002__baseline__20261015-170356, a100-edgedit-flux-klein-base-002__attempt__20261015-170234]
updated: 2026-10-15
---

# a100-edgedit-flux-klein-base-002: edge-dit.cpp step cache `easycache` (engine defaults), klein Base GGUF Q4_0 1024², 50 steps CFG 4; 29 of 100 transformer passes skipped, denoise −28.7%

Status: **complete** (2026-10-15, job 11960173 on gilbreth-g002; full run 21:03:56–21:12:22 UTC, smoke 21:02:34–21:03:56 UTC).
Full 1+3+10 protocol; the engine skipped 29 transformer passes in every generation.

## Question

Does edge-dit.cpp's `easycache` step cache, at its default parameters, skip or reuse transformer work in the 50-step
CFG klein Base configuration [a100-edgedit-flux-klein-base-001](a100-edgedit-flux-klein-base-001.md), and what does it
change in latency, device memory and output? Single change against
[a100-edgedit-flux-klein-base-q4-1024.json](../configs/a100-edgedit-flux-klein-base-q4-1024.json). Step-reuse lever under
a quality requirement (RQ2).

## Setup

- **Config**: [a100-edgedit-flux-klein-base-q4-1024-easycache.json](../configs/a100-edgedit-flux-klein-base-q4-1024-easycache.json);
  smoke [a100-edgedit-flux-klein-base-q4-1024-easycache-smoke.json](../configs/a100-edgedit-flux-klein-base-q4-1024-easycache-smoke.json)
  (first + one measured). The run-directory `config.json` copies equal these files; `reference_config` is
  `configs/a100-edgedit-flux-klein-base-q4-1024.json`.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "easycache"}`, mapped by
  [scripts/run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) to `--cache_method easycache` with no tuning flags, so
  the engine's per-method defaults apply ([command.json](../results/runs/a100-edgedit-flux-klein-base-002__baseline__20261015-170356/command.json)). The engine log does not
  print the default parameter values.
- Everything else as in the [reference setup](a100-edgedit-flux-klein-base-001.md#setup): edge-dit.cpp `97cc585`,
  `ed-sample` sha256 `fc6b14a7…`; GGUF Q4_0 transformer, Q4_K_M Qwen3 encoder, BF16 VAE, FP32 activations; flash
  attention on; VAE tiling off; 1024×1024, 50 Euler steps, guidance 4.0, CFG 4.0, seed 0, one prompt; four threads;
  runner sha256 `153d3ea9…`. Same job and node as the reference.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`.
  [scripts/gilbreth-gap-base-edgedit-20261015.slurm](../scripts/gilbreth-gap-base-edgedit-20261015.slurm) runs the
  full protocol only when the smoke's `step-cache.json` reports skipped/reused work (here [29, 29]).
- **Cache counts are per transformer pass.** With CFG each step runs two transformer passes (conditional and
  unconditional; the engine reports `variants=2`), 100 per image. The engine's per-generation line
  `<Method> skipped k/50 steps` prints the step count as denominator, but k counts passes: MagCache and DiCache print
  k > 50 on this workload ([a100-edgedit-flux-klein-base-007](a100-edgedit-flux-klein-base-007.md),
  [-008](a100-edgedit-flux-klein-base-008.md)), and the measured denoise time scales with (100 − k)/100 (Results).
  `step-cache.json → skipped_steps_per_generation` copies k unchanged.
- Metric derivation: [Jetson edge-dit runner](../wiki/methods/baseline-metrics.md#jetson-edge-dit-runner) (wall from
  `ed-sample` stdout at 1 ms rounding; text / sampling / VAE / other from the engine breakdown at 10 ms rounding; no
  per-step times; device memory is one NVML peak over the whole child process including load; only the final pass's
  image retained).

## Results

Full run [a100-edgedit-flux-klein-base-002__baseline__20261015-170356](../results/runs/a100-edgedit-flux-klein-base-002__baseline__20261015-170356/): [status.json](../results/runs/a100-edgedit-flux-klein-base-002__baseline__20261015-170356/status.json) `complete`.

Cache receipt ([engine.log](../results/runs/a100-edgedit-flux-klein-base-002__baseline__20261015-170356/engine.log),
[step-cache.json](../results/runs/a100-edgedit-flux-klein-base-002__baseline__20261015-170356/step-cache.json)): every pass logs `cache enabled: mode=EasyCache model=flux2-klein granularity=output matched_sites=1 variants=2 slots=1` and
`EasyCache skipped 29/50 steps (2.38x estimated speedup)` (14 of 14 passes; smoke 2 of 2; source `easycache_policy.cpp`). `skipped_steps_per_generation` is 29 in all 14
generations. The log does not say which passes were skipped. "(2.38x)" is the engine's own estimate (50 / (50 − k)), not a measured speed-up. The engine still
printed 50 `flux step` lines per pass (700 in all).

### Latency (measured n = 10; [summary.json](../results/runs/a100-edgedit-flux-klein-base-002__baseline__20261015-170356/summary.json), [runs.csv](../results/runs/a100-edgedit-flux-klein-base-002__baseline__20261015-170356/runs.csv), [stages.csv](../results/runs/a100-edgedit-flux-klein-base-002__baseline__20261015-170356/stages.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall (1 ms rounding) | **35435.5** | 35426.0–35453.0 (stdev 9.8) | 100% |
| text_encode (engine breakdown) | 120 | 120–120 | 0.3% |
| denoise (engine `sampling`) | 34740 | 34730–34760 | 98.0% |
| vae_decode | 280 | 270–280 | 0.8% |
| other | 300 | 300–300 | 0.8% |
| `[[phase]]` markers: encode_setup / denoise / vae_decode | 414.8 / 34739.6 / 276.6 | 413.5–415.6 / 34731.0–34755.4 / 274.5–278.7 | – |
| First generation: wall | 35610.0 | – | text 180 / denoise 34750 / vae 380 / other 310 |
| Warm-ups (3, discarded): wall | – | 35380.0–35432.0 | – |
| Model load (s) | 2.16 | – | page cache warm |

Denoise per computed pass (derived): 34740 / (100 − 29) = 489.3 ms; reference 48740 / 100 = 487.4 ms.

### Memory (GiB; whole child process, includes load)

| Quantity | Value |
|---|---:|
| device_used_peak_gib (NVML, 20 ms; whole capture) | 11.732 (reference 11.732) |

Per-generation device peaks, allocator metrics and host RSS are unavailable for this runner.

### Smoke run

[a100-edgedit-flux-klein-base-002__attempt__20261015-170234](../results/runs/a100-edgedit-flux-klein-base-002__attempt__20261015-170234/): complete; skipped 29 in both passes. First 35543.0 ms (text
190 / denoise 34670 / vae 380 / other 310); one
measured 35376.0 ms (120 / 34690 / 280 / 290). Load
2.10 s. Device peak 11.732 GiB. Not a baseline (n = 1).

### Delta vs reference ([a100-edgedit-flux-klein-base-001 baseline](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/), same job and node)

| Metric | base-001 median | 002 median | Δ |
|---|---:|---:|---:|
| wall_ms | 49428.5 | 35435.5 | −13993.0 (−28.3%) |
| denoise | 48740 | 34740 | −14000 (−28.7%) |
| denoise / reference denoise | 1 | 0.713 | computed-pass fraction (100 − k)/100 = 0.71 |
| text_encode / vae_decode / other | 120 / 280 / 290 | 120 / 280 / 300 | 0 / 0 / +10 |
| First generation wall | 49476.0 | 35610.0 | −13866.0 (−28.0%) |
| device_used_peak_gib (whole capture) | 11.732 | 11.732 | 0.000 |

### Repeatability and functional check (visual, not a quality metric)

Final-pass image (run index 13) RGB SHA256 `eaf83530913a90c1…`; the smoke's final image has the same hash, so the output
reproduces across two processes. It differs from the reference final image (`dd6b02c1…`): mean absolute RGB difference
12.2 levels; 85.3% of pixels differ by more than 8 levels in any channel (computed from the two `output.png` files).

Visual check of `output.png` against the reference's: same composition, cat, cardboard sign and pink background as the reference; the background is slightly paler and less saturated, and the script lettering differs in small details (the final "o" of "hello" has a trailing stroke). The sign reads "hello world". No blur or artefact.

## Observations

- The engine skipped 29 of 100 transformer passes in every generation; denoise fell from 48740 to 34740 ms
  (−28.7%) and wall from 49428.5 to 35435.5 ms (−28.3%).
- The measured denoise ratio (0.713) matches the computed-pass fraction ((100 − 29)/100 = 0.71); time per computed pass is 489.3 ms against 487.4 ms in the reference.
- Text encode, VAE decode and other medians are within 10 ms of the reference.
- Whole-process device peak 11.732 GiB (unchanged from the reference capture).
- Output deterministic across the smoke and full processes (final-image hash); 12.2 mean RGB levels from the reference image.

## Interpretation

- At engine defaults on this 50-step CFG schedule, EasyCache skips 29 of 100 transformer passes and the saving is proportional to that count, because the sampling loop is 98.6% of the reference wall and a computed pass costs within 0.5% of a reference pass. The image keeps the reference content and a legible sign but differs in pixels (12.2 mean levels); whether that meets a declared quality requirement needs a quality metric, not this single-image check.
- Hypothesis: EasyCache's default threshold admits reuse mainly where consecutive transformer outputs change slowly (mid-to-late steps of a 50-step schedule); the log does not say which passes were skipped, so this is not tested here.

## Next experiment

- Copy of `a100-edgedit-flux-klein-base-q4-1024-easycache.json` with a paired saved-image quality metric against base-001 (calibration prompts only).
- Runner: record which passes each cache skips (engine debug output), so skipped passes can be located in the schedule.
