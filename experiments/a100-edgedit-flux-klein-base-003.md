---
type: experiment-record
id: a100-edgedit-flux-klein-base-003
status: complete
device: a100-pcie-40gb
engine: edge-dit-cpp
runs: [a100-edgedit-flux-klein-base-003__baseline__20261015-171344, a100-edgedit-flux-klein-base-003__attempt__20261015-171222]
updated: 2026-10-15
---

# a100-edgedit-flux-klein-base-003: edge-dit.cpp step cache `ucache` (engine defaults), klein Base GGUF Q4_0 1024², 50 steps CFG 4; 26 of 100 transformer passes skipped, denoise −25.7%

Status: **complete** (2026-10-15, job 11960173 on gilbreth-g002; full run 21:13:44–21:22:30 UTC, smoke 21:12:22–21:13:43 UTC).
Full 1+3+10 protocol; the engine skipped 26 transformer passes in every generation.

## Question

Does edge-dit.cpp's `ucache` step cache, at its default parameters, skip or reuse transformer work in the 50-step
CFG klein Base configuration [a100-edgedit-flux-klein-base-001](a100-edgedit-flux-klein-base-001.md), and what does it
change in latency, device memory and output? Single change against
[a100-edgedit-flux-klein-base-q4-1024.json](../configs/a100-edgedit-flux-klein-base-q4-1024.json). Step-reuse lever under
a quality requirement (RQ2).

## Setup

- **Config**: [a100-edgedit-flux-klein-base-q4-1024-ucache.json](../configs/a100-edgedit-flux-klein-base-q4-1024-ucache.json);
  smoke [a100-edgedit-flux-klein-base-q4-1024-ucache-smoke.json](../configs/a100-edgedit-flux-klein-base-q4-1024-ucache-smoke.json)
  (first + one measured). The run-directory `config.json` copies equal these files; `reference_config` is
  `configs/a100-edgedit-flux-klein-base-q4-1024.json`.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "ucache"}`, mapped by
  [scripts/run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) to `--cache_method ucache` with no tuning flags, so
  the engine's per-method defaults apply ([command.json](../results/runs/a100-edgedit-flux-klein-base-003__baseline__20261015-171344/command.json)). The engine log does not
  print the default parameter values.
- Everything else as in the [reference setup](a100-edgedit-flux-klein-base-001.md#setup): edge-dit.cpp `97cc585`,
  `ed-sample` sha256 `fc6b14a7…`; GGUF Q4_0 transformer, Q4_K_M Qwen3 encoder, BF16 VAE, FP32 activations; flash
  attention on; VAE tiling off; 1024×1024, 50 Euler steps, guidance 4.0, CFG 4.0, seed 0, one prompt; four threads;
  runner sha256 `153d3ea9…`. Same job and node as the reference.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`.
  [scripts/gilbreth-gap-base-edgedit-20261015.slurm](../scripts/gilbreth-gap-base-edgedit-20261015.slurm) runs the
  full protocol only when the smoke's `step-cache.json` reports skipped/reused work (here [26, 26]).
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

Full run [a100-edgedit-flux-klein-base-003__baseline__20261015-171344](../results/runs/a100-edgedit-flux-klein-base-003__baseline__20261015-171344/): [status.json](../results/runs/a100-edgedit-flux-klein-base-003__baseline__20261015-171344/status.json) `complete`.

Cache receipt ([engine.log](../results/runs/a100-edgedit-flux-klein-base-003__baseline__20261015-171344/engine.log),
[step-cache.json](../results/runs/a100-edgedit-flux-klein-base-003__baseline__20261015-171344/step-cache.json)): every pass logs `cache enabled: mode=UCache model=flux2-klein granularity=output matched_sites=1 variants=2 slots=1` and
`UCache skipped 26/50 steps (2.08x estimated speedup)` (14 of 14 passes; smoke 2 of 2; source `ucache_policy.cpp`). `skipped_steps_per_generation` is 26 in all 14
generations. The log does not say which passes were skipped. "(2.08x)" is the engine's own estimate (50 / (50 − k)), not a measured speed-up. The engine still
printed 50 `flux step` lines per pass (700 in all).

### Latency (measured n = 10; [summary.json](../results/runs/a100-edgedit-flux-klein-base-003__baseline__20261015-171344/summary.json), [runs.csv](../results/runs/a100-edgedit-flux-klein-base-003__baseline__20261015-171344/runs.csv), [stages.csv](../results/runs/a100-edgedit-flux-klein-base-003__baseline__20261015-171344/stages.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall (1 ms rounding) | **36879.0** | 36875.0–36884.0 (stdev 3.7) | 100% |
| text_encode (engine breakdown) | 120 | 120–120 | 0.3% |
| denoise (engine `sampling`) | 36190 | 36190–36200 | 98.1% |
| vae_decode | 280 | 280–280 | 0.8% |
| other | 290 | 290–290 | 0.8% |
| `[[phase]]` markers: encode_setup / denoise / vae_decode | 405.7 / 36193.4 / 275.7 | 404.5–406.9 / 36186.6–36198.1 / 275.1–277.0 | – |
| First generation: wall | 36992.0 | – | text 170 / denoise 36160 / vae 370 / other 290 |
| Warm-ups (3, discarded): wall | – | 36829.0–36871.0 | – |
| Model load (s) | 2.22 | – | page cache warm |

Denoise per computed pass (derived): 36190 / (100 − 26) = 489.1 ms; reference 48740 / 100 = 487.4 ms.

### Memory (GiB; whole child process, includes load)

| Quantity | Value |
|---|---:|
| device_used_peak_gib (NVML, 20 ms; whole capture) | 11.732 (reference 11.732) |

Per-generation device peaks, allocator metrics and host RSS are unavailable for this runner.

### Smoke run

[a100-edgedit-flux-klein-base-003__attempt__20261015-171222](../results/runs/a100-edgedit-flux-klein-base-003__attempt__20261015-171222/): complete; skipped 26 in both passes. First 37060.0 ms (text
170 / denoise 36230 / vae 370 / other 300); one
measured 36838.0 ms (120 / 36150 / 280 / 290). Load
2.10 s. Device peak 11.732 GiB. Not a baseline (n = 1).

### Delta vs reference ([a100-edgedit-flux-klein-base-001 baseline](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/), same job and node)

| Metric | base-001 median | 003 median | Δ |
|---|---:|---:|---:|
| wall_ms | 49428.5 | 36879.0 | −12549.5 (−25.4%) |
| denoise | 48740 | 36190 | −12550 (−25.7%) |
| denoise / reference denoise | 1 | 0.743 | computed-pass fraction (100 − k)/100 = 0.74 |
| text_encode / vae_decode / other | 120 / 280 / 290 | 120 / 280 / 290 | 0 / 0 / 0 |
| First generation wall | 49476.0 | 36992.0 | −12484.0 (−25.2%) |
| device_used_peak_gib (whole capture) | 11.732 | 11.732 | 0.000 |

### Repeatability and functional check (visual, not a quality metric)

Final-pass image (run index 13) RGB SHA256 `0f3a72e5f0609323…`; the smoke's final image has the same hash, so the output
reproduces across two processes. It differs from the reference final image (`dd6b02c1…`): mean absolute RGB difference
9.0 levels; 67.4% of pixels differ by more than 8 levels in any channel (computed from the two `output.png` files).

Visual check of `output.png` against the reference's: same composition, cat, sign and lettering as the reference; the background is slightly paler. The sign reads "hello world". No blur or artefact.

## Observations

- The engine skipped 26 of 100 transformer passes in every generation; denoise fell from 48740 to 36190 ms
  (−25.7%) and wall from 49428.5 to 36879.0 ms (−25.4%).
- The measured denoise ratio (0.743) matches the computed-pass fraction ((100 − 26)/100 = 0.74); time per computed pass is 489.1 ms against 487.4 ms in the reference.
- Text encode, VAE decode and other medians equal the reference medians.
- Whole-process device peak 11.732 GiB (unchanged from the reference capture).
- Output deterministic across the smoke and full processes (final-image hash); 9.0 mean RGB levels from the reference image.

## Interpretation

- At engine defaults on this 50-step CFG schedule, UCache skips 26 of 100 transformer passes and the saving is proportional to that count, because the sampling loop is 98.6% of the reference wall and a computed pass costs within 0.5% of a reference pass. The image keeps the reference content and a legible sign but differs in pixels (9.0 mean levels); whether that meets a declared quality requirement needs a quality metric, not this single-image check.
- Hypothesis: UCache's default criterion is slightly stricter than EasyCache's at this schedule (fewer skipped passes, smaller pixel difference); the engine does not log the default thresholds, so this is not tested here.

## Next experiment

- Copy of `a100-edgedit-flux-klein-base-q4-1024-ucache.json` with a paired saved-image quality metric against base-001 (calibration prompts only).
