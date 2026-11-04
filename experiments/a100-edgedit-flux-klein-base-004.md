---
type: experiment-record
id: a100-edgedit-flux-klein-base-004
status: complete
device: a100-pcie-40gb
engine: edge-dit-cpp
runs: [a100-edgedit-flux-klein-base-004__baseline__20261015-172354, a100-edgedit-flux-klein-base-004__attempt__20261015-172231]
updated: 2026-10-15
---

# a100-edgedit-flux-klein-base-004: edge-dit.cpp step cache `dbcache` (engine defaults), klein Base GGUF Q4_0 1024², 50 steps CFG 4; 24 of 100 transformer passes skipped, denoise −23.8%

Status: **complete** (2026-10-15, job 11960173 on gilbreth-g002; full run 21:23:54–21:32:55 UTC, smoke 21:22:31–21:23:54 UTC).
Full 1+3+10 protocol; the engine skipped 24 transformer passes in every generation.

## Question

Does edge-dit.cpp's `dbcache` step cache, at its default parameters, skip or reuse transformer work in the 50-step
CFG klein Base configuration [a100-edgedit-flux-klein-base-001](a100-edgedit-flux-klein-base-001.md), and what does it
change in latency, device memory and output? Single change against
[a100-edgedit-flux-klein-base-q4-1024.json](../configs/a100-edgedit-flux-klein-base-q4-1024.json). Step-reuse lever under
a quality requirement (RQ2).

## Setup

- **Config**: [a100-edgedit-flux-klein-base-q4-1024-dbcache.json](../configs/a100-edgedit-flux-klein-base-q4-1024-dbcache.json);
  smoke [a100-edgedit-flux-klein-base-q4-1024-dbcache-smoke.json](../configs/a100-edgedit-flux-klein-base-q4-1024-dbcache-smoke.json)
  (first + one measured). The run-directory `config.json` copies equal these files; `reference_config` is
  `configs/a100-edgedit-flux-klein-base-q4-1024.json`.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "dbcache"}`, mapped by
  [scripts/run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) to `--cache_method dbcache` with no tuning flags, so
  the engine's per-method defaults apply ([command.json](../results/runs/a100-edgedit-flux-klein-base-004__baseline__20261015-172354/command.json)). The engine log does not
  print the default parameter values.
- Everything else as in the [reference setup](a100-edgedit-flux-klein-base-001.md#setup): edge-dit.cpp `97cc585`,
  `ed-sample` sha256 `fc6b14a7…`; GGUF Q4_0 transformer, Q4_K_M Qwen3 encoder, BF16 VAE, FP32 activations; flash
  attention on; VAE tiling off; 1024×1024, 50 Euler steps, guidance 4.0, CFG 4.0, seed 0, one prompt; four threads;
  runner sha256 `153d3ea9…`. Same job and node as the reference.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`.
  [scripts/gilbreth-gap-base-edgedit-20261015.slurm](../scripts/gilbreth-gap-base-edgedit-20261015.slurm) runs the
  full protocol only when the smoke's `step-cache.json` reports skipped/reused work (here [24, 24]).
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

Full run [a100-edgedit-flux-klein-base-004__baseline__20261015-172354](../results/runs/a100-edgedit-flux-klein-base-004__baseline__20261015-172354/): [status.json](../results/runs/a100-edgedit-flux-klein-base-004__baseline__20261015-172354/status.json) `complete`.

Cache receipt ([engine.log](../results/runs/a100-edgedit-flux-klein-base-004__baseline__20261015-172354/engine.log),
[step-cache.json](../results/runs/a100-edgedit-flux-klein-base-004__baseline__20261015-172354/step-cache.json)): every pass logs `cache enabled: mode=DBCache model=flux2-klein granularity=output matched_sites=1 variants=2 slots=1` and
`DBCache skipped 24/50 steps (1.92x estimated speedup), accum_diff=1.0272` (pass 1) and `DBCache skipped 24/50 steps (1.92x estimated speedup), accum_diff=1.0282` (13 passes) (14 of 14 passes; smoke 2 of 2; source `condition_policy.cpp`). `skipped_steps_per_generation` is 24 in all 14
generations. The log does not say which passes were skipped. "(1.92x)" is the engine's own estimate (50 / (50 − k)), not a measured speed-up. The engine still
printed 50 `flux step` lines per pass (700 in all).

### Latency (measured n = 10; [summary.json](../results/runs/a100-edgedit-flux-klein-base-004__baseline__20261015-172354/summary.json), [runs.csv](../results/runs/a100-edgedit-flux-klein-base-004__baseline__20261015-172354/runs.csv), [stages.csv](../results/runs/a100-edgedit-flux-klein-base-004__baseline__20261015-172354/stages.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall (1 ms rounding) | **37910.0** | 37827.0–38070.0 (stdev 108.9) | 100% |
| text_encode (engine breakdown) | 120 | 120–340 | 0.3% |
| denoise (engine `sampling`) | 37150 | 37140–37310 | 98.0% |
| vae_decode | 280 | 280–340 | 0.7% |
| other | 290 | 280–470 | 0.8% |
| `[[phase]]` markers: encode_setup / denoise / vae_decode | 404.8 / 37147.5 / 276.4 | 399.4–620.5 / 37144.6–37313.7 / 274.9–335.3 | – |
| First generation: wall | 37996.0 | – | text 190 / denoise 37120 / vae 380 / other 310 |
| Warm-ups (3, discarded): wall | – | 37820.0–37924.0 | – |
| Model load (s) | 2.18 | – | page cache warm |

Denoise per computed pass (derived): 37150 / (100 − 24) = 488.8 ms; reference 48740 / 100 = 487.4 ms.

### Memory (GiB; whole child process, includes load)

| Quantity | Value |
|---|---:|
| device_used_peak_gib (NVML, 20 ms; whole capture) | 11.732 (reference 11.732) |

Per-generation device peaks, allocator metrics and host RSS are unavailable for this runner.

### Smoke run

[a100-edgedit-flux-klein-base-004__attempt__20261015-172231](../results/runs/a100-edgedit-flux-klein-base-004__attempt__20261015-172231/): complete; skipped 24 in both passes. First 37973.0 ms (text
170 / denoise 37140 / vae 370 / other 300); one
measured 37798.0 ms (120 / 37110 / 280 / 290). Load
2.11 s. Device peak 11.732 GiB. Not a baseline (n = 1).

### Delta vs reference ([a100-edgedit-flux-klein-base-001 baseline](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/), same job and node)

| Metric | base-001 median | 004 median | Δ |
|---|---:|---:|---:|
| wall_ms | 49428.5 | 37910.0 | −11518.5 (−23.3%) |
| denoise | 48740 | 37150 | −11590 (−23.8%) |
| denoise / reference denoise | 1 | 0.762 | computed-pass fraction (100 − k)/100 = 0.76 |
| text_encode / vae_decode / other | 120 / 280 / 290 | 120 / 280 / 290 | 0 / 0 / 0 |
| First generation wall | 49476.0 | 37996.0 | −11480.0 (−23.2%) |
| device_used_peak_gib (whole capture) | 11.732 | 11.732 | 0.000 |

### Repeatability and functional check (visual, not a quality metric)

Final-pass image (run index 13) RGB SHA256 `c6f2e2be0e0eb5ef…`; the smoke's final image has the same hash, so the output
reproduces across two processes. It differs from the reference final image (`dd6b02c1…`): mean absolute RGB difference
10.6 levels; 69.4% of pixels differ by more than 8 levels in any channel (computed from the two `output.png` files).

Visual check of `output.png` against the reference's: same composition, cat, sign and pink background as the reference; the background is slightly paler and the "o" of "hello" has a trailing stroke. The sign reads "hello world". No blur or artefact.

## Observations

- The engine skipped 24 of 100 transformer passes in every generation; denoise fell from 48740 to 37150 ms
  (−23.8%) and wall from 49428.5 to 37910.0 ms (−23.3%).
- The measured denoise ratio (0.762) matches the computed-pass fraction ((100 − 24)/100 = 0.76); time per computed pass is 488.8 ms against 487.4 ms in the reference.
- Text encode, VAE decode and other medians equal the reference medians.
- Whole-process device peak 11.732 GiB (unchanged from the reference capture).
- Output deterministic across the smoke and full processes (final-image hash); 10.6 mean RGB levels from the reference image.
- The measured wall range is 243.0 ms; denoise varies by 170 ms, and text encode reaches 340 ms (median 120), VAE decode reaches 340 ms (median 280), other reaches 470 ms (median 290) in at least one measured generation.

## Interpretation

- At engine defaults on this 50-step CFG schedule, DBCache skips 24 of 100 transformer passes and the saving is proportional to that count, because the sampling loop is 98.6% of the reference wall and a computed pass costs within 0.5% of a reference pass. The image keeps the reference content and a legible sign but differs in pixels (10.6 mean levels); whether that meets a declared quality requirement needs a quality metric, not this single-image check.
- Hypothesis: DBCache and cache-dit share one engine policy (`condition_policy.cpp`) with the same defaults; the identical counts, `accum_diff` and output hash of [a100-edgedit-flux-klein-base-006](a100-edgedit-flux-klein-base-006.md) are consistent with that.

## Next experiment

- Copy of `a100-edgedit-flux-klein-base-q4-1024-dbcache.json` with a paired saved-image quality metric against base-001 (calibration prompts only).
