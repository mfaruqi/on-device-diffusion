---
type: experiment-record
id: a100-edgedit-flux-klein-base-005
status: complete
device: a100-pcie-40gb
engine: edge-dit-cpp
runs: [a100-edgedit-flux-klein-base-005__baseline__20261015-173406, a100-edgedit-flux-klein-base-005__attempt__20261015-173255]
updated: 2026-10-15
---

# a100-edgedit-flux-klein-base-005: edge-dit.cpp step cache `taylorseer` (engine defaults), klein Base GGUF Q4_0 1024², 50 steps CFG 4; 38 of 100 transformer passes reused, denoise −37.7%

Status: **complete** (2026-10-15, job 11960173 on gilbreth-g002; full run 21:34:06–21:41:33 UTC, smoke 21:32:55–21:34:06 UTC).
Full 1+3+10 protocol; the engine reused 38 transformer passes in every generation.

## Question

Does edge-dit.cpp's `taylorseer` step cache, at its default parameters, skip or reuse transformer work in the 50-step
CFG klein Base configuration [a100-edgedit-flux-klein-base-001](a100-edgedit-flux-klein-base-001.md), and what does it
change in latency, device memory and output? Single change against
[a100-edgedit-flux-klein-base-q4-1024.json](../configs/a100-edgedit-flux-klein-base-q4-1024.json). Step-reuse lever under
a quality requirement (RQ2).

## Setup

- **Config**: [a100-edgedit-flux-klein-base-q4-1024-taylorseer.json](../configs/a100-edgedit-flux-klein-base-q4-1024-taylorseer.json);
  smoke [a100-edgedit-flux-klein-base-q4-1024-taylorseer-smoke.json](../configs/a100-edgedit-flux-klein-base-q4-1024-taylorseer-smoke.json)
  (first + one measured). The run-directory `config.json` copies equal these files; `reference_config` is
  `configs/a100-edgedit-flux-klein-base-q4-1024.json`.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "taylorseer"}`, mapped by
  [scripts/run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) to `--cache_method taylorseer` with no tuning flags, so
  the engine's per-method defaults apply ([command.json](../results/runs/a100-edgedit-flux-klein-base-005__baseline__20261015-173406/command.json)). The engine log does not
  print the default parameter values.
- Everything else as in the [reference setup](a100-edgedit-flux-klein-base-001.md#setup): edge-dit.cpp `97cc585`,
  `ed-sample` sha256 `fc6b14a7…`; GGUF Q4_0 transformer, Q4_K_M Qwen3 encoder, BF16 VAE, FP32 activations; flash
  attention on; VAE tiling off; 1024×1024, 50 Euler steps, guidance 4.0, CFG 4.0, seed 0, one prompt; four threads;
  runner sha256 `153d3ea9…`. Same job and node as the reference.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`.
  [scripts/gilbreth-gap-base-edgedit-20261015.slurm](../scripts/gilbreth-gap-base-edgedit-20261015.slurm) runs the
  full protocol only when the smoke's `step-cache.json` reports skipped/reused work (here [38, 38]).
- **Cache counts are per transformer pass.** With CFG each step runs two transformer passes (conditional and
  unconditional; the engine reports `variants=2`), 100 per image. The engine's per-generation line
  `<Method> reused k/50 steps` prints the step count as denominator, but k counts passes: MagCache and DiCache print
  k > 50 on this workload ([a100-edgedit-flux-klein-base-007](a100-edgedit-flux-klein-base-007.md),
  [-008](a100-edgedit-flux-klein-base-008.md)), and the measured denoise time scales with (100 − k)/100 (Results).
  `step-cache.json → skipped_steps_per_generation` copies k unchanged.
- Metric derivation: [Jetson edge-dit runner](../wiki/methods/baseline-metrics.md#jetson-edge-dit-runner) (wall from
  `ed-sample` stdout at 1 ms rounding; text / sampling / VAE / other from the engine breakdown at 10 ms rounding; no
  per-step times; device memory is one NVML peak over the whole child process including load; only the final pass's
  image retained).

## Results

Full run [a100-edgedit-flux-klein-base-005__baseline__20261015-173406](../results/runs/a100-edgedit-flux-klein-base-005__baseline__20261015-173406/): [status.json](../results/runs/a100-edgedit-flux-klein-base-005__baseline__20261015-173406/status.json) `complete`.

Cache receipt ([engine.log](../results/runs/a100-edgedit-flux-klein-base-005__baseline__20261015-173406/engine.log),
[step-cache.json](../results/runs/a100-edgedit-flux-klein-base-005__baseline__20261015-173406/step-cache.json)): every pass logs `cache enabled: mode=TaylorSeer model=flux2-klein granularity=feature matched_sites=1 variants=2 slots=1` and
`TaylorSeer reused 38/50 steps (4.17x)` (14 of 14 passes; smoke 2 of 2; source `taylorseer_policy.cpp`). `skipped_steps_per_generation` is 38 in all 14
generations. The log does not say which passes were reused. "(4.17x)" is the engine's own estimate (50 / (50 − k)), not a measured speed-up. The engine still
printed 50 `flux step` lines per pass (700 in all).

### Latency (measured n = 10; [summary.json](../results/runs/a100-edgedit-flux-klein-base-005__baseline__20261015-173406/summary.json), [runs.csv](../results/runs/a100-edgedit-flux-klein-base-005__baseline__20261015-173406/runs.csv), [stages.csv](../results/runs/a100-edgedit-flux-klein-base-005__baseline__20261015-173406/stages.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall (1 ms rounding) | **31128.0** | 31049.0–31372.0 (stdev 108.8) | 100% |
| text_encode (engine breakdown) | 120 | 120–120 | 0.4% |
| denoise (engine `sampling`) | 30370 | 30360–30500 | 97.6% |
| vae_decode | 280 | 270–360 | 0.9% |
| other | 320 | 290–470 | 1.0% |
| `[[phase]]` markers: encode_setup / denoise / vae_decode | 405.4 / 30370.2 / 275.6 | 403.0–411.6 / 30356.0–30504.7 / 274.5–361.6 | – |
| First generation: wall | 31485.0 | – | text 200 / denoise 30600 / vae 370 / other 310 |
| Warm-ups (3, discarded): wall | – | 31161.0–31428.0 | – |
| Model load (s) | 2.48 | – | page cache warm |

Denoise per computed pass (derived): 30370 / (100 − 38) = 489.8 ms; reference 48740 / 100 = 487.4 ms.

### Memory (GiB; whole child process, includes load)

| Quantity | Value |
|---|---:|
| device_used_peak_gib (NVML, 20 ms; whole capture) | 12.206 (reference 11.732) |

Per-generation device peaks, allocator metrics and host RSS are unavailable for this runner.

### Smoke run

[a100-edgedit-flux-klein-base-005__attempt__20261015-173255](../results/runs/a100-edgedit-flux-klein-base-005__attempt__20261015-173255/): complete; reused 38 in both passes. First 31728.0 ms (text
200 / denoise 30410 / vae 600 / other 520); one
measured 31296.0 ms (120 / 30460 / 280 / 440). Load
2.41 s. Device peak 12.206 GiB. Not a baseline (n = 1).

### Delta vs reference ([a100-edgedit-flux-klein-base-001 baseline](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/), same job and node)

| Metric | base-001 median | 005 median | Δ |
|---|---:|---:|---:|
| wall_ms | 49428.5 | 31128.0 | −18300.5 (−37.0%) |
| denoise | 48740 | 30370 | −18370 (−37.7%) |
| denoise / reference denoise | 1 | 0.623 | computed-pass fraction (100 − k)/100 = 0.62 |
| text_encode / vae_decode / other | 120 / 280 / 290 | 120 / 280 / 320 | 0 / 0 / +30 |
| First generation wall | 49476.0 | 31485.0 | −17991.0 (−36.4%) |
| device_used_peak_gib (whole capture) | 11.732 | 12.206 | +0.475 |

### Repeatability and functional check (visual, not a quality metric)

Final-pass image (run index 13) RGB SHA256 `6842dd1476ea9918…`; the smoke's final image has the same hash, so the output
reproduces across two processes. It differs from the reference final image (`dd6b02c1…`): mean absolute RGB difference
3.0 levels; 6.6% of pixels differ by more than 8 levels in any channel (computed from the two `output.png` files).

Visual check of `output.png` against the reference's: indistinguishable from the reference at full resolution: same cat, cardboard sign, script lettering and background. The sign reads "hello world". No blur or artefact.

## Observations

- The engine reused 38 of 100 transformer passes in every generation; denoise fell from 48740 to 30370 ms
  (−37.7%) and wall from 49428.5 to 31128.0 ms (−37.0%).
- The measured denoise ratio (0.623) matches the computed-pass fraction ((100 − 38)/100 = 0.62); time per computed pass is 489.8 ms against 487.4 ms in the reference.
- Text encode, VAE decode and other medians are within 30 ms of the reference.
- Whole-process device peak 12.206 GiB (+0.475 GiB vs the reference capture).
- Output deterministic across the smoke and full processes (final-image hash); 3.0 mean RGB levels from the reference image.
- The measured wall range is 323.0 ms; denoise varies by 140 ms, and VAE decode reaches 360 ms (median 280), other reaches 470 ms (median 320) in at least one measured generation.

## Interpretation

- At engine defaults on this 50-step CFG schedule, TaylorSeer reuses 38 of 100 transformer passes and the saving is proportional to that count, because the sampling loop is 98.6% of the reference wall and a computed pass costs within 0.5% of a reference pass. The image keeps the reference content and a legible sign but differs in pixels (3.0 mean levels); whether that meets a declared quality requirement needs a quality metric, not this single-image check.
- Hypothesis: the +0.475 GiB whole-process device peak is the Taylor-expansion feature state (`granularity=feature`, two variants); a single whole-process peak cannot separate it from allocator variation.

## Next experiment

- Copy of `a100-edgedit-flux-klein-base-q4-1024-taylorseer.json` with a paired saved-image quality metric against base-001 (calibration prompts only).
