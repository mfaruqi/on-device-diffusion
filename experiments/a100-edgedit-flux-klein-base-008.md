---
type: experiment-record
id: a100-edgedit-flux-klein-base-008
status: complete
device: a100-pcie-40gb
engine: edge-dit-cpp
runs: [a100-edgedit-flux-klein-base-008__baseline__20261015-175727, a100-edgedit-flux-klein-base-008__attempt__20261015-175629]
updated: 2026-10-15
---

# a100-edgedit-flux-klein-base-008: edge-dit.cpp step cache `dicache` (engine defaults), klein Base GGUF Q4_0 1024², 50 steps CFG 4; 54 of 100 transformer passes reused, denoise −49.9%

Status: **complete** (2026-10-15, job 11960173 on gilbreth-g002; full run 21:57:27–22:03:29 UTC, smoke 21:56:29–21:57:26 UTC).
Full 1+3+10 protocol; the engine reused 54 transformer passes in every generation.

## Question

Does edge-dit.cpp's `dicache` step cache, at its default parameters, skip or reuse transformer work in the 50-step
CFG klein Base configuration [a100-edgedit-flux-klein-base-001](a100-edgedit-flux-klein-base-001.md), and what does it
change in latency, device memory and output? Single change against
[a100-edgedit-flux-klein-base-q4-1024.json](../configs/a100-edgedit-flux-klein-base-q4-1024.json). Step-reuse lever under
a quality requirement (RQ2).

## Setup

- **Config**: [a100-edgedit-flux-klein-base-q4-1024-dicache.json](../configs/a100-edgedit-flux-klein-base-q4-1024-dicache.json);
  smoke [a100-edgedit-flux-klein-base-q4-1024-dicache-smoke.json](../configs/a100-edgedit-flux-klein-base-q4-1024-dicache-smoke.json)
  (first + one measured). The run-directory `config.json` copies equal these files; `reference_config` is
  `configs/a100-edgedit-flux-klein-base-q4-1024.json`.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "dicache"}`, mapped by
  [scripts/run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) to `--cache_method dicache` with no tuning flags, so
  the engine's per-method defaults apply ([command.json](../results/runs/a100-edgedit-flux-klein-base-008__baseline__20261015-175727/command.json)). The engine log does not
  print the default parameter values.
- Everything else as in the [reference setup](a100-edgedit-flux-klein-base-001.md#setup): edge-dit.cpp `97cc585`,
  `ed-sample` sha256 `fc6b14a7…`; GGUF Q4_0 transformer, Q4_K_M Qwen3 encoder, BF16 VAE, FP32 activations; flash
  attention on; VAE tiling off; 1024×1024, 50 Euler steps, guidance 4.0, CFG 4.0, seed 0, one prompt; four threads;
  runner sha256 `153d3ea9…`. Same job and node as the reference.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`.
  [scripts/gilbreth-gap-base-edgedit-20261015.slurm](../scripts/gilbreth-gap-base-edgedit-20261015.slurm) runs the
  full protocol only when the smoke's `step-cache.json` reports skipped/reused work (here [54, 54]).
- **Cache counts are per transformer pass.** With CFG each step runs two transformer passes (conditional and
  unconditional), 100 per image; the other caches report `variants=2`, DiCache reports `variants=3` and `slots=4`. The engine's per-generation line
  `<Method> reused k/50 steps` prints the step count as denominator, but k counts passes: this run prints
  k = 54 > 50 (as does MagCache, [a100-edgedit-flux-klein-base-007](a100-edgedit-flux-klein-base-007.md)); for the other caches the measured denoise time scales with (100 − k)/100.
  `step-cache.json → skipped_steps_per_generation` copies k unchanged.
- Metric derivation: [Jetson edge-dit runner](../wiki/methods/baseline-metrics.md#jetson-edge-dit-runner) (wall from
  `ed-sample` stdout at 1 ms rounding; text / sampling / VAE / other from the engine breakdown at 10 ms rounding; no
  per-step times; device memory is one NVML peak over the whole child process including load; only the final pass's
  image retained).

## Results

Full run [a100-edgedit-flux-klein-base-008__baseline__20261015-175727](../results/runs/a100-edgedit-flux-klein-base-008__baseline__20261015-175727/): [status.json](../results/runs/a100-edgedit-flux-klein-base-008__baseline__20261015-175727/status.json) `complete`.

Cache receipt ([engine.log](../results/runs/a100-edgedit-flux-klein-base-008__baseline__20261015-175727/engine.log),
[step-cache.json](../results/runs/a100-edgedit-flux-klein-base-008__baseline__20261015-175727/step-cache.json)): every pass logs `cache enabled: mode=DiCache model=flux2-klein granularity=probe matched_sites=2 variants=3 slots=4` and
`DiCache reused 54/50 steps` (14 of 14 passes; smoke 2 of 2; source `dicache_policy.cpp`). `skipped_steps_per_generation` is 54 in all 14
generations. The log does not say which passes were reused. The engine still
printed 50 `flux step` lines per pass (700 in all).

### Latency (measured n = 10; [summary.json](../results/runs/a100-edgedit-flux-klein-base-008__baseline__20261015-175727/summary.json), [runs.csv](../results/runs/a100-edgedit-flux-klein-base-008__baseline__20261015-175727/runs.csv), [stages.csv](../results/runs/a100-edgedit-flux-klein-base-008__baseline__20261015-175727/stages.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall (1 ms rounding) | **25154.5** | 25023.0–25462.0 (stdev 138.6) | 100% |
| text_encode (engine breakdown) | 120 | 120–120 | 0.5% |
| denoise (engine `sampling`) | 24400 | 24340–24500 | 97.0% |
| vae_decode | 280 | 280–370 | 1.1% |
| other | 290 | 290–570 | 1.2% |
| `[[phase]]` markers: encode_setup / denoise / vae_decode | 404.0 / 24399.0 / 275.9 | 400.8–687.3 / 24337.7–24498.7 / 274.8–365.8 | – |
| First generation: wall | 25291.0 | – | text 190 / denoise 24390 / vae 380 / other 330 |
| Warm-ups (3, discarded): wall | – | 24953.0–25352.0 | – |
| Model load (s) | 2.23 | – | page cache warm |

Denoise per computed pass (derived): 24400 / (100 − 54) = 530.4 ms; reference 48740 / 100 = 487.4 ms.

### Memory (GiB; whole child process, includes load)

| Quantity | Value |
|---|---:|
| device_used_peak_gib (NVML, 20 ms; whole capture) | 12.630 (reference 11.732) |

Per-generation device peaks, allocator metrics and host RSS are unavailable for this runner.

### Smoke run

[a100-edgedit-flux-klein-base-008__attempt__20261015-175629](../results/runs/a100-edgedit-flux-klein-base-008__attempt__20261015-175629/): complete; reused 54 in both passes. First 25486.0 ms (text
170 / denoise 24550 / vae 370 / other 400); one
measured 24993.0 ms (120 / 24300 / 280 / 290). Load
2.10 s. Device peak 12.630 GiB. Not a baseline (n = 1).

### Delta vs reference ([a100-edgedit-flux-klein-base-001 baseline](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/), same job and node)

| Metric | base-001 median | 008 median | Δ |
|---|---:|---:|---:|
| wall_ms | 49428.5 | 25154.5 | −24274.0 (−49.1%) |
| denoise | 48740 | 24400 | −24340 (−49.9%) |
| denoise / reference denoise | 1 | 0.501 | computed-pass fraction (100 − k)/100 = 0.46 |
| text_encode / vae_decode / other | 120 / 280 / 290 | 120 / 280 / 290 | 0 / 0 / 0 |
| First generation wall | 49476.0 | 25291.0 | −24185.0 (−48.9%) |
| device_used_peak_gib (whole capture) | 11.732 | 12.630 | +0.898 |

### Repeatability and functional check (visual, not a quality metric)

Final-pass image (run index 13) RGB SHA256 `4e716bdfaeffe989…`; the smoke's final image has the same hash, so the output
reproduces across two processes. It differs from the reference final image (`dd6b02c1…`): mean absolute RGB difference
4.1 levels; 10.2% of pixels differ by more than 8 levels in any channel (computed from the two `output.png` files).

Visual check of `output.png` against the reference's: near-identical to the reference: same cat, sign, script lettering and background. The sign reads "hello world". No blur or artefact.

## Observations

- The engine reused 54 of 100 transformer passes in every generation; denoise fell from 48740 to 24400 ms
  (−49.9%) and wall from 49428.5 to 25154.5 ms (−49.1%).
- The measured denoise ratio (0.501) is above the computed-pass fraction ((100 − 54)/100 = 0.46); time per computed pass is 530.4 ms against 487.4 ms in the reference.
- Text encode, VAE decode and other medians equal the reference medians.
- Whole-process device peak 12.630 GiB (+0.898 GiB vs the reference capture).
- Output deterministic across the smoke and full processes (final-image hash); 4.1 mean RGB levels from the reference image.
- The measured wall range is 439.0 ms; denoise varies by 160 ms, and VAE decode reaches 370 ms (median 280), other reaches 570 ms (median 290) in at least one measured generation.

## Interpretation

- At engine defaults, DiCache reuses 54 of 100 transformer passes, but the saving (denoise −49.9%) is smaller than the 54% pass reduction because each computed pass costs about 9% more than in the reference. The image stays close to the reference (4.1 mean levels, 10.2% of pixels above 8 levels; legible sign).
- Hypothesis: the higher time per computed pass (530 vs 487 ms) and the +0.898 GiB device peak come from DiCache's probe (`granularity=probe`, 2 matched sites, 4 slots), which runs extra shallow computation and keeps per-site state; per-pass timing is not available to confirm it.

## Next experiment

- Copy of `a100-edgedit-flux-klein-base-q4-1024-dicache.json` with a paired saved-image quality metric against base-001 (calibration prompts only).
