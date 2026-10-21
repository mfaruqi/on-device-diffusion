---
type: experiment-record
id: a100-edgedit-flux-klein-base-006
status: complete
device: a100-pcie-40gb
engine: edge-dit-cpp
runs: [a100-edgedit-flux-klein-base-006__baseline__20261015-174257, a100-edgedit-flux-klein-base-006__attempt__20261015-174134]
updated: 2026-10-15
---

# a100-edgedit-flux-klein-base-006: edge-dit.cpp step cache `cache-dit` (engine defaults), klein Base GGUF Q4_0 1024², 50 steps CFG 4; 24 of 100 transformer passes skipped, denoise −23.8%

Status: **complete** (2026-10-15, job 11960173 on gilbreth-g002; full run 21:42:57–21:51:59 UTC, smoke 21:41:34–21:42:57 UTC).
Full 1+3+10 protocol; the engine skipped 24 transformer passes in every generation.

## Question

Does edge-dit.cpp's `cache-dit` step cache, at its default parameters, skip or reuse transformer work in the 50-step
CFG klein Base configuration [a100-edgedit-flux-klein-base-001](a100-edgedit-flux-klein-base-001.md), and what does it
change in latency, device memory and output? Single change against
[a100-edgedit-flux-klein-base-q4-1024.json](../configs/a100-edgedit-flux-klein-base-q4-1024.json). Step-reuse lever under
a quality requirement (RQ2).

## Setup

- **Config**: [a100-edgedit-flux-klein-base-q4-1024-cache-dit.json](../configs/a100-edgedit-flux-klein-base-q4-1024-cache-dit.json);
  smoke [a100-edgedit-flux-klein-base-q4-1024-cache-dit-smoke.json](../configs/a100-edgedit-flux-klein-base-q4-1024-cache-dit-smoke.json)
  (first + one measured). The run-directory `config.json` copies equal these files; `reference_config` is
  `configs/a100-edgedit-flux-klein-base-q4-1024.json`.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "cache-dit"}`, mapped by
  [scripts/run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) to `--cache_method cache-dit` with no tuning flags, so
  the engine's per-method defaults apply ([command.json](../results/runs/a100-edgedit-flux-klein-base-006__baseline__20261015-174257/command.json)). The engine log does not
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

Full run [a100-edgedit-flux-klein-base-006__baseline__20261015-174257](../results/runs/a100-edgedit-flux-klein-base-006__baseline__20261015-174257/): [status.json](../results/runs/a100-edgedit-flux-klein-base-006__baseline__20261015-174257/status.json) `complete`.

Cache receipt ([engine.log](../results/runs/a100-edgedit-flux-klein-base-006__baseline__20261015-174257/engine.log),
[step-cache.json](../results/runs/a100-edgedit-flux-klein-base-006__baseline__20261015-174257/step-cache.json)): every pass logs `cache enabled: mode=CacheDiT model=flux2-klein granularity=output matched_sites=1 variants=2 slots=1` and
`CacheDiT skipped 24/50 steps (1.92x estimated speedup), accum_diff=1.0272` (pass 1) and `CacheDiT skipped 24/50 steps (1.92x estimated speedup), accum_diff=1.0282` (13 passes) (14 of 14 passes; smoke 2 of 2; source `condition_policy.cpp`). `skipped_steps_per_generation` is 24 in all 14
generations. The log does not say which passes were skipped. "(1.92x)" is the engine's own estimate (50 / (50 − k)), not a measured speed-up. The engine still
printed 50 `flux step` lines per pass (700 in all).

### Latency (measured n = 10; [summary.json](../results/runs/a100-edgedit-flux-klein-base-006__baseline__20261015-174257/summary.json), [runs.csv](../results/runs/a100-edgedit-flux-klein-base-006__baseline__20261015-174257/runs.csv), [stages.csv](../results/runs/a100-edgedit-flux-klein-base-006__baseline__20261015-174257/stages.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall (1 ms rounding) | **37925.0** | 37810.0–38079.0 (stdev 117.5) | 100% |
| text_encode (engine breakdown) | 120 | 120–140 | 0.3% |
| denoise (engine `sampling`) | 37140 | 37130–37360 | 97.9% |
| vae_decode | 280 | 280–440 | 0.7% |
| other | 290 | 280–500 | 0.8% |
| `[[phase]]` markers: encode_setup / denoise / vae_decode | 409.7 / 37141.6 / 276.1 | 400.9–611.7 / 37129.6–37360.3 / 274.4–436.1 | – |
| First generation: wall | 38188.0 | – | text 180 / denoise 37320 / vae 370 / other 310 |
| Warm-ups (3, discarded): wall | – | 37825.0–38023.0 | – |
| Model load (s) | 2.11 | – | page cache warm |

Denoise per computed pass (derived): 37140 / (100 − 24) = 488.7 ms; reference 48740 / 100 = 487.4 ms.

### Memory (GiB; whole child process, includes load)

| Quantity | Value |
|---|---:|
| device_used_peak_gib (NVML, 20 ms; whole capture) | 11.732 (reference 11.732) |

Per-generation device peaks, allocator metrics and host RSS are unavailable for this runner.

### Smoke run

[a100-edgedit-flux-klein-base-006__attempt__20261015-174134](../results/runs/a100-edgedit-flux-klein-base-006__attempt__20261015-174134/): complete; skipped 24 in both passes. First 38131.0 ms (text
200 / denoise 37240 / vae 370 / other 320); one
measured 37970.0 ms (120 / 37110 / 450 / 280). Load
2.16 s. Device peak 11.732 GiB. Not a baseline (n = 1).

### Delta vs reference ([a100-edgedit-flux-klein-base-001 baseline](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/), same job and node)

| Metric | base-001 median | 006 median | Δ |
|---|---:|---:|---:|
| wall_ms | 49428.5 | 37925.0 | −11503.5 (−23.3%) |
| denoise | 48740 | 37140 | −11600 (−23.8%) |
| denoise / reference denoise | 1 | 0.762 | computed-pass fraction (100 − k)/100 = 0.76 |
| text_encode / vae_decode / other | 120 / 280 / 290 | 120 / 280 / 290 | 0 / 0 / 0 |
| First generation wall | 49476.0 | 38188.0 | −11288.0 (−22.8%) |
| device_used_peak_gib (whole capture) | 11.732 | 11.732 | 0.000 |

### Repeatability and functional check (visual, not a quality metric)

Final-pass image (run index 13) RGB SHA256 `c6f2e2be0e0eb5ef…`; the smoke's final image has the same hash, so the output
reproduces across two processes. It differs from the reference final image (`dd6b02c1…`): mean absolute RGB difference
10.6 levels; 69.4% of pixels differ by more than 8 levels in any channel (computed from the two `output.png` files).

Visual check of `output.png` against the reference's: hash-identical to [a100-edgedit-flux-klein-base-004](a100-edgedit-flux-klein-base-004.md) (DBCache): same composition, cat, sign and pink background as the reference, background slightly paler, trailing stroke on the "o" of "hello". The sign reads "hello world". No blur or artefact.

## Observations

- The engine skipped 24 of 100 transformer passes in every generation; denoise fell from 48740 to 37140 ms
  (−23.8%) and wall from 49428.5 to 37925.0 ms (−23.3%).
- The measured denoise ratio (0.762) matches the computed-pass fraction ((100 − 24)/100 = 0.76); time per computed pass is 488.7 ms against 487.4 ms in the reference.
- Text encode, VAE decode and other medians equal the reference medians.
- Whole-process device peak 11.732 GiB (unchanged from the reference capture).
- Output deterministic across the smoke and full processes (final-image hash); 10.6 mean RGB levels from the reference image.
- The measured wall range is 269.0 ms; denoise varies by 230 ms, and VAE decode reaches 440 ms (median 280), other reaches 500 ms (median 290) in at least one measured generation.

## Interpretation

- At engine defaults on this 50-step CFG schedule, cache-dit skips 24 of 100 transformer passes and the saving is proportional to that count, because the sampling loop is 98.6% of the reference wall and a computed pass costs within 0.5% of a reference pass. The image keeps the reference content and a legible sign but differs in pixels (10.6 mean levels); whether that meets a declared quality requirement needs a quality metric, not this single-image check.
- Hypothesis: cache-dit and DBCache share one engine policy (`condition_policy.cpp`) with the same defaults; identical skip counts, `accum_diff` and output hash are consistent with that.

## Next experiment

- None for cache-dit alone; it reproduces [a100-edgedit-flux-klein-base-004](a100-edgedit-flux-klein-base-004.md). Quality follow-up as in that record.
