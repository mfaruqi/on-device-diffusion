---
type: experiment-record
id: a100-edgedit-flux-klein-007
status: complete
device: a100-pcie-40gb
engine: edge-dit-cpp
runs: [a100-edgedit-flux-klein-007__baseline__20261015-152117, a100-edgedit-flux-klein-007__attempt__20261015-152110]
updated: 2026-10-15
---

# a100-edgedit-flux-klein-007: edge-dit.cpp step cache `magcache` (engine defaults), BF16 1024² klein; 3 of 4 steps reused, denoise −73.5%, output visibly degraded

Status: **complete** (2026-10-15, job 11959806 on gilbreth-g008; full run 19:21:17–19:21:36 UTC, smoke
19:21:10–19:21:17 UTC). Full 1+3+10 protocol. The engine reused 3 of 4 steps in every pass; the output image is
visibly degraded.

## Question

Does edge-dit.cpp's MagCache step cache, at its default parameters, reuse any of the 4 denoising steps of the A100
no-cache configuration [a100-edgedit-flux-klein-001](a100-edgedit-flux-klein-001.md), and what does it change in
latency, memory and output? Single change against
[a100-edgedit-flux-klein-bf16.json](../configs/a100-edgedit-flux-klein-bf16.json). Step-reuse lever under a
quality requirement (RQ2).

## Setup

- **Config**: [a100-edgedit-flux-klein-bf16-magcache.json](../configs/a100-edgedit-flux-klein-bf16-magcache.json);
  smoke [a100-edgedit-flux-klein-bf16-magcache-smoke.json](../configs/a100-edgedit-flux-klein-bf16-magcache-smoke.json)
  (first + one measured). The run-directory `config.json` copies equal these files; `reference_config` is
  `configs/a100-edgedit-flux-klein-bf16.json`.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "magcache"}`, mapped by
  [scripts/run_edgedit.py](../scripts/run_edgedit.py) to `--cache_method magcache` with no tuning flags, so the
  engine's per-method defaults apply ([command.json](../results/runs/a100-edgedit-flux-klein-007__baseline__20261015-152117/command.json)).
  The engine log does not print the default parameter values.
- Everything else as in the [reference](a100-edgedit-flux-klein-001.md#full-repeated-baseline-2026-10-06):
  edge-dit.cpp `97cc585`, CUDA performance build, same `ed-sample` binary (sha256 `fc6b14a7…`); BF16 weights with
  engine-reported FP32 transformer activations; flash attention on; VAE tiling off; 1024×1024, 4 Euler steps,
  guidance 1.0, CFG 1.0, seed 0, one prompt; four threads.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke
  `--repeat 2`. Launched by [scripts/gilbreth-gap-20261015.slurm](../scripts/gilbreth-gap-20261015.slurm), which
  runs the full protocol only when the smoke's `step-cache.json` reports skipped/reused steps (here [3, 3]).
  The runner script hash is not recorded in the run directory.
- The reference baseline ran on 2026-10-05 in a different Slurm job (11891242) on a different node
  (gilbreth-g007); this run used gilbreth-g008.
- Metric derivation: [edge-dit adapter](../wiki/methods/baseline-metrics.md#edge-dit-ed-sample-adapter) (wall from
  `ed-sample` stdout at 1 ms rounding; `encode_setup` / `denoise` / `vae_decode` from host `system_clock` phase
  markers, not isolated text encoding; per-step times unavailable; device memory is one NVML peak over the whole
  child process; only the final pass's image retained).

## Results

Full run [a100-edgedit-flux-klein-007__baseline__20261015-152117](../results/runs/a100-edgedit-flux-klein-007__baseline__20261015-152117/):
[status.json](../results/runs/a100-edgedit-flux-klein-007__baseline__20261015-152117/status.json) `complete`.

Cache receipt ([engine.log](../results/runs/a100-edgedit-flux-klein-007__baseline__20261015-152117/engine.log),
[step-cache.json](../results/runs/a100-edgedit-flux-klein-007__baseline__20261015-152117/step-cache.json)): every
pass logs `cache enabled: mode=MagCache model=flux2-klein granularity=feature matched_sites=1 variants=2 slots=1`
and `MagCache reused 3/4 steps (4.00x)` (14 of 14 passes; smoke 2 of 2); `skipped_steps_per_generation` is 3 in all
14 generations. The log does not say which three steps were reused; "(4.00x)" is the engine's own ratio, not a
measured speed-up.

### Latency (measured n = 10; [summary.json](../results/runs/a100-edgedit-flux-klein-007__baseline__20261015-152117/summary.json), [runs.csv](../results/runs/a100-edgedit-flux-klein-007__baseline__20261015-152117/runs.csv), [stages.csv](../results/runs/a100-edgedit-flux-klein-007__baseline__20261015-152117/stages.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall (1 ms rounding) | 825.5 | 821.0–833.0 (stdev 3.9) | 100% |
| encode_setup (phase markers) | 257.2 | 256.5–266.6 | 31.2% |
| denoise (sampling loop, 4 steps) | 290.1 | 285.0–291.4 | 35.1% |
| vae_decode | 272.5 | 270.8–276.5 | 33.0% |
| First generation: wall | 1167.0 | – | encode_setup 349.8 / denoise 439.1 / vae 371.9 |
| Warm-ups (3, discarded): wall | – | 824.0–830.0 | – |
| Model load (s) | 3.62 | – | filesystem cache may be warm |

The remainder of wall not covered by the three phases is about 5 ms at the median.

### Memory (GiB; whole child process, includes load)

| Quantity | Value |
|---|---:|
| capture_device_used_peak_gib (NVML, 20 ms) | 21.847 |
| host_peak_rss_gib | 2.725 |

Per-generation device peaks and allocator metrics are unavailable for this engine.

### Smoke run

[a100-edgedit-flux-klein-007__attempt__20261015-152110](../results/runs/a100-edgedit-flux-klein-007__attempt__20261015-152110/):
complete; reused 3/4 in both passes. First 1187.0 ms (encode_setup 351.1 / denoise 460.1 / vae 369.8); one measured
830.0 ms (258.4 / 293.4 / 272.1). Load 3.20 s. Device peak 21.847 GiB. Not a baseline (n = 1).

### Delta vs reference ([a100-edgedit-flux-klein-001 baseline](../results/runs/a100-edgedit-flux-klein-001__baseline__20261005-233237/), different job and node)

| Metric | 001 median | 007 median | Δ |
|---|---:|---:|---:|
| wall_ms | 1641.0 | 825.5 | −815.5 (−49.7%) |
| denoise | 1096.2 | 290.1 | −806.0 (−73.5%) |
| encode_setup | 269.6 | 257.2 | −12.4 |
| vae_decode | 269.9 | 272.5 | +2.7 (+1.0%) |
| First generation wall | 2066.0 | 1167.0 | −899.0 (−43.5%) |
| capture_device_used_peak_gib | 21.607 | 21.847 | +0.240 |
| host_peak_rss_gib | 2.740 | 2.725 | −0.015 |

### Repeatability and functional check (visual, not a quality metric)

Final-pass image (run index 13) RGB SHA256 `725fdd49fad8e43f335d0efdc3755b75f7ee59a804d96c2b02c3f9b8bb40219f`; the
smoke's final image has the same hash, so the output reproduces across two processes. It differs from the
reference's recorded final hash (`c476fc32…`, [validation.json](../results/runs/a100-edgedit-flux-klein-001__baseline__20261005-233237/validation.json)).
The reference PNG is not retained in the local reference run directory; the visual and pixel comparison below uses
the `output.png` of the zero-reuse screen
[a100-edgedit-flux-klein-002](../results/runs/a100-edgedit-flux-klein-002__attempt__20261015-152019/), whose RGB
hash equals the reference hash. Mean absolute RGB difference 34.5 levels; 88.9% of pixels differ by more than 8
levels in any channel.

Visual check of `output.png`: same scene type (tabby cat behind a white sign held in both paws), but the cat is
lighter and greyer with soft, smeared fur and a blurred face; the paws are blurry; the background is a featureless
grey-beige blur. The sign text is garbled: thick, smeared, overlapping strokes, roughly "hwUlnor", not "hello world"
(reference image: a sharp, legible "hello world"). The image is visibly degraded.

## Observations

- MagCache reused 3 of 4 steps in every pass; the sampling loop fell from 1096.2 to 290.1 ms (−73.5%) and wall
  from 1641.0 to 825.5 ms (−49.7%).
- The remaining denoise time is about one quarter of the reference loop (1096.2 / 4 = 274.0 ms) plus about 16 ms.
- `encode_setup` and VAE decode are within 13 ms of the reference; together they are now 64% of wall.
- The output is visibly degraded (blur, garbled sign) and differs from the reference by 34.5 mean RGB levels.
- The whole-process device peak is 0.24 GiB above the reference capture (single captures, including load).

## Interpretation

- With only 4 steps, reusing 3 leaves a single computed transformer evaluation; the latency saving is close to
  the maximum possible for the sampling loop, and the image shows that one evaluation plus three magnitude-based
  reuses does not reproduce the reference output at this step count.
- Under any quality requirement that includes legible text or sharp detail, this default configuration does not
  qualify; it serves as a latency bound for step reuse on this configuration.
- Hypothesis: MagCache's default magnitude-ratio threshold is calibrated for longer schedules, where consecutive
  steps differ less; on a 4-step distilled schedule the per-step changes are large, so the reuse error is large.
- Hypothesis: the +0.24 GiB device peak is the cached feature slot; a single whole-process peak cannot separate
  it from run-to-run and node variation.

## Next experiment

- Copy of `a100-edgedit-flux-klein-bf16-magcache.json` with a stricter `--cache-threshold` that reuses at most one step (needs a runner option; separately labelled), with a saved-image comparison against the no-cache output.
