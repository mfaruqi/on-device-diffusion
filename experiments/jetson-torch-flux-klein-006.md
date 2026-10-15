---
type: experiment-record
id: jetson-torch-flux-klein-006
status: complete
device: jetson-orin-nano
engine: pytorch-diffusers
runs: [jetson-torch-flux-klein-006__baseline__20261015-142456, jetson-torch-flux-klein-006__attempt__20261015-142002]
updated: 2026-10-15
---

# jetson-torch-flux-klein-006: diffusers TaylorSeer cache (interval 2, from step 1, order 1), PyTorch/diffusers klein Q4_0 + NF4, sequential residency, 512×512

Status: **complete** (2026-10-15, run `__baseline__20261015-142456`, 18:24:56–18:45:00 UTC; no Slurm job).
Smoke gate `__attempt__20261015-142002` complete. The output image changes visibly (sign lettering degraded).

## Question

How much denoising time does diffusers' TaylorSeer cache remove from the 4-step distilled klein workload of
[jetson-torch-flux-klein-001](jetson-torch-flux-klein-001.md), and how does it change memory and the output?
Reference: the 001 baseline
[`__baseline__20261014-171410`](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/).
Step-reuse lever under a quality requirement (RQ2).

## Setup

- [Configuration](../configs/jetson-torch-flux-klein-q4-nf4-512-taylorseer.json) and
  [smoke configuration](../configs/jetson-torch-flux-klein-q4-nf4-512-taylorseer-smoke.json); the
  run-directory `config.json` copies equal these files. Reference config:
  [jetson-torch-flux-klein-q4-nf4-512.json](../configs/jetson-torch-flux-klein-q4-nf4-512.json).
- **One change vs the reference:** `optimizations.step_cache: {"method": "taylorseer", "cache_interval": 2,
  "disable_cache_before_step": 1, "max_order": 1}`.
- How it is applied ([scripts/run_flux_portable.py](../scripts/run_flux_portable.py) `apply_step_cache`):
  `transformer.enable_cache(TaylorSeerCacheConfig(cache_interval=2, disable_cache_before_step=1, max_order=1))`
  after load and again on every reloaded transformer (sequential residency reloads it each generation).
  Other `TaylorSeerCacheConfig` fields are at their diffusers 0.40.0 defaults.
- **Receipt limitation.** The runner's [step-cache.json](../results/runs/jetson-torch-flux-klein-006__baseline__20261015-142456/step-cache.json)
  field `full_transformer_computations_per_generation` counts forward-pre-hook calls on the **last single-stream
  block** (`count_full_computations`). TaylorSeer predicts the outputs of sub-modules inside the blocks, so
  the blocks are still entered on predicted steps. The counter reads 4 for all 14 generations and **does not
  measure TaylorSeer's skipping**. Per-step timings (`denoise_step_i_ms`) are used below instead.
- Workload, compressed checkpoints (GGUF Q4_0 transformer, NF4 encoder, BF16 VAE; separately labelled),
  sequential residency (encoder load and transformer reload every generation, inside `other_ms`), protocol
  (1 + 3 + 10; smoke 1 + 1), tegrastats at 200 ms and device/software (torch 2.8.0, diffusers 0.40.0,
  transformers 5.16.1) as in 001
  ([environment.json](../results/runs/jetson-torch-flux-klein-006__baseline__20261015-142456/environment.json)).
  Runner sha256 `44e44608…`. Launched by [scripts/jetson_gap_queue_20261015d.sh](../scripts/jetson_gap_queue_20261015d.sh)
  under the memory guard (400 MB floor, 1536 MB swap-growth limit).
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md). With a step cache, the mean
  of steps 1–3 mixes computed and predicted steps; per-step medians are reported alongside it.

## Results

[Run](../results/runs/jetson-torch-flux-klein-006__baseline__20261015-142456/),
[summary.json](../results/runs/jetson-torch-flux-klein-006__baseline__20261015-142456/summary.json),
[runs.csv](../results/runs/jetson-torch-flux-klein-006__baseline__20261015-142456/runs.csv),
[stages.csv](../results/runs/jetson-torch-flux-klein-006__baseline__20261015-142456/stages.csv),
[stage-residency.json](../results/runs/jetson-torch-flux-klein-006__baseline__20261015-142456/stage-residency.json),
[memguard.json](../results/runs/jetson-torch-flux-klein-006__baseline__20261015-142456/memguard.json).
`status.json` `complete`; guard not triggered (exit 0).

### Latency

| Metric | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall_ms | 77644.2 | 74823.3–79469.1 | – |
| gpu_span_ms | 77640.2 | 74821.6–79467.4 | – |
| text_encode_ms | 1414.9 | 1212.4–2034.7 | 1.8% |
| denoise_ms | 9155.6 | 9121.6–9621.9 | 11.8% |
| denoise_step_0_ms (computed) | 4060.1 | 4034.4–4488.4 | – |
| denoise_step_1_ms (predicted) | 810.4 | 795.3–840.3 | – |
| denoise_step_2_ms (computed) | 3603.6 | 3590.1–3611.2 | – |
| denoise_step_3_ms (predicted) | 684.4 | 681.4–685.9 | – |
| Mean of steps 1–3 per generation | 1698.3 | 1692.7–1711.1 | – |
| vae_decode_ms | 489.1 | 485.8–859.1 | 0.6% |
| postprocess_ms | 44.3 | 40.0–51.1 | 0.1% |
| other_ms (includes encoder load and transformer reload) | 66196.6 | 63778.3–67076.4 | 85.3% |
| First generation: wall_ms | 89896.2 | – | – |
| First generation: text_encode / denoise (steps 0/1/2/3) / vae_decode / other | 5540.7 / 10409.4 (5086.5 / 1038.1 / 3601.3 / 683.4) / 4269.5 / 69496.2 | – | – |
| Warm-ups (3, discarded): wall_ms | – | 77318.1–80173.6 | – |
| Cold load: load_total_s | 81.23 s (`from_pretrained` 80.78 s); cached, not a cold read | – | – |

"Computed"/"predicted" labels follow from the step times (steps 1 and 3 take 0.68–0.84 s, steps 0 and 2
take 3.59–4.49 s) and from the configured interval; the runner does not record which steps were predicted.
Measured `wall_ms` stdev 1683.8 ms. Per-generation residency, measured medians: encoder load 34.142 s,
transformer reload 30.385 s, frees 0.523 / 0.468 s, sum 66.160 s; `other_ms` minus that sum is 30–357 ms
(median 35.5 ms).

### Memory

| Quantity | Value | Source |
|---|---:|---|
| peak_alloc_gib, measured (all 10 equal) | 3.521 GiB, set by `vae_decode` | summary.json, stages.csv |
| peak_reserved_gib, measured | 3.947 GiB | summary.json |
| Allocated at stage start, median (steps 0 / 1 / 2 / 3 / VAE decode) | 2.485 / 2.705 / 2.705 / 2.924 / 2.925 GiB | stages.csv |
| Stage peaks, max (step 0 / 1 / 2 / 3 / VAE decode / text encode) | 3.036 / 3.127 / 3.255 / 3.346 / 3.521 / 3.067 GiB | stages.csv |
| First generation peak alloc / reserved | 3.522 / 3.951 GiB | summary.json |
| host_rss_gib, measured | 1.512 (1.299–1.578) GiB | summary.json |
| host_peak_rss_gib (includes load) | 5.667 GiB | summary.json |
| System RAM peak, whole capture (tegrastats, 5653 samples) | 6.562 / 7.441 GiB | summary.json `system_memory` |
| Swap occupancy, whole capture (tegrastats) | 0.316 → 1.176 GiB | summary.json `system_memory` |
| Guard: MemAvailable start → minimum | 5999 → 668 MB | memguard.json |
| Guard: swap start → maximum | 324 → 1203 MB (growth 879 MB) | memguard.json |

Allocated memory rises by about 0.22 GiB after step 0 and again after step 2 and is still held during VAE
decode and postprocess (2.926–2.929 GiB); the next generation's text encode starts at 2.817 GiB, as in 001,
after the transformer is released.

### Repeatability and functional check (visual, not a quality metric)

All 14 generations have `image_sha256` `d37e4d22eb50313b96bfb50f0154e882e54544502d9bd5fe613a990ba56d2f72`
(`deterministic_output: true`); the smoke run has the same hash. It differs from 001 (`fe94904f…`).
Mean absolute RGB difference from 001's retained image is 28.1 levels, with 85% of pixels differing by more
than 8 levels (computed from the two `measured-0.png` files).

Visual check of `images/measured-0.png` against 001's: same composition (cat centred, sign held in both paws,
indoor background). The fur is darker and more contrasted and the sign is beige/cardboard-coloured instead of
white. The sign text is degraded: thick, partly merged strokes reading roughly "lhollwı", not a recognisable
"hello world" (001 reads "HOLLO WORLD"). Edges around the paws and the background are slightly smeared.

### Smoke gate

[`__attempt__20261015-142002`](../results/runs/jetson-torch-flux-klein-006__attempt__20261015-142002/):
complete. First 84606.2 ms; one measured 82556.4 ms (text 1999.3 / denoise 9385.2 [steps 4251.8 / 745.5 /
3613.6 / 774.3] / VAE 1455.1 / other 69668.9 ms). Load 67.50 s. Peak alloc 3.522 / reserved 3.951 GiB.
Tegrastats RAM peak 6.563 GiB; guard MemAvailable minimum 702 MB. Not a baseline (n = 1).

### Delta vs reference (001 baseline)

| Metric | 001 median (ms) | 006 median (ms) | Δ |
|---|---:|---:|---:|
| wall_ms | 80314.9 | 77644.2 | −2670.7 (−3.3%) |
| text_encode_ms | 1241.3 | 1414.9 | +173.6 (+14.0%) |
| denoise_ms | 14795.5 | 9155.6 | −5639.9 (−38.1%) |
| denoise_step_0 / 1 / 2 / 3_ms | 4015.3 / 3591.2 / 3590.4 / 3590.4 | 4060.1 / 810.4 / 3603.6 / 684.4 | +44.8 / −2780.8 / +13.2 / −2906.0 |
| Mean of steps 1–3 | 3590.8 | 1698.3 | −1892.5 (−52.7%) |
| vae_decode_ms | 491.4 | 489.1 | −2.3 |
| other_ms | 63343.9 | 66196.6 | +2852.7 (+4.5%) |
| First generation wall_ms | 94586.1 | 89896.2 | −4689.9 (−5.0%) |
| peak_alloc_gib / peak_reserved_gib | 3.081 / 3.422 | 3.521 / 3.947 | +0.440 / +0.525 |
| System RAM peak (tegrastats) | 6.114 GiB | 6.562 GiB | +0.448 GiB |
| Guard MemAvailable minimum | 1206 MB | 668 MB | −538 MB |

## Observations

- Steps 1 and 3 take 0.81 s and 0.68 s against 3.59 s for the computed steps; denoising falls by 5.64 s
  (−38.1%).
- Wall time falls only 3.3% because per-generation reloads (`other_ms`, 85.3% of wall) dominate and were
  2.85 s longer at the median than in 001.
- The step-cache receipt reports 4 full computations in every generation, which contradicts the step
  timings; the counter does not observe TaylorSeer's predictions (see Setup).
- The allocator peak rises by 0.44 GiB to 3.521 GiB, held from step 1 to the end of the generation, and the
  guard's MemAvailable minimum falls to 668 MB.
- The output is deterministic across generations but visibly different from 001, with illegible sign text.

## Interpretation

- At 4 steps with interval 2, TaylorSeer predicts two of four steps; this halves the transformer work but
  extrapolates from only one or two computed points, and the visible text degradation shows the
  approximation error at this step count.
- Hypothesis: the 0.68–0.81 s residual on predicted steps is the computation the cache does not replace
  (embeddings, modulation, norms and any non-cached modules inside the blocks) plus hook overhead. A
  per-module profile would separate them.
- Hypothesis: the +0.44 GiB is the TaylorSeer state (stored outputs and first-order differences per cached
  sub-module), allocated after the first computed step and grown after the second, and released with the
  transformer at the next generation's swap.
- Under sequential residency, this lever acts on about 12% of wall time; its effect on generation latency is
  bounded by the reload cost, not by denoising.
- The +173.6 ms text-encode and +2.85 s `other_ms` medians are within the variation of disk-bound reloads seen
  in 001's history; this record does not attribute them to the cache (hypothesis: storage/page-cache state).

## Next experiment

- Copy of `jetson-torch-flux-klein-q4-nf4-512-taylorseer.json` with `conditioning_reuse: true` against [004](jetson-torch-flux-klein-004.md), where denoising dominates.
- Runner: count predicted steps from the TaylorSeer hook state instead of the last-block pre-hook.
