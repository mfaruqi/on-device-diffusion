---
type: experiment-record
id: a100-flux-klein-base-torch28-003
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-base-torch28-003__baseline__20261015-160821, a100-flux-klein-base-torch28-003__attempt__20261015-160743]
updated: 2026-10-15
---

# a100-flux-klein-base-torch28-003: diffusers TaylorSeer cache (interval 5, from step 3, order 1), BF16 1024² klein Base, torch 2.8.0; 37 of 50 steps predicted, denoise −63.0%

Status: **complete** (2026-10-15, job 11959986 on gilbreth-g009; full run 20:08:21–20:10:58 UTC, smoke
20:07:43–20:08:20 UTC). The output image changes (different sign lettering) but shows no visible artefact.

## Question

How much denoising time does diffusers' TaylorSeer cache at its default parameters remove from the 50-step CFG klein
Base workload, and how does it change memory and the output? Reference:
[a100-flux-klein-base-torch28-001](a100-flux-klein-base-torch28-001.md), baseline
[`__baseline__20261015-155338`](../results/runs/a100-flux-klein-base-torch28-001__baseline__20261015-155338/).
Step-reuse lever under a quality requirement (RQ2).

## Setup

- Config [a100-flux-klein-base-torch28-bf16-taylorseer.json](../configs/a100-flux-klein-base-torch28-bf16-taylorseer.json);
  smoke [a100-flux-klein-base-torch28-bf16-taylorseer-smoke.json](../configs/a100-flux-klein-base-torch28-bf16-taylorseer-smoke.json).
  The run-directory `config.json` copies equal these files.
- **One change vs [a100-flux-klein-base-torch28-bf16.json](../configs/a100-flux-klein-base-torch28-bf16.json):**
  `optimizations.step_cache: {"method": "taylorseer", "cache_interval": 5, "disable_cache_before_step": 3,
  "max_order": 1}` (the config describes these as the diffusers defaults); otherwise only `id`, `description` and
  `reference_config` differ.
- How it is applied ([scripts/run_flux_portable.py](../scripts/run_flux_portable.py) `apply_step_cache`;
  [definition](../wiki/methods/baseline-metrics.md#portable-pytorch-runner-options)):
  `pipe.transformer.enable_cache(TaylorSeerCacheConfig(...))` once after load; the cache applies to both CFG calls.
- **Receipt limitation.** [step-cache.json](../results/runs/a100-flux-klein-base-torch28-003__baseline__20261015-160821/step-cache.json)
  `full_transformer_computations_per_generation` reads 100 for all 14 generations: it counts pre-hook calls on the last
  single-stream block (2 CFG calls × 50 steps), which TaylorSeer still enters on predicted steps. It **does not
  measure** TaylorSeer's predictions. Computed and predicted steps are identified from `denoise_step_i_ms` instead.
- Workload, protocol (1 + 3 + 10; smoke 1 + 1), device and environment as in
  [base-torch28-001 setup](a100-flux-klein-base-torch28-001.md#setup). Runner sha256 `06f0a949…`. Two transformer
  calls per step (from `stages.csv`; `transformer_calls_per_step` is absent from these `summary.json` copies).
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md). With a step cache, the mean of steps
  1–49 mixes computed and predicted steps; both classes are reported separately below.

## Results

[Run](../results/runs/a100-flux-klein-base-torch28-003__baseline__20261015-160821/),
[summary.json](../results/runs/a100-flux-klein-base-torch28-003__baseline__20261015-160821/summary.json),
[runs.csv](../results/runs/a100-flux-klein-base-torch28-003__baseline__20261015-160821/runs.csv),
[stages.csv](../results/runs/a100-flux-klein-base-torch28-003__baseline__20261015-160821/stages.csv).
`status.json` `complete`.

### Step pattern

In all 14 generations (and the smoke), the same 13 steps take more than 500 ms and the other 37 take 75–80 ms:
computed steps 0, 1, 2, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49; predicted steps 3, 5–8, 10–13, …, 45–48. "Computed" and
"predicted" are inferred from the step times and the configured interval; the runner does not record the decision.

### Latency

| Stage | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| text_encode_ms | 92.8 | 92.2–98.8 | 0.9% |
| denoise_ms | 9672.4 | 9633.6–9678.8 | 96.9% |
| Sum of the 13 computed steps | 6798.5 | – | – |
| Sum of the 37 predicted steps | 2872.6 | – | – |
| Mean computed step (2 calls) | 523.0 | 520.6–523.4 | – |
| Mean predicted step (2 calls) | 77.6 | 77.4–77.7 | – |
| Mean of steps 1–49 per generation | 186.9 | 186.2–187.0 | – |
| vae_decode_ms | 169.5 | 168.6–170.7 | 1.7% |
| postprocess_ms | 33.4 | 32.1–46.7 | 0.3% |
| other_ms | 13.3 | 12.5–14.3 | 0.1% |
| **wall_ms** | **9980.7** | 9940.1–9992.9 (stdev 15.6) | 100% |
| First generation: wall_ms | 10363.1 | – | text 426.5 / denoise 9567.4 / VAE 299.6 / post 44.5 / other 24.7 |
| Warm-ups (3, discarded): wall_ms | – | 9900.1–9932.2 | – |
| Load (cached, s) | 9.77 | – | `from_pretrained` 2.81 + `.to("cuda")` 6.96 |

Computed and predicted sums and means are medians over measured generations of per-generation values from `runs.csv`.

### Memory (GiB)

| Quantity | Value |
|---|---:|
| peak_alloc_gib (all measured equal) | **19.97**, set by `vae_decode` |
| Allocated at stage start: step 0 call 1 / call 2 / step 1 call 1 / call 2 / step 2 call 1 → VAE decode | 14.95 / 15.62 / 16.28 / 16.93 / 17.60 → 17.60 |
| Stage peaks (denoise / VAE decode / postprocess) | 18.16 / 19.97 / 17.61 |
| peak_reserved_gib | 22.36 |
| device_used_peak_gib (NVML, 20 ms) | 23.37 |
| host_rss_gib, measured median | 1.69 (1.69–1.69) |
| host_peak_rss_gib (includes load) | 8.06 |

Allocated memory rises in four steps of about 0.66 GiB across the first four transformer calls, stays at 17.60 GiB
through VAE decode and postprocess, and is back to 14.94 GiB at the next generation's text encode (`stages.csv`).

### Repeatability and functional check (visual, not a quality metric)

All 14 generations have `image_sha256` `36f0b0fda683a670240953474bc280b7f897dee1993b883b87de923e9babf8c0`
(`deterministic_output: true`); the smoke run has the same hash. It differs from base-torch28-001 (`7357668f…`): mean
absolute RGB difference 25.1 levels, 63.2% of pixels differ by more than 8 levels in any channel (computed from the two
`images/measured-0.png` files).

Visual check of `images/measured-0.png` against base-torch28-001's: same scene (tabby cat behind a white sign held in
both paws, blurred indoor background), with small differences in the cat's face and markings. The sign reads
"hello world" in bold lowercase serif letters, where the reference reads "HELLLO WORLD" in thin capitals. No blur,
smear or other visible artefact. Visual only; no quality metric was computed.

### Smoke gate

[`__attempt__20261015-160743`](../results/runs/a100-flux-klein-base-torch28-003__attempt__20261015-160743/):
complete; same 13 computed steps. First 10464.1 ms; one measured 9926.5 ms (text 97.6 / denoise 9603.8 / VAE
169.2 ms). Load 9.12 s. Memory 19.97 / 22.39 / 23.40 GiB. Not a baseline (n = 1).

### Delta vs reference (base-torch28-001 baseline)

| Metric | base-torch28-001 median | base-torch28-003 median | Δ |
|---|---:|---:|---:|
| wall_ms | 26476.6 | 9980.7 | −16495.9 (−62.3%) |
| denoise_ms | 26171.4 | 9672.4 | −16498.9 (−63.0%) |
| Mean of steps 1–49 | 523.6 | 186.9 | −336.7 (−64.3%) |
| text_encode_ms | 93.0 | 92.8 | −0.1 |
| vae_decode_ms | 170.0 | 169.5 | −0.5 |
| First generation wall_ms | 26518.4 | 10363.1 | −16155.3 (−60.9%) |
| peak_alloc / peak_reserved / device_used_peak (GiB) | 17.33 / 19.61 / 20.62 | 19.97 / 22.36 / 23.37 | +2.64 / +2.75 / +2.75 |

## Observations

- 37 of 50 steps take 77.6 ms (both CFG calls) against 523.0 ms for computed steps, which match the reference
  steady per-step time (523.6 ms); denoising falls by 63.0% and wall by 62.3%.
- The step-cache receipt reports 100 full computations per generation, which the step timings contradict; the counter
  does not observe TaylorSeer's predictions (see Setup).
- The allocator peak rises by 2.64 GiB to 19.97 GiB and the device-wide peak by 2.75 GiB to 23.37 GiB; the extra
  allocation builds up over the first four transformer calls and is released between generations.
- The output is deterministic and differs from the reference in pixels (25.1 mean levels); the sign text changes in
  case and typeface, and reads correctly as "hello world", while the reference sign has an extra L.

## Interpretation

- At the default parameters over 50 steps, TaylorSeer computes 13 steps and extrapolates the rest; with transformer
  calls at 98.8% of the reference wall, this cuts end-to-end latency by 62%. The changed sign lettering shows the
  extrapolation alters content detail; whether that is acceptable depends on the declared quality requirement, which
  needs a quality metric rather than this single-image check (the corrected spelling is not evidence of higher quality).
- Hypothesis: the 77.6 ms residual on predicted steps is work outside the cached sub-modules (embeddings, modulation,
  norms, final projection) plus hook overhead for both CFG calls; a per-module profile would separate them.
- Hypothesis: the four 0.66 GiB increments are TaylorSeer state (stored outputs and first-order differences) for the
  two CFG branches, filled during the first computed steps and held until the end of the generation.

## Next experiment

- Runner: record TaylorSeer's compute/predict decisions per step from the hook state instead of the last-block pre-hook.
- Copy of `a100-flux-klein-base-torch28-bf16-taylorseer.json` with a paired saved-image quality metric against base-torch28-001 (calibration prompts only).
