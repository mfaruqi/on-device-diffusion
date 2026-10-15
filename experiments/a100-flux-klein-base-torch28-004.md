---
type: experiment-record
id: a100-flux-klein-base-torch28-004
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-base-torch28-004__baseline__20261015-161253, a100-flux-klein-base-torch28-004__attempt__20261015-161059]
updated: 2026-10-15
---

# a100-flux-klein-base-torch28-004: torch.compile (Inductor, mode default) of the transformer forward, BF16 1024² klein Base, torch 2.8.0; denoise −26.6%

Status: **complete** (2026-10-15, job 11959986 on gilbreth-g009; full run 20:12:53–20:17:50 UTC, smoke
20:10:59–20:12:50 UTC). The first generation of each process includes compilation and is reported separately.

## Question

How much does `torch.compile` (default mode) of the transformer forward reduce steady denoising and end-to-end latency
of the 50-step CFG klein Base workload, and what does it cost in the first generation and in memory? Reference:
[a100-flux-klein-base-torch28-001](a100-flux-klein-base-torch28-001.md), baseline
[`__baseline__20261015-155338`](../results/runs/a100-flux-klein-base-torch28-001__baseline__20261015-155338/)
(same environment, node and job). Compiler lever (RQ1).

## Setup

- Config [a100-flux-klein-base-torch28-bf16-compile.json](../configs/a100-flux-klein-base-torch28-bf16-compile.json);
  smoke [a100-flux-klein-base-torch28-bf16-compile-smoke.json](../configs/a100-flux-klein-base-torch28-bf16-compile-smoke.json).
  The run-directory `config.json` copies equal these files.
- **One change vs [a100-flux-klein-base-torch28-bf16.json](../configs/a100-flux-klein-base-torch28-bf16.json):**
  `optimizations.compiler: {"name": "torch_compile", "mode": "default"}`; otherwise only `id`, `description` and
  `reference_config` differ.
- How it is applied ([scripts/run_flux_portable.py](../scripts/run_flux_portable.py) `apply_compiler`;
  [definition](../wiki/methods/baseline-metrics.md#portable-pytorch-runner-options)):
  `pipe.transformer.forward = torch.compile(pipe.transformer.forward, mode="default")` once after load; stage hooks stay
  outside the compiled region; compilation is lazy (first transformer call).
  [compiler.json](../results/runs/a100-flux-klein-base-torch28-004__baseline__20261015-161253/compiler.json):
  `compilations_applied: 1` (counts wrappings, not executed compilations). `TORCHINDUCTOR_COMPILE_THREADS` unset;
  the Inductor cache location and hit counters are not recorded.
- Workload, protocol (1 + 3 + 10; smoke 1 + 1), device and environment as in
  [base-torch28-001 setup](a100-flux-klein-base-torch28-001.md#setup): torch 2.8.0+cu126, Triton 3.4.0, diffusers
  0.40.0. Runner sha256 `06f0a949…`. Two transformer calls per step (from `stages.csv`; `transformer_calls_per_step`
  is absent from these `summary.json` copies).
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md); steady per-step as in base-torch28-001.

## Results

[Run](../results/runs/a100-flux-klein-base-torch28-004__baseline__20261015-161253/),
[summary.json](../results/runs/a100-flux-klein-base-torch28-004__baseline__20261015-161253/summary.json),
[runs.csv](../results/runs/a100-flux-klein-base-torch28-004__baseline__20261015-161253/runs.csv),
[stages.csv](../results/runs/a100-flux-klein-base-torch28-004__baseline__20261015-161253/stages.csv).
`status.json` `complete`.

### Latency

| Stage | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| text_encode_ms | 94.5 | 93.4–98.6 | 0.5% |
| denoise_ms | 19203.5 | 19141.7–19266.8 | 98.4% |
| Steady per-step (mean of steps 1–49) | 384.2 | 383.0–385.5 | – |
| denoise_step_0_ms | 376.3 | – | – |
| vae_decode_ms | 172.3 | 171.6–173.2 | 0.9% |
| postprocess_ms | 37.9 | 33.0–47.3 | 0.2% |
| other_ms | 12.9 | 12.3–14.1 | 0.1% |
| **wall_ms** | **19525.4** | 19455.9–19600.2 (stdev 50.6) | 100% |
| **First generation (includes compilation): wall_ms** | 25021.7 | – | text 411.5 / denoise 24240.8 / VAE 302.8 / post 42.7 / other 23.4 |
| First generation: denoise_step_0 (includes compilation) / mean of steps 1–49 | 5711.0 / 378.2 | – | – |
| Warm-ups (3, discarded): wall_ms | – | 19381.7–19561.9 | – |
| Load (cached, s) | 9.79 | – | `from_pretrained` 2.83 + `.to("cuda")` 6.96 |

No step after step 0 of the first generation exceeds 420 ms in either process, so no recompilation is visible in the
step times.

### Memory (GiB)

| Quantity | Value |
|---|---:|
| peak_alloc_gib (all measured equal) | 17.33, set by `vae_decode` (denoise steps 15.48) |
| peak_reserved_gib | 19.42 |
| device_used_peak_gib (NVML, 20 ms) | 20.44 |
| host_rss_gib, measured median | 1.79 (1.76–1.80) |
| host_peak_rss_gib (includes load) | 8.07 |

### Repeatability and functional check (visual, not a quality metric)

All 14 generations of the full run have `image_sha256`
`f75c8617f8200b47296b6e2a117486fbcc6f6e128efe95bc1bc940802b3a77f7` (`deterministic_output: true`). The smoke process's
two generations share a **different** hash (`992b8557…`), so the compiled output is deterministic within a process but
not identical across the two processes (smoke vs full `measured-0.png`: mean absolute RGB difference 5.2 levels, 20.1%
of pixels differ by more than 8 levels in any channel). Against base-torch28-001 (`7357668f…`): full run 9.8 mean
levels and 38.8% of pixels above 8 levels; smoke 12.6 levels and 46.9%.

Visual check of both `measured-0.png` files against the reference: same composition, cat, white sign and background;
the sign reads "HELLLO WORLD" as in the reference. Differences are small shape changes (ear outline and tip, fur
detail), no blur or artefact. Visually near-identical; no quality metric was computed.

### Smoke gate

[`__attempt__20261015-161059`](../results/runs/a100-flux-klein-base-torch28-004__attempt__20261015-161059/):
complete, `compilations_applied: 1`. First generation 50287.7 ms (denoise 49431.3 ms, of which step 0 31039.8 ms);
one measured 19269.0 ms (text 98.8 / denoise 18948.5 / VAE 171.6 ms). Load 9.05 s. Memory 17.33 / 19.44 / 20.45 GiB.
Not a baseline (n = 1).

### Delta vs reference (base-torch28-001 baseline)

| Metric | base-torch28-001 median | base-torch28-004 median | Δ |
|---|---:|---:|---:|
| wall_ms | 26476.6 | 19525.4 | −6951.2 (−26.3%) |
| denoise_ms | 26171.4 | 19203.5 | −6967.9 (−26.6%) |
| Steady per-step | 523.6 | 384.2 | −139.4 (−26.6%) |
| text_encode_ms | 93.0 | 94.5 | +1.5 |
| vae_decode_ms | 170.0 | 172.3 | +2.3 (+1.4%) |
| postprocess_ms | 28.8 | 37.9 | +9.1 |
| First generation wall_ms (full process) | 26518.4 | 25021.7 | −1496.8 (−5.6%) |
| First generation wall_ms (smoke process) | 26518.4 | 50287.7 | +23769.3 |
| peak_alloc / peak_reserved / device_used_peak (GiB) | 17.33 / 19.61 / 20.62 | 17.33 / 19.42 / 20.44 | 0.00 / −0.19 / −0.19 |
| host_rss_gib | 1.69 | 1.79 | +0.09 |

## Observations

- Compiled steps (two CFG calls) take 384.2 ms against 523.6 ms (factor 1.36); denoising falls by 6967.9 ms and the
  end-to-end median by 6951.2 ms (−26.3%).
- The saving is confined to the transformer calls; text encode and VAE decode are within 2.3 ms of the reference.
- The first generation includes compilation in step 0: 5.71 s in the full process and 31.04 s in the smoke process.
  In the full process the first compiled generation is still 1.50 s faster than the reference first generation,
  because the 49 remaining compiled steps save more than the compilation costs.
- Allocated peak is unchanged; reserved and device-wide peaks are 0.19 GiB lower; host RSS is 0.09 GiB higher.
- The compiled output is deterministic within each process but differs between the smoke and full processes and from
  the eager reference in pixels, with no visible content change.

## Interpretation

- For this workload, Inductor default-mode compilation removes about 27% of transformer time, and because transformer
  calls are 98.8% of the reference wall, nearly all of it reaches end-to-end latency. At the smoke process's cold
  first-generation excess (+23.8 s vs the reference first generation) the steady saving of 6.95 s per generation pays
  back after about 3.4 generations; at the full process's first generation it never costs time.
- Hypothesis: the 31.0 s smoke step 0 is a cold compilation and the 5.7 s full-run step 0 reuses Inductor's on-disk
  cache written by the smoke process (graph capture and guards still run). The cache directory and hit counters were
  not recorded, so the two figures bound the compile cost rather than isolate it.
- Hypothesis: the cross-process hash difference comes from different generated or selected kernels between a cold
  compile and a cache-backed one (reduction order), not from a precision change; recording the Inductor cache state
  per process would test it.

## Next experiment

- Runner: record the Inductor cache directory and hit/miss counters per process, to explain the cross-process output difference.
- Copy of `a100-flux-klein-base-torch28-bf16-compile.json` with `mode: "max-autotune"` (separately labelled; first generation reported separately).
