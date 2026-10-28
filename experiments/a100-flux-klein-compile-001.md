---
type: experiment-record
id: a100-flux-klein-compile-001
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-compile-001__baseline__20261015-151740, a100-flux-klein-compile-001__attempt__20261015-151623]
updated: 2026-10-15
---

# a100-flux-klein-compile-001: torch.compile (Inductor, mode default) of the transformer forward, BF16 1024² klein, torch 2.8.0

Status: **complete** (2026-10-15, job 11959806 on gilbreth-g008; full run 19:17:40–19:18:18 UTC, smoke
19:16:23–19:17:38 UTC). The first generation of each process includes compilation and is reported separately.

## Question

How much does `torch.compile` (default mode) of the transformer forward reduce steady denoising and
end-to-end latency of the BF16 reference workload, and what does it cost in the first generation and in
memory? Reference: [a100-flux-klein-torch28-001](a100-flux-klein-torch28-001.md), baseline
[`__baseline__20261015-151548`](../results/runs/a100-flux-klein-torch28-001__baseline__20261015-151548/)
(same environment, node and job). Compiler lever (RQ1).

## Setup

- Config [a100-flux-klein-torch28-bf16-compile.json](../configs/a100-flux-klein-torch28-bf16-compile.json);
  smoke [a100-flux-klein-torch28-bf16-compile-smoke.json](../configs/a100-flux-klein-torch28-bf16-compile-smoke.json).
  The run-directory `config.json` copies equal these files.
- **One change vs [a100-flux-klein-torch28-bf16.json](../configs/a100-flux-klein-torch28-bf16.json):**
  `optimizations.compiler: {"name": "torch_compile", "mode": "default"}`; otherwise only `id` and `description`
  differ. (The config's `reference_config` field still names `configs/a100-flux-klein-bf16.resolved.json`,
  inherited from the torch28 reference config; its `description` names torch28-001 as the reference.)
- How it is applied ([scripts/run_flux_portable.py](../scripts/run_flux_portable.py) `apply_compiler`):
  `pipe.transformer.forward = torch.compile(pipe.transformer.forward, mode="default")` once after load; the
  stage hooks stay outside the compiled region. Compilation is lazy (first transformer call).
  [compiler.json](../results/runs/a100-flux-klein-compile-001__baseline__20261015-151740/compiler.json):
  `compilations_applied: 1`. `TORCHINDUCTOR_COMPILE_THREADS` unset (Inductor default). The Inductor on-disk
  cache location was not set by the runner and is not recorded.
- Workload, protocol (1 + 3 + 10; smoke 1 + 1), device and environment as in
  [torch28-001 setup](a100-flux-klein-torch28-001.md#setup): BF16, 1024×1024, 4 steps, guidance 1.0, seed 0;
  torch 2.8.0+cu126, Triton 3.4.0 (not recorded in `environment.json`), diffusers 0.40.0, transformers 5.16.1.
  Runner sha256 `44e44608…`.
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md); steady per-step as in torch28-001.

## Results

[Run](../results/runs/a100-flux-klein-compile-001__baseline__20261015-151740/),
[summary.json](../results/runs/a100-flux-klein-compile-001__baseline__20261015-151740/summary.json),
[runs.csv](../results/runs/a100-flux-klein-compile-001__baseline__20261015-151740/runs.csv),
[stages.csv](../results/runs/a100-flux-klein-compile-001__baseline__20261015-151740/stages.csv).
`status.json` `complete`.

### Latency

| Stage | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| text_encode_ms | 46.3 | 46.1–47.9 | 4.5% |
| denoise_ms | 758.1 | 754.2–759.5 | 74.2% |
| denoise_step_0 / 1 / 2 / 3_ms | 185.6 / 190.3 / 191.0 / 190.7 | 184.4–191.8 (all four) | – |
| Steady per-step (mean of steps 1–3) | 190.8 | 189.7–191.0 | – |
| vae_decode_ms | 171.7 | 170.4–172.9 | 16.8% |
| postprocess_ms | 37.9 | 26.8–41.0 | 3.7% |
| other_ms | 7.9 | 7.5–8.8 | 0.8% |
| **wall_ms** | **1021.9** | 1011.3–1023.8 (stdev 4.0) | 100% |
| **First generation (includes compilation): wall_ms** | 6790.3 | – | text 385.0 / denoise 6029.8 / VAE 304.7 / post 50.2 / other 20.5 |
| First generation: denoise_step_0 (includes compilation) / steps 1–3 | 5466.7 / 184.7, 189.1, 189.3 | – | – |
| Warm-ups (3, discarded): wall_ms | – | 1017.1–1024.4 (step 0: 184.5–185.0) | – |
| Load (cached, s) | 8.80 | – | `from_pretrained` 2.76 + `.to("cuda")` 6.04 |

### Memory (GiB)

| Quantity | Value |
|---|---:|
| peak_alloc_gib (all measured equal) | 17.32, set by `vae_decode` (each denoise step 15.48) |
| peak_reserved_gib | 19.42 |
| device_used_peak_gib (NVML, 20 ms) | 20.44 |
| host_rss_gib, measured median | 1.80 (1.80–1.80) |
| host_peak_rss_gib (includes load) | 8.07 |

### Repeatability and functional check (visual, not a quality metric)

All 14 generations have `image_sha256` `cd3a65a19f1564370da0132b31464736d8a62cc7c8654cdf333570c37ebe323d`
(`deterministic_output: true`); the smoke run has the same hash, so the compiled output reproduces across the
two processes. It differs from torch28-001 (`f11220f6…`): mean absolute RGB difference 2.3 levels, 5.5% of
pixels differ by more than 8 levels in any channel (computed from the two `images/measured-0.png` files).
Visual check against torch28-001's image: composition, cat, sign and background look the same; the sign reads
"hellword" as in the reference. No visible difference was found at full resolution.

### Smoke gate

[`__attempt__20261015-151623`](../results/runs/a100-flux-klein-compile-001__attempt__20261015-151623/):
complete, `compilations_applied: 1`. First generation 32155.9 ms (denoise 31399.6 ms, of which step 0
30826.5 ms); one measured 1024.2 ms (denoise 755.3, steady per-step 190.1, VAE 168.6 ms). Load 8.81 s. Memory
17.32 / 19.44 / 20.45 GiB. Not a baseline (n = 1).

### Delta vs reference (torch28-001 baseline)

| Metric | torch28-001 median | compile-001 median | Δ |
|---|---:|---:|---:|
| wall_ms | 1290.5 | 1021.9 | −268.6 (−20.8%) |
| denoise_ms | 1037.8 | 758.1 | −279.6 (−26.9%) |
| Steady per-step | 260.3 | 190.8 | −69.5 (−26.7%) |
| denoise_step_0_ms | 255.6 | 185.6 | −70.0 (−27.4%) |
| text_encode_ms | 46.2 | 46.3 | +0.1 |
| vae_decode_ms | 168.2 | 171.7 | +3.5 (+2.1%) |
| postprocess_ms | 29.8 | 37.9 | +8.2 |
| other_ms | 7.5 | 7.9 | +0.4 |
| First generation wall_ms | 1830.9 | 6790.3 | +4959.4 (+270.9%) |
| peak_alloc / peak_reserved / device_used_peak (GiB) | 17.32 / 19.61 / 20.62 | 17.32 / 19.42 / 20.44 | 0.00 / −0.18 / −0.18 |
| host_rss_gib | 1.69 | 1.80 | +0.12 |

## Observations

- Compiled transformer steps take 190.8 ms against 260.3 ms (factor 1.36); denoising falls by 279.6 ms and
  the end-to-end median by 268.6 ms (−20.8%).
- The saving is confined to the transformer steps; text encode, VAE decode and `other_ms` are within a few ms
  of the reference (VAE +3.5 ms). Postprocess (host-bound) varies most.
- The step 0 → 3 rise seen in the reference (255.6 → 266.0 ms) is absent: compiled steps are flat at 185–191 ms.
- The first generation costs 6.8 s in the full process and 32.2 s in the smoke process; step 0 holds almost
  all of the excess in both (5.47 s and 30.83 s).
- Allocated peak is unchanged; reserved and device-wide peaks are 0.18 GiB lower; host RSS is 0.12 GiB higher.
- The compiled output differs slightly from the eager output in pixels (2.3 mean levels), with no visible
  change, and is deterministic within and across the two processes.

## Interpretation

- For this workload, Inductor default-mode compilation of the transformer removes about 27% of transformer
  time. Steady-state saving is 268.6 ms per generation; at the full process's first-generation excess
  (+4959.4 ms vs the reference first generation) it pays back after about 18.5 generations.
- Hypothesis: the 30.8 s smoke step 0 is a cold compilation, and the 5.5 s full-run step 0 reuses Inductor's
  on-disk cache written by the smoke process 2 s earlier (graph capture and guards still run). The cache
  directory and hit counters were not recorded, so the two first-generation figures bound the compile cost
  rather than isolate it.
- Hypothesis: the gain comes from fusing elementwise, normalization and cast kernels in each step; no
  profile was taken in this run, so a kernel profile of one compiled and one eager step would test it.
- The pixel change is a numerical change from different kernel fusion and reduction order, not a precision
  change; it is visible in the hash but not in the image at this check.

## Next experiment

- Copy of `a100-flux-klein-torch28-bf16-compile.json` with `mode: "max-autotune"` (separately labelled; first generation reported separately).
- Runner: record the Inductor cache directory and cache hit/miss counters, and the Triton version, in the run directory.
