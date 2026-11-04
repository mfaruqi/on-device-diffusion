---
type: experiment-record
id: jetson-torch-flux-klein-003
status: complete
device: jetson-orin-nano
engine: pytorch-diffusers
runs: [jetson-torch-flux-klein-003__baseline__20261014-180843, jetson-torch-flux-klein-003__attempt__20261014-180328]
updated: 2026-10-14
---

# jetson-torch-flux-klein-003: tiled VAE decode (128 px tiles, 0.25 overlap), PyTorch/diffusers klein Q4_0 + NF4, sequential residency, 512×512

Status: **complete** (2026-10-14, run `__baseline__20261014-180843`; no Slurm job, run directly on the
device). Smoke gate `__attempt__20261014-180328` complete.

## Question

How much does diffusers' tiled VAE decode with 128-pixel tiles lower the allocator peak, which VAE
decode sets in [jetson-torch-flux-klein-001](jetson-torch-flux-klein-001.md), and what does it cost in
decode time, with everything else unchanged? Reference: the same-session 001 baseline
[`__baseline__20261014-171410`](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/).
This is a memory-residency lever under a memory budget (RQ1).

## Setup

- [Configuration](../configs/jetson-torch-flux-klein-q4-nf4-512-vaetile.json) and
  [smoke configuration](../configs/jetson-torch-flux-klein-q4-nf4-512-vaetile-smoke.json); the
  run-directory `config.json` copies match these files. Reference config:
  [jetson-torch-flux-klein-q4-nf4-512.json](../configs/jetson-torch-flux-klein-q4-nf4-512.json).
- **One change vs the reference:** `optimizations.vae_tiling: {"tile_sample_px": 128, "overlap": 0.25}`
  (reference: `false`). The config otherwise differs only in `id`, `description` and `reference_config`.
- How the change is applied ([scripts/run_flux_portable.py](../scripts/run_flux_portable.py),
  `apply_vae_tiling`): `pipe.vae.enable_tiling()`, then `tile_sample_min_size = 128`,
  `tile_latent_min_size = 128 // 2^(len(block_out_channels) − 1)` and `tile_overlap_factor = 0.25` on the
  VAE. `vae.tiled_decode` is wrapped to count calls, and the runner fails the run unless the tiled path
  ran once per generation.
- Workload, compressed checkpoints and execution policy unchanged from 001: distilled klein
  `e7b7dc27f91deacad38e78976d1f2b499d76a294`, batch 1, 512×512, 4 steps, guidance 1.0,
  `max_sequence_length` 512, fixed prompt, seed 0; GGUF Q4_0 transformer (BF16 compute), pre-quantized
  bitsandbytes NF4 text encoder, BF16 VAE; sequential stage residency, so encoder load and transformer
  reload are inside `wall_ms` and `other_ms`. No offload, VAE slicing, compile, compiler or attention
  restriction.
- Protocol: 1 first + 3 warm-up (discarded) + 10 measured; smoke: 1 first + 1 measured.
  Tegrastats every 200 ms.
- Runner sha256 `7d8b7d39…` (`source-provenance.json`), same as the 001 reference run. Launched under
  [scripts/jetson_memguard.sh](../scripts/jetson_memguard.sh) with `MIN_AVAIL_MB=400` and
  `MAX_SWAP_GROWTH_MB=1536` (memguard.json); the guard is a launcher safety setting outside the measured config.
- Device and software identical to the 001 reference run (`environment.json` differs only in its
  timestamp): Jetson Orin Nano 8 GB, R36.4.4, 25W, CUDA 12.6, torch 2.8.0, diffusers 0.40.0,
  transformers 5.16.1, bitsandbytes 0.48.0.dev0+ff389db. NVML unavailable; whole-system RAM from tegrastats.
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md). `vae_decode_ms` is
  the CUDA-event time around `pipe.vae.decode`, which with tiling enabled contains the whole tiled decode.

## Results

[Run](../results/runs/jetson-torch-flux-klein-003__baseline__20261014-180843/),
[summary.json](../results/runs/jetson-torch-flux-klein-003__baseline__20261014-180843/summary.json),
[runs.csv](../results/runs/jetson-torch-flux-klein-003__baseline__20261014-180843/runs.csv),
[stages.csv](../results/runs/jetson-torch-flux-klein-003__baseline__20261014-180843/stages.csv),
[stage-residency.json](../results/runs/jetson-torch-flux-klein-003__baseline__20261014-180843/stage-residency.json),
[memguard.json](../results/runs/jetson-torch-flux-klein-003__baseline__20261014-180843/memguard.json).
Started 2026-10-14T22:08:43Z, finished 22:30:03Z; `status.json` `complete`; guard not triggered (exit 0).

### Tiled-decode evidence

The runner writes no separate tiling file; the evidence is indirect but complete:
- The run's `status.json` is `complete`, which requires the runner's check that `vae.tiled_decode` ran
  once per generation (14 of 14; smoke run 2 of 2) to pass.
- Every generation's `vae_decode` stage peak allocation is 2.527–2.529 GiB (all 14 rows in stages.csv;
  both smoke rows likewise), 0.041–0.043 GiB above the stage's starting allocation, against
  3.081–3.083 GiB (0.596–0.597 GiB above start) in all 14 reference generations; every measured decode
  takes 1839.9–2535.3 ms against 489.6–540.6 ms in the reference.

### Latency

| Metric | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall_ms | 82991.0 | 81456.6–86355.4 | – |
| gpu_span_ms | 82989.6 | 81455.2–86353.9 | – |
| text_encode_ms | 1450.2 | 1174.4–2178.9 | 1.7% |
| denoise_ms | 14748.4 | 14708.0–15269.8 | 17.8% |
| denoise_step_0_ms | 4007.1 | 3959.1–4470.1 | – |
| denoise_step_1_ms | 3581.8 | 3572.5–3602.0 | – |
| denoise_step_2_ms | 3580.4 | 3569.4–3598.7 | – |
| denoise_step_3_ms | 3580.8 | 3571.3–3599.0 | – |
| Steady per-step (mean of steps 1–3 per generation) | 3581.4 | 3571.1–3599.9 | – |
| vae_decode_ms | 1850.4 | 1839.9–2535.3 | 2.2% |
| postprocess_ms | 45.4 | 38.1–74.6 | 0.1% |
| other_ms (includes encoder load and transformer reload) | 64953.6 | 63596.5–67256.8 | 78.3% |
| First generation: wall_ms | 92993.6 | – | – |
| First generation: text_encode / denoise (step 0) / vae_decode / postprocess / other | 4986.2 / 16082.3 (5295.8) / 5128.3 / 50.4 / 66700.6 | – | – |
| Warm-ups (3, discarded): wall_ms | – | 86455.2–88497.3 | – |
| Cold load: load_total_s | 67.08 s (`from_pretrained` 66.65 s, `.to` 0.43 s); cached, not a cold read | – | – |

The steady per-step row is computed from `runs.csv`, not a `summary.json` field. Per-run
`other_ms / wall_ms` ranged 0.772–0.784.

Residency costs, measured generations ([stage-residency.json](../results/runs/jetson-torch-flux-klein-003__baseline__20261014-180843/stage-residency.json)):
transformer free 0.503 s (0.487–0.561), text encoder load 33.356 s (32.964–34.338), text encoder free
0.458 s (0.450–0.479), transformer reload 30.582 s (29.457–31.833); median sum 64.921 s.
`other_ms` minus the residency sum is 29–48 ms in nine measured generations and 349 ms in index 5.

### Memory

| Quantity | Value | Source |
|---|---:|---|
| Weights: transformer / VAE | 2.291 / 0.157 GiB | load.json |
| Allocated / reserved after load | 2.475 / 2.523 GiB | load.json |
| peak_alloc_gib, measured | 3.066 GiB (3.065–3.068) | summary.json |
| peak_reserved_gib, measured | 3.168 GiB (3.162–3.170) | summary.json |
| First generation peak alloc / reserved | 3.065 / 3.178 GiB | summary.json |
| Stage setting the allocator peak | `text_encode`, 3.065–3.068 GiB; each denoise step 2.907 GiB; `vae_decode` 2.527–2.529 GiB | stages.csv |
| host_rss_gib, measured | median 1.371 (1.366–1.668) GiB | summary.json |
| host_peak_rss_gib (includes load) | 5.483 GiB | summary.json |
| System RAM peak, whole capture (tegrastats, 6040 samples) | 5.663 GiB of 7.441 GiB | summary.json `system_memory` |
| Swap occupancy, whole capture (tegrastats) | 0.312 → 1.106 GiB | summary.json `system_memory` |
| Guard: swap at start → maximum | 319 → 1133 MB (growth 814 MB) | memguard.json |
| Guard: MemAvailable at start → minimum | 5994 → 1618 MB | memguard.json |

Allocated and reserved are not added; host RSS and GPU allocations share one physical pool and are not
added. Maximum sampled GPU temperature 55.0 °C.

### Repeatability

All 14 generations have the same `image_sha256`
(`7358e45c5f7f5457ee3ca3d2074bb220e05dc0e266e1c3eb835689580f222eb2`; `deterministic_output: true`), as do
both smoke generations. This hash differs from the 001 reference output (`fe94904f…`): tiled decode with
blended overlaps changes the decoded pixels. The images were not visually inspected for this record and
no quality metric was computed. Measured VAE decode is 1839.9–1859.7 ms except indices 4 (2011.0 ms) and
5 (2535.3 ms); index 5 also has the maximum step 0 (4470.1 ms), while the other nine step 0 values are
3959.1–4026.6 ms.

### Smoke gate

[`__attempt__20261014-180328`](../results/runs/jetson-torch-flux-klein-003__attempt__20261014-180328/):
first / measured wall 91918.8 / 88993.7 ms; VAE decode 5856.0 / 2584.1 ms; denoise 16150.0 / 15249.0 ms;
peak alloc 3.067 GiB, reserved 3.168 GiB; load 71.44 s; guard minimum MemAvailable 1567 MB,
swap 314 → 635 MB. Not a baseline.

### Delta vs reference (001, `__baseline__20261014-171410`, medians of 10 measured)

| Metric | 001 | 003 (tiled VAE) | Δ | Δ % |
|---|---:|---:|---:|---:|
| wall_ms | 80314.9 | 82991.0 | +2676.1 | +3.3% |
| text_encode_ms | 1241.3 | 1450.2 | +208.9 | +16.8% |
| denoise_ms | 14795.5 | 14748.4 | −47.1 | −0.3% |
| denoise_step_0_ms | 4015.3 | 4007.1 | −8.2 | −0.2% |
| Steady per-step (mean of steps 1–3) | 3590.8 | 3581.4 | −9.4 | −0.3% |
| vae_decode_ms | 491.4 | 1850.4 | +1359.0 | +276.6% |
| postprocess_ms | 47.8 | 45.4 | −2.4 | −5.1% |
| other_ms | 63343.9 | 64953.6 | +1609.6 | +2.5% |
| `vae_decode` stage peak alloc (GiB) | 3.081 | 2.527 | −0.554 | −18.0% |
| peak_alloc_gib (whole generation) | 3.081 | 3.066 | −0.015 | −0.5% |
| peak_reserved_gib | 3.422 | 3.168 | −0.254 | −7.4% |
| host_rss_gib | 1.277 | 1.371 | +0.094 | +7.3% |
| System RAM peak, whole capture (GiB) | 6.114 | 5.663 | −0.451 | −7.4% |
| Guard minimum MemAvailable (MB) | 1206 | 1618 | +412 | +34.2% |

Residency medians in the reference: text encoder load 33.411 s, transformer reload 29.526 s.

## Observations

- Tiled decode ran in every generation; the VAE-decode transient falls from 0.596 GiB to 0.041 GiB
  above the stage's starting allocation, and the stage peak from 3.081 to 2.527 GiB.
- The generation's allocator peak falls only 0.015 GiB (to 3.066 GiB), because text encoding
  (3.065–3.068 GiB) becomes the peak-setting stage; reserved peak falls 0.254 GiB.
- VAE decode takes 1.36 s longer (491 → 1850 ms, ×3.77).
- Denoise is unchanged (−0.3%); the text-encode median is 0.21 s higher, but its range (1174–2179 ms)
  overlaps the reference range (1187–2167 ms).
- Wall time rises 2.68 s (+3.3%): 1.36 s from VAE decode and 1.61 s from `other_ms`, where the
  transformer reload median is 1.06 s longer than in the reference (30.58 s vs 29.53 s).
- Whole-capture system RAM peak is 0.451 GiB lower and minimum MemAvailable 412 MB higher than in the reference.
- Output is bit-identical within the run but differs from the reference output.

## Interpretation

- With 128 px tiles, tiling removes the VAE decode as a peak-setting stage at a cost of about 1.36 s
  per generation. Under this residency policy the next limit is text encoding with the NF4 encoder
  resident, so further allocator-peak reductions need a change to the encoding stage, not the VAE.
- The `other_ms` increase comes from reload-time variation, which tiling does not touch; it is not
  attributable to this change. Hypothesis: it reflects storage and page-cache variation between runs.
- Hypothesis: the lower whole-capture RAM peak and higher minimum MemAvailable follow from the smaller
  VAE-decode transient; tegrastats is not aligned to stages, so the data do not show when either extreme occurred.
- The changed image hash means this is not a bit-exact execution change; any quality statement needs an
  image comparison.

## Next experiment

- New config: `vae_tiling` with `tile_sample_px` 256 on `jetson-torch-flux-klein-q4-nf4-512`, to measure the decode-time/peak trade-off at a larger tile.
- Saved-image diagnostics (PSNR/LPIPS) of the 003 output against the 001 output.
