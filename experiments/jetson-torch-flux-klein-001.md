---
type: experiment-record
id: jetson-torch-flux-klein-001
status: complete
device: jetson-orin-nano
engine: pytorch-diffusers
runs: [jetson-torch-flux-klein-001__baseline__20261014-171410, jetson-torch-flux-klein-001__baseline__20261014-161717, jetson-torch-flux-klein-001__baseline__20261014-161622, jetson-torch-flux-klein-001__baseline__20261014-161555, jetson-torch-flux-klein-001__baseline__20261012-153954, jetson-torch-flux-klein-001__attempt__20261012-153448, jetson-torch-flux-klein-001__attempt__20261012-153146, jetson-torch-flux-klein-001__attempt__20261012-152208, jetson-torch-flux-klein-001__attempt__20261012-143714, jetson-torch-flux-klein-001__attempt__20261012-123054, jetson-torch-flux-klein-001__attempt__20261012-122056, jetson-torch-env-t0__attempt__20261011-ailab, jetson-torch-env-t0__attempt__20261011, jetson-torch-t1-memory-diag-20261012]
updated: 2026-10-14
---

# jetson-torch-flux-klein-001: PyTorch/diffusers klein, GGUF Q4_0 transformer + NF4 text encoder, sequential stage residency, 512×512

Status: **complete** (2026-10-14, run `__baseline__20261014-171410`; no Slurm job, run directly on the
device). The full 1+3+10 protocol completed with all 10 measured generations. This is the record's
baseline result. Earlier evidence is retained below: the 2026-10-12 full-protocol run, stopped by the
memory guard after 8 of 10 measured generations, the 2026-10-12 smoke gate, and the attempt history.

## Question

Can the PyTorch/diffusers engine run distilled FLUX.2 [klein] 4B at 512×512 on the Jetson Orin
Nano (8 GB shared CPU/GPU memory), and what are its stage latencies and memory under a
sequential stage-residency policy? This is a separately labelled compressed-checkpoint workload
(GGUF Q4_0 transformer, bitsandbytes NF4 text encoder), not the BF16 reference configuration.
It serves the image-baseline work and RQ1.

## Setup

- [Configuration](../configs/jetson-torch-flux-klein-q4-nf4-512.json) (full protocol) and
  [smoke configuration](../configs/jetson-torch-flux-klein-q4-nf4-512-smoke.json) (1 first + 1 measured,
  no warm-ups). The run-directory `config.json` copies of the final smoke run and of all full-protocol
  runs match these files. No `*.resolved.json` exists; the model revision is pinned inline:
  `black-forest-labs/FLUX.2-klein-4B` @ `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
- Workload: distilled klein, batch 1, 512×512, 4 steps, guidance 1.0, `max_sequence_length` 512,
  prompt "A cat holding a sign that says hello world", seed 0.
  Scheduler: FlowMatchEulerDiscreteScheduler ([scheduler.json](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/scheduler.json)).
- Compressed checkpoints (reported separately from execution-only changes):
  - Transformer: GGUF Q4_0 file `flux-2-klein-4b-Q4_0.gguf`
    (sha256 `d1023499ef3f2f82ff7c50e6778495195c1b6cc34835741778868428111f9ff4`), loaded through diffusers
    `GGUFQuantizationConfig` with BF16 compute. The runner decodes the 60 BF16 attention-norm tensors
    stored outside `GGUFLinear` to BF16 parameters (lossless; listed in
    [load.json](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/load.json)).
  - Text encoder (Qwen3): bitsandbytes NF4, BF16 compute, from a pre-quantized copy of the pinned
    revision's encoder produced once by `scripts/save_quantized_text_encoder.py`
    (sha256 `eaa2048d95dbe273fc7361a0016261757928314b82e046ee90c900be7dc65883`, verified at load).
  - VAE: BF16 from the pinned snapshot.
- Execution policy (`optimizations.stage_residency`): `text_encoder: release_after_encode`,
  `transformer: release_during_encode`. Only one large component is resident at a time: each
  generation frees the transformer, loads the encoder, encodes, frees the encoder and reloads the
  transformer. Reloads happen between timed stages, so they are inside `wall_ms` and `other_ms`,
  not `text_encode_ms` (runner docstring, [scripts/run_flux_portable.py](../scripts/run_flux_portable.py)).
  No CPU offload, VAE tiling/slicing, `torch.compile` or compiler, and no `attention_backend`
  restriction (PyTorch selects the SDPA kernel; which kernel it selected is not recorded).
- Protocol: 1 first + 3 warm-up (discarded) + 10 measured. Tegrastats sampled every 200 ms.
- Runner: [scripts/run_flux_portable.py](../scripts/run_flux_portable.py), sha256 `7d8b7d39…` for the
  completed 2026-10-14 run and `ad55377e…` for the 2026-10-12 smoke and full runs
  (`source-provenance.json`). The later version adds the `attention_backend` option and explicit
  `vae_tiling` tile settings; neither is set in this configuration.
- Memory guard: [scripts/jetson_memguard.sh](../scripts/jetson_memguard.sh) kills the process group if
  MemAvailable falls below `MIN_AVAIL_MB` (400 MB) or swap use grows more than `MAX_SWAP_GROWTH_MB` from
  the start (polled every 0.1 s). The completed run used `MAX_SWAP_GROWTH_MB=1536`
  ([memguard.json](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/memguard.json));
  the 2026-10-12 runs used the default 768 MB. The guard threshold is a launcher safety setting outside
  the measured configuration; it does not change what the runner executes.
- Device: Jetson Orin Nano 8 GB (`mfaruqi-desktop`), Jetson Linux R36.4.4, kernel 5.15.148-tegra,
  25W power mode, driver 540.4.0, CUDA 12.6, cuDNN 9.3.0, compute capability 8.7
  ([environment.json](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/environment.json);
  identical to the 2026-10-12 full run's file apart from the timestamp).
- Software (Jetson AI Lab jp6/cu126 index): **torch 2.8.0**, diffusers 0.40.0, transformers 5.16.1,
  bitsandbytes 0.48.0.dev0+ff389db, torchao 0.13.0, Python 3.10.12. `environment.json` records no git
  commit (`git_commit` empty); script hashes in `source-provenance.json` identify the code. NVML is
  unavailable on Jetson, so `device_used_peak_gib` is not reported; whole-system RAM comes from tegrastats.
  `HF_HOME=/home/mfaruqi/hf`, `HF_HUB_OFFLINE=1`.
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md).

## Results: full protocol (complete, 2026-10-14)

[Run](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/),
[summary.json](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/summary.json),
[runs.csv](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/runs.csv),
[stages.csv](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/stages.csv),
[stage-residency.json](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/stage-residency.json),
[memguard.json](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/memguard.json).
Started 2026-10-14T21:14:10Z, finished 21:35:11Z; `status.json` `complete`; guard not triggered (exit 0).

### Latency

| Metric | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall_ms | 80314.9 | 78169.8–82599.2 | – |
| gpu_span_ms | 80313.1 | 78168.5–82600.7 | – |
| text_encode_ms | 1241.3 | 1187.0–2167.5 | 1.5% |
| denoise_ms | 14795.5 | 14705.3–14817.0 | 18.4% |
| denoise_step_0_ms | 4015.3 | 3955.8–4038.4 | – |
| denoise_step_1_ms | 3591.2 | 3580.2–3603.7 | – |
| denoise_step_2_ms | 3590.4 | 3578.5–3599.1 | – |
| denoise_step_3_ms | 3590.4 | 3579.3–3599.9 | – |
| Steady per-step (mean of steps 1–3 per generation) | 3590.8 | 3579.3–3600.9 | – |
| vae_decode_ms | 491.4 | 489.6–540.6 | 0.6% |
| postprocess_ms | 47.8 | 41.0–67.3 | 0.1% |
| other_ms (includes encoder load and transformer reload) | 63343.9 | 61584.6–65354.9 | 78.9% |
| First generation: wall_ms | 94586.1 | – | – |
| First generation: text_encode / denoise (step 0) / vae_decode / postprocess / other | 4274.3 / 15992.9 (5224.7) / 4375.6 / 183.9 / 69676.9 | – | – |
| Warm-ups (3, discarded): wall_ms | – | 83039.5–87126.4 | – |
| Cold load: load_total_s | 82.35 s (`from_pretrained` 81.49 s, `.to` 0.86 s); cached, not a cold read | – | – |

The steady per-step row is computed from `runs.csv` (median over the 10 measured generations of each
generation's mean of steps 1–3); it is not a `summary.json` field. Shares divide each stage median by
the wall median. Per-run `other_ms / wall_ms` ranged 0.787–0.797; measured `wall_ms` stdev 1444.9 ms.

Per-generation residency costs, measured generations
([stage-residency.json](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/stage-residency.json)):

| Residency step | Median (s) | Min–max (s) | First generation (s) |
|---|---:|---:|---:|
| transformer free | 0.483 | 0.478–0.487 | 0.827 |
| text encoder load (NF4) | 33.411 | 31.839–33.903 | 34.146 |
| text encoder free | 0.456 | 0.446–0.462 | 0.475 |
| transformer reload (GGUF Q4_0) | 29.526 | 27.975–30.623 | 33.183 |
| Sum of the four | 63.312 | – | – |

In each measured generation, `other_ms` minus the residency sum is 28–39 ms (median 34 ms): the
reloads account for essentially all of `other_ms`. The residency sum is a median 79.1% of `wall_ms`.

### Memory

| Quantity | Value | Source |
|---|---:|---|
| Transformer weights (GGUF Q4_0 + BF16 norms) | 2.291 GiB | load.json |
| VAE weights | 0.157 GiB | load.json |
| Text encoder | not resident after load; loaded per generation | load.json (`resident: false`) |
| Allocated / reserved after load (transformer + VAE) | 2.475 / 2.523 GiB | load.json |
| peak_alloc_gib, measured (all 10 equal) | 3.081 GiB | summary.json |
| peak_reserved_gib, measured (all 10 equal) | 3.422 GiB | summary.json |
| First generation peak alloc / reserved | 3.083 / 3.426 GiB | summary.json |
| Stage setting the allocator peak | `vae_decode` 3.081 GiB (from 2.485 GiB at stage start); `text_encode` 3.067 GiB; each denoise step 2.907 GiB | stages.csv |
| Allocated at denoise-step start | 2.485 GiB | stages.csv |
| host_rss_gib, measured | median 1.277 (1.224–1.334) GiB | summary.json |
| host_peak_rss_gib (includes load) | 5.478 GiB | summary.json |
| System RAM peak, whole capture (tegrastats, 200 ms, 5929 samples) | 6.114 GiB of 7.441 GiB | summary.json `system_memory` |
| Swap occupancy, whole capture (tegrastats) | 0.293 → 1.227 GiB | summary.json `system_memory` |
| Guard (0.1 s polling): swap at start → maximum | 299 → 1256 MB (growth 957 MB) | memguard.json |
| Guard: MemAvailable at start → minimum | 5913 → 1206 MB | memguard.json |
| Device-wide peak (NVML) | not available on Jetson | environment.json |

Allocated and reserved are different views of the same pool and are not added. Host RSS and GPU
allocations share one physical pool on this device and are not added either. Tegrastats MB are treated
as MiB. Swap occupancy rose across the capture (tegrastats deciles: 300, 300, 985, 1108, 1148, 1154,
1153, 1200, 1248, 1256, 1251 MB). The observed swap growth (957 MB) is above the 768 MB limit that
stopped the 2026-10-12 run and below the 1536 MB limit used here. Maximum sampled GPU temperature 55.1 °C.

### Repeatability

All 14 generations (first, 3 warm-ups, 10 measured) have the same `image_sha256`
(`fe94904f2740c0901c3d8c423dd9be16946341189a51d88ab3c615179c522ae3`; `deterministic_output: true`),
identical to every generation of the 2026-10-12 smoke and full runs, so the 2026-10-12 visual check
(below) applies to this output unchanged. Denoise steps 1–3 span 3578.5–3603.7 ms across measured
generations; step 0 spans 3955.8–4038.4 ms. Measured VAE decode is 489.6–494.0 ms except the first
measured generation (index 4, 540.6 ms). Text encode is 1187.0–1504.7 ms except indices 4 (1700.4 ms)
and 7 (2167.5 ms).

## Results: attempt history

| Run directory | Runner sha256 (prefix) | Policy / change | Outcome |
|---|---|---|---|
| [`__attempt__20261012-122056`](../results/runs/jetson-torch-flux-klein-001__attempt__20261012-122056/) | `d38a4574` | all components resident; encoder quantized to NF4 while loading | Killed by the memory guard during loading (MemAvailable 396 MB). `status.json` still says `running` because the process was SIGKILLed; the failure is recorded in [status-note.json](../results/runs/jetson-torch-flux-klein-001__attempt__20261012-122056/status-note.json). Recorded cause: a runner bug kept the GGUF transformer on the CPU while the encoder loaded, then copied it to the GPU, so both copies coexisted. Fixed in the runner. |
| [`__attempt__20261012-123054`](../results/runs/jetson-torch-flux-klein-001__attempt__20261012-123054/) | `647f2d44` | all components resident | Load completed (163.6 s; 5.139 GiB allocated after load). Killed by the guard during the first text encoding (MemAvailable 398 MB < 400 MB, [memguard.json](../results/runs/jetson-torch-flux-klein-001__attempt__20261012-123054/memguard.json)). `status.json` stale (`running`); failure in [status-note.json](../results/runs/jetson-torch-flux-klein-001__attempt__20261012-123054/status-note.json). |
| [`__attempt__20261012-143714`](../results/runs/jetson-torch-flux-klein-001__attempt__20261012-143714/) | `01c62b3c` | encoder `release_after_encode` only (transformer resident); pre-quantized encoder | Load completed (77.7 s; 2.475 GiB allocated). `status.json` stale (`running`); no `memguard.json`, `status-note.json`, `runs.csv` rows or images. Tegrastats ends at 14:39:39 with RAM 7007/7620 MB (≈6.84 GiB), swap 496 MB and GPU load 99%. **Inference:** the process ended abruptly during the first generation under near-full system memory; whether the guard, the kernel OOM killer or something else ended it cannot be determined from the files (the missing `memguard.json` means the guard script did not finish writing its record). |
| [`__attempt__20261012-152208`](../results/runs/jetson-torch-flux-klein-001__attempt__20261012-152208/) | `9ab31188` | sequential residency (current smoke config) | **Failed** in the first generation's first transformer call: `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: unsigned char != c10::BFloat16` in the GGUF linear of `time_guidance_embed` ([status.json](../results/runs/jetson-torch-flux-klein-001__attempt__20261012-152208/status.json)). Load 92.3 s. |
| [`__attempt__20261012-153146`](../results/runs/jetson-torch-flux-klein-001__attempt__20261012-153146/) | `e7741c7f` | as above, plus a dtype shim (`transformer.dtype` reports bfloat16) | **Failed** in the first transformer call: `RuntimeError: Expected weight to be of same shape as normalized_shape, but got weight of shape [256] and normalized_shape = [128]` in `attn.norm_q` ([status.json](../results/runs/jetson-torch-flux-klein-001__attempt__20261012-153146/status.json)). Load 51.9 s. **Inference:** the BF16 norm tensors were still raw bytes (256 bytes for 128 BF16 values); the next runner version decodes them to BF16 parameters (recorded in the later `load.json`). |
| [`__attempt__20261012-153448`](../results/runs/jetson-torch-flux-klein-001__attempt__20261012-153448/) | `ad55377e` | sequential residency, smoke config | **Complete**: first + 1 measured (below). |
| [`__baseline__20261012-153954`](../results/runs/jetson-torch-flux-klein-001__baseline__20261012-153954/) | `ad55377e` | sequential residency, full config; guard swap limit 768 MB | **Failed** at generation index 12: guard stop on swap growth (below). |
| [`__baseline__20261014-161555`](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-161555/) | – (not written) | full config | **Refused** before loading: `AssertionError('Inference must run offline (HF_HUB_OFFLINE=1)')` ([status.json](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-161555/status.json)); `HF_HUB_OFFLINE` was not set. No measurement. |
| [`__baseline__20261014-161622`](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-161622/) | `ad55377e` | full config | **Failed** at load: `LocalEntryNotFoundError`, no cached snapshot for the pinned revision ([status.json](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-161622/status.json)). `environment.json` records `HF_HOME` unset, so the default cache location was searched. No measurement. |
| [`__baseline__20261014-161717`](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-161717/) | `ad55377e` | full config | **Stopped by the operator** (SIGINT, 20:22:41Z) after the first generation: an unrelated `sha256sum` of about 5 GB of model files ran on the same microSD from about 20:20Z to 20:22Z, overlapping this disk-reload-bound run ([status-note.json](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-161717/status-note.json)). Contaminated; measurements not reported. `status.json` stays `running`. |
| [`__baseline__20261014-171410`](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/) | `7d8b7d39` | full config; guard swap limit 1536 MB | **Complete** (above). |

Images are git-ignored. None of the failed attempts produced an image; the `images/` directories of
the 143714, 152208 and 153146 attempts are empty in the copied run directories.

## Results: smoke gate (2026-10-12, complete)

[Run](../results/runs/jetson-torch-flux-klein-001__attempt__20261012-153448/),
[summary](../results/runs/jetson-torch-flux-klein-001__attempt__20261012-153448/summary.json),
[stage residency](../results/runs/jetson-torch-flux-klein-001__attempt__20261012-153448/stage-residency.json).
One first and one measured generation; not a baseline.

| Metric | First (ms) | Second observation (ms) |
|---|---:|---:|
| wall_ms | 88249.0 | 84636.2 |
| text_encode_ms | 5431.9 | 2073.7 |
| denoise_ms | 15682.1 | 15157.1 |
| vae_decode_ms | 3412.8 | 957.4 |
| postprocess_ms | 110.8 | 38.7 |
| other_ms | 63577.1 | 66408.0 |

Per-generation residency costs ([stage-residency.json](../results/runs/jetson-torch-flux-klein-001__attempt__20261012-153448/stage-residency.json)):
encoder load 28.3 s / 33.0 s, transformer reload 33.1 s / 31.8 s, frees 0.5–0.8 s each. Their sums
(63.07 s, 66.10 s) account for all but 511 ms and 312 ms of `other_ms` in the two generations.
Load: 76.21 s (`from_pretrained` 75.68 s). Allocator peak 3.083 / 3.081 GiB (reserved 3.426 GiB);
host peak RSS 5.149 GiB including load. Whole-capture tegrastats RAM peak 6.090 GiB, swap
0.282–0.511 GiB. Guard: not triggered; minimum MemAvailable 1140 MB.

## Results: earlier full protocol (2026-10-12, partial, failed)

[Run](../results/runs/jetson-torch-flux-klein-001__baseline__20261012-153954/),
[runs.csv](../results/runs/jetson-torch-flux-klein-001__baseline__20261012-153954/runs.csv),
[stages.csv](../results/runs/jetson-torch-flux-klein-001__baseline__20261012-153954/stages.csv),
[status-note.json](../results/runs/jetson-torch-flux-klein-001__baseline__20261012-153954/status-note.json),
[memguard.json](../results/runs/jetson-torch-flux-klein-001__baseline__20261012-153954/memguard.json).

Error: the guard killed the process group (exit 137) during generation index 12 because swap use grew
774 MB, above the 768 MB limit (swap 293 MB at start, 1067 MB maximum; minimum MemAvailable 1198 MB).
`status.json` stays `running` because the process was SIGKILLed; no `summary.json`,
`memory_segments.json` or `stage-residency.json` was written (the runner writes them after the last
generation). The statistics below were computed from `runs.csv` and are **not** a valid
1+3+10 baseline: **n = 8 of 10 planned measured generations** (indices 4–11).

### Latency

| Metric | Median (ms), n = 8 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall_ms | 79293.2 | 76850.6–84326.6 | – |
| gpu_span_ms | 79291.9 | 76849.4–84325.3 | – |
| text_encode_ms | 1279.2 | 1183.4–2282.4 | 1.6% |
| denoise_ms | 14758.0 | 14661.5–15453.7 | 18.6% |
| denoise_step_0_ms | 3995.3 | 3941.0–4681.8 | – |
| denoise_step_1_ms | 3584.3 | 3573.3–3601.7 | – |
| denoise_step_2_ms | 3586.7 | 3573.9–3601.3 | – |
| denoise_step_3_ms | 3587.4 | 3573.2–3601.7 | – |
| vae_decode_ms | 492.7 | 488.7–1440.5 | 0.6% |
| postprocess_ms | 48.1 | 39.3–107.3 | 0.1% |
| other_ms (includes encoder load and transformer reload) | 62672.1 | 60354.3–65041.4 | 79.0% |
| First generation: wall_ms | 85422.0 | – | – |
| First generation: text_encode / denoise / vae_decode / other | 4899.5 / 15665.5 / 3721.6 / 60977.3 | – | – |
| Warm-ups (3, discarded): wall_ms | – | 83658.7–85811.5 | – |
| Cold load: load_total_s | 78.98 s (`from_pretrained` 78.55 s, `.to` 0.43 s); cached, not a cold read | – | – |

Shares divide each stage median by the wall median, so they do not sum to exactly 100%. Per-run
`other_ms / wall_ms` ranged 0.771–0.796. `gpu_span_ms` is within 0.003% of `wall_ms` in every row.

### Memory

| Quantity | Value | Source |
|---|---:|---|
| peak_alloc_gib, measured (all 8 equal) | 3.081 GiB | runs.csv |
| peak_reserved_gib, measured (all 8 equal) | 3.422 GiB | runs.csv |
| First generation peak alloc / reserved | 3.083 / 3.426 GiB | runs.csv |
| Stage setting the allocator peak | `vae_decode` (3.081 GiB); `text_encode` 3.067 GiB; each denoise step 2.907 GiB | stages.csv |
| host_rss_gib, measured | median 1.529 (1.501–1.613) GiB | runs.csv |
| System RAM peak, whole capture (tegrastats, 200 ms) | 6.035 GiB of 7.441 GiB | tegrastats.log |
| Swap occupancy, whole capture (tegrastats) | 0.286 → 1.024 GiB | tegrastats.log |

Weight sizes and after-load allocations equal those of the 2026-10-14 run (load.json). The guard
(0.1 s polling) recorded a higher swap maximum (1067 MB) than the 200 ms tegrastats samples (1049 MB).
Swap occupancy rose stepwise across the capture (tegrastats deciles: 293, 567, 596, 602, 645, 910, 914,
958, 965, 967, 1049 MB). Maximum sampled GPU temperature 56.5 °C.

### Repeatability

All 12 completed generations (first, 3 warm-ups, 8 measured) have the same `image_sha256`
(`fe94904f2740c0901c3d8c423dd9be16946341189a51d88ab3c615179c522ae3`), as do both smoke-gate
generations. The last measured generation (index 11) is the slowest in wall time and has the maximum
text encode (2282.4 ms), VAE decode (1440.5 ms) and step 0 (4681.8 ms); the other seven measured VAE
decodes are 488.7–495.3 ms. Visual check of the saved first image: a coherent cat holding a sign; the
sign text reads "HOLLO WORLD" rather than "hello world". This is functional evidence, not a quality evaluation.

## Results: environment gate (T0)

- [Default JetPack wheel](../results/runs/jetson-torch-env-t0__attempt__20261011/status.json):
  **failed**. torch `2.5.0a0+872d972e41.nv24.08` (NVIDIA JetPack 6.1 redist) passed CUDA, BF16 matmul
  and SDPA (flash, efficient, math) checks, but the wheel lacks `torch._C._distributed_c10d`, so
  importing the diffusers Flux2 pipeline, GGUF support and torchao failed; transformers disabled PyTorch
  because it parsed `2.5.0a0` as < 2.5; the bitsandbytes NF4 check exited with CUDA
  "named symbol not found" ([probe.log](../results/runs/jetson-torch-env-t0__attempt__20261011/probe.log)).
- [Jetson AI Lab stack](../results/runs/jetson-torch-env-t0__attempt__20261011-ailab/status.json):
  **complete**. torch 2.8.0; all checks passed: `Flux2KleinPipeline`, `Flux2Transformer2DModel`,
  `GGUFQuantizationConfig`, `AutoencoderKLFlux2` import; GGUF utilities present; torchao int4 and
  bitsandbytes NF4 CUDA linears return finite outputs; SDPA flash/efficient/math run; 2048² BF16
  matmul 4.39 ms (3.91 TFLOPS). All later runs use this stack.

## Results: memory diagnostics

[Diagnostics](../results/runs/jetson-torch-t1-memory-diag-20261012/README.md), run under the guard, MemAvailable
from `/proc/meminfo`, sizes recorded in MiB:

- GGUF Q4_0 transformer stays quantized on CUDA: 100 Q4_0 tensors (1974.4 MB) + 69 BF16 tensors (372.0 MB); 2365 MB on CUDA.
- NF4 Qwen3 encoder quantized while loading: 2721 MB on CUDA, load 99.8 s.
- Transformer, encoder and VAE all on CUDA: 5263 MB allocated, 843 MB MemAvailable left.
- A pipeline-equivalent encoder call (chat template, padded to 512 tokens, attention mask) lowers
  MemAvailable by about 815 MB; the prompt has 21 real tokens. A random-token forward lowers it by about
  450 MB at L = 64 and 940 MB at L = 512 including logits; about 370–400 MB is first-call library/kernel
  loading outside the torch allocator.
- An encode-only pipeline (transformer and VAE set to None) was also killed by the guard; cause not isolated.

## Observations

- The full 1+3+10 protocol completes under sequential residency: median wall 80.31 s (78.17–82.60 s),
  allocator peak 3.081 GiB in every measured generation, output bit-identical across all 14 generations.
- The all-resident policy does not complete a generation: after load, 843 MB MemAvailable remains and
  the first encoder call needs about 815 MB plus allocator growth; the guard stopped it at 398 MB.
- The allocator peak is set by VAE decode (3.081 GiB, 0.596 GiB above the 2.485 GiB held at stage
  start); text encoding reaches 3.067 GiB and each denoise step 2.907 GiB.
- `other_ms` is 78.9% of median wall time; per-generation encoder load (median 33.41 s) plus
  transformer reload (median 29.53 s) account for all but 28–39 ms of it.
- Steady per-step denoise time is 3.59 s (steps 1–3, spread under 26 ms); step 0 is 4.02 s, about 0.42 s longer.
- Swap occupancy grew by 957 MB over the run while MemAvailable never fell below 1206 MB and system RAM
  stayed at or below 6.114 GiB; the run's 1536 MB guard limit was not reached.
- Relative to the 2026-10-12 partial medians (n = 8), the 2026-10-14 medians differ by +1.3% (wall),
  +0.3% (denoise), −0.3% (VAE decode) and −3.0% (text encode); allocator peaks are identical.

## Interpretation

- Under this policy, generation latency is dominated by per-generation weight reloading (≈ 63 s of
  ≈ 80 s), not by compute. The compute stages (text encode, denoise, VAE decode, postprocess) total
  about 16.6 s per generation at the medians, of which denoising is 14.8 s.
- Hypothesis (from [status-note.json](../results/runs/jetson-torch-flux-klein-001__baseline__20261012-153954/status-note.json),
  not verified): each generation streams roughly 5 GB of weights through the page cache and the kernel
  gradually moves idle anonymous pages to swap; the swap growth reflects that and not a shortage of
  available memory. The completed run is consistent with it (swap grew while MemAvailable stayed
  above 1.2 GB), but swap occupancy alone does not establish swap I/O; swap-in/out counters would test it.
- Hypothesis: reload times depend on storage read speed and page-cache state, which would explain why a
  concurrent 5 GB read on the same microSD contaminated the 161717 attempt. The data do not separate
  disk reads from host-to-device copies within the reload times.
- `text_encode_ms` here excludes the encoder load by definition; a reader comparing text encoding across
  policies must add the residency costs in `other_ms`.

## Next experiment

- Rerun `jetson-torch-flux-klein-q4-nf4-512` with swap-in/out counters (`/proc/vmstat`) recorded alongside tegrastats, to test the swap hypothesis.
- New config: keep the encoder released but the transformer resident with a shorter `max_sequence_length` (separately labelled workload).
- Compare with the stable-diffusion.cpp Jetson baseline in a `compare-*` record.
