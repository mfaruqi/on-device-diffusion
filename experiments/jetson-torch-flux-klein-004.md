---
type: experiment-record
id: jetson-torch-flux-klein-004
status: complete
device: jetson-orin-nano
engine: pytorch-diffusers
runs: [jetson-torch-flux-klein-004__baseline__20261015-141028, jetson-torch-flux-klein-004__attempt__20261015-140608, jetson-torch-flux-klein-004__profile__20261017-215429, jetson-torch-flux-klein-004__profile__20261017-222734]
updated: 2026-10-18
---

# jetson-torch-flux-klein-004: exact conditioning reuse, PyTorch/diffusers klein Q4_0 + NF4, sequential residency, 512×512

Status: **complete** (2026-10-15, run `__baseline__20261015-141028`, 18:10:28–18:17:06 UTC; no Slurm job,
run directly on the device). Smoke gate `__attempt__20261015-140608` complete.

## Question

How much of the generation latency of [jetson-torch-flux-klein-001](jetson-torch-flux-klein-001.md)
(per-generation encoder load and transformer reload under sequential residency) disappears when the
prompt encoding of a fixed workload is computed once and reused, with everything else unchanged?
Reference: the 001 baseline
[`__baseline__20261014-171410`](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/).
Execution-policy lever under a memory budget (RQ1).

## Setup

- [Configuration](../configs/jetson-torch-flux-klein-q4-nf4-512-condreuse.json) and
  [smoke configuration](../configs/jetson-torch-flux-klein-q4-nf4-512-condreuse-smoke.json) (1 first + 1
  measured); the run-directory `config.json` copies equal these files. Reference config:
  [jetson-torch-flux-klein-q4-nf4-512.json](../configs/jetson-torch-flux-klein-q4-nf4-512.json).
- **One change vs the reference:** `optimizations.conditioning_reuse: true`. The config otherwise differs
  only in `id`, `description` and `reference_config`.
- How the change is applied ([scripts/run_flux_portable.py](../scripts/run_flux_portable.py) `benchmark()`,
  [scripts/mps_conditioning.py](../scripts/mps_conditioning.py)): `pipe.encode_prompt` is wrapped three
  times, innermost first: (1) the stage-residency swaps (free transformer, load encoder, encode, free
  encoder, reload transformer); (2) `ConditioningCache`, keyed on the full bound argument list of
  `encode_prompt`, capacity 1 (guidance 1.0, no negative prompt); (3) the stage recorder. A cache hit returns
  the stored embeddings without calling the inner function, so it skips the encoder load **and** the
  transformer release and reload; the transformer loaded at the first generation stays resident. The
  single miss (first generation) runs the swaps inside the recorder's timed `text_encode` stage. Because
  the recorder is outside the cache, `text_encode_ms` on a hit is the retrieval time. The runner fails the
  run unless there is exactly 1 miss and 13 hits, and unless exactly one residency swap was recorded.
- Workload, compressed checkpoints and policy unchanged from 001: distilled klein
  `e7b7dc27f91deacad38e78976d1f2b499d76a294`, batch 1, 512×512, 4 steps, guidance 1.0,
  `max_sequence_length` 512, fixed prompt, seed 0; GGUF Q4_0 transformer (BF16 compute), pre-quantized
  bitsandbytes NF4 text encoder, BF16 VAE (compressed checkpoints, separately labelled);
  `stage_residency` `text_encoder: release_after_encode`, `transformer: release_during_encode`. No offload,
  tiling, compile or attention restriction.
- Protocol: 1 first + 3 warm-up (discarded) + 10 measured; smoke 1 + 1. Tegrastats every 200 ms.
- Runner sha256 `44e44608…`, `mps_conditioning.py` `80a2bc49…`
  ([source-provenance.json](../results/runs/jetson-torch-flux-klein-004__baseline__20261015-141028/source-provenance.json));
  launched by [scripts/jetson_gap_queue_20261015d.sh](../scripts/jetson_gap_queue_20261015d.sh) under
  [scripts/jetson_memguard.sh](../scripts/jetson_memguard.sh) (`MIN_AVAIL_MB=400`, `MAX_SWAP_GROWTH_MB=1536`;
  launcher setting outside the measured config).
- Device and software as in 001: Jetson Orin Nano 8 GB, R36.4.4, 25W, CUDA 12.6, torch 2.8.0, diffusers
  0.40.0, transformers 5.16.1, bitsandbytes 0.48.0.dev0+ff389db, torchao 0.13.0
  ([environment.json](../results/runs/jetson-torch-flux-klein-004__baseline__20261015-141028/environment.json)).
  `git_commit` empty; provenance is the script hashes. NVML unavailable; whole-system RAM from tegrastats.
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md). Steady per-step = median over
  measured generations of each generation's mean of `denoise_step_1..3_ms` (computed from `runs.csv`).

## Results

[Run](../results/runs/jetson-torch-flux-klein-004__baseline__20261015-141028/),
[summary.json](../results/runs/jetson-torch-flux-klein-004__baseline__20261015-141028/summary.json),
[runs.csv](../results/runs/jetson-torch-flux-klein-004__baseline__20261015-141028/runs.csv),
[stages.csv](../results/runs/jetson-torch-flux-klein-004__baseline__20261015-141028/stages.csv),
[conditioning-cache.json](../results/runs/jetson-torch-flux-klein-004__baseline__20261015-141028/conditioning-cache.json),
[stage-residency.json](../results/runs/jetson-torch-flux-klein-004__baseline__20261015-141028/stage-residency.json),
[memguard.json](../results/runs/jetson-torch-flux-klein-004__baseline__20261015-141028/memguard.json).
`status.json` `complete`; guard not triggered (exit 0).

Conditioning receipt: `policy exact_prompt_encoding`, hits 13, misses 1, entries 1, capacity 1.
Residency receipt: one swap only (first generation): transformer free 0.805 s, encoder load 34.984 s,
encoder free 0.480 s, transformer reload 34.017 s (sum 70.287 s).

### Latency

| Metric | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall_ms | 14911.4 | 14880.6–14961.3 | – |
| gpu_span_ms | 14910.6 | 14879.7–14960.4 | – |
| text_encode_ms (cache retrieval) | 0.9 | 0.7–2.5 | 0.0% |
| denoise_ms | 14377.3 | 14350.5–14426.8 | 96.4% |
| denoise_step_0_ms | 3599.6 | 3583.9–3609.2 | – |
| denoise_step_1 / 2 / 3_ms | 3592.4 / 3594.4 / 3592.4 | 3585.9–3606.9 (all three) | – |
| Steady per-step (mean of steps 1–3) | 3593.1 | 3588.9–3605.9 | – |
| vae_decode_ms | 485.7 | 485.4–486.2 | 3.3% |
| postprocess_ms | 17.1 | 11.9–17.9 | 0.1% |
| other_ms | 30.6 | 23.2–31.6 | 0.2% |
| First generation: wall_ms | 96501.7 | – | – |
| First generation: text_encode (includes the one residency swap, 70.287 s) / denoise (step 0) / vae_decode / postprocess / other | 75090.5 / 15875.0 (5100.8) / 4449.9 / 144.0 / 922.3 | – | – |
| Warm-ups (3, discarded): wall_ms | – | 14954.3–15009.9 | – |
| Cold load: load_total_s | 81.84 s (`from_pretrained` 81.45 s); cached, not a cold read | – | – |

Measured `wall_ms` stdev 23.7 ms. In the first generation, `text_encode_ms` minus the swap sum is about
4.80 s, the encoder computation itself.

### Memory

| Quantity | Value | Source |
|---|---:|---|
| Transformer / VAE weights | 2.291 / 0.157 GiB | summary.json |
| peak_alloc_gib, measured (all 10 equal) | 3.083 GiB, set by `vae_decode` (from 2.487 GiB at stage start) | summary.json, stages.csv |
| peak_reserved_gib, measured | 3.426 GiB | summary.json |
| Allocated at hit `text_encode` / each denoise step peak | 2.486 / 2.908 GiB | stages.csv |
| First generation peak alloc / reserved | 3.083 / 3.426 GiB (`text_encode` 3.065 GiB) | summary.json, stages.csv |
| host_rss_gib, measured | 1.834 (1.834–1.834) GiB | summary.json |
| host_peak_rss_gib (includes load) | 5.501 GiB | summary.json |
| System RAM peak, whole capture (tegrastats, 1803 samples) | 5.982 / 7.441 GiB | summary.json `system_memory` |
| Swap occupancy, whole capture (tegrastats) | 0.313 → 0.688 GiB | summary.json `system_memory` |
| Guard: MemAvailable start → minimum | 5910 → 1234 MB | memguard.json |
| Guard: swap start → maximum | 329 → 704 MB (growth 375 MB) | memguard.json |

### Repeatability and functional check

All 14 generations have `image_sha256` `fe94904f2740c0901c3d8c423dd9be16946341189a51d88ab3c615179c522ae3`
(`deterministic_output: true`), the same hash as every generation of the 001 baseline; the retained
`images/measured-0.png` is pixel-identical to 001's (RGB difference 0). The 001 visual check (coherent
cat holding a sign reading "HOLLO WORLD") therefore applies unchanged. The smoke run has the same hash.

### Smoke gate

[`__attempt__20261015-140608`](../results/runs/jetson-torch-flux-klein-004__attempt__20261015-140608/):
complete; cache hits 1, misses 1. First 95831.8 ms (text_encode 74193.0 including a 69.464 s swap);
one measured 15003.5 ms (text 0.6 / denoise 14475.5 / VAE 484.9 / other 23.2 ms). Load 82.08 s.
Tegrastats RAM peak 6.057 GiB; guard MemAvailable minimum 1170 MB, swap 316 → 665 MB. Not a baseline (n = 1).

### Delta vs reference (001 baseline)

| Metric | 001 median (ms) | 004 median (ms) | Δ |
|---|---:|---:|---:|
| wall_ms | 80314.9 | 14911.4 | −65403.5 (−81.4%) |
| text_encode_ms | 1241.3 | 0.9 (retrieval) | −1240.4 |
| denoise_ms | 14795.5 | 14377.3 | −418.2 (−2.8%) |
| denoise_step_0_ms | 4015.3 | 3599.6 | −415.7 (−10.4%) |
| Steady per-step | 3590.8 | 3593.1 | +2.3 (+0.1%) |
| vae_decode_ms | 491.4 | 485.7 | −5.7 (−1.2%) |
| postprocess_ms | 47.8 | 17.1 | −30.7 |
| other_ms | 63343.9 | 30.6 | −63313.3 (−100.0%) |
| First generation wall_ms | 94586.1 | 96501.7 | +1915.6 (+2.0%) |
| peak_alloc_gib / peak_reserved_gib | 3.081 / 3.422 | 3.083 / 3.426 | +0.002 / +0.004 |
| System RAM peak (tegrastats) | 6.114 GiB | 5.982 GiB | −0.132 GiB |
| Guard MemAvailable minimum | 1206 MB | 1234 MB | +28 MB |
| Guard swap growth | 957 MB | 375 MB | −582 MB |

## Kernel profile (2026-10-17/18)

Queues [jetson_gap_queue_20261017h.sh](../scripts/jetson_gap_queue_20261017h.sh) (as the user) and [jetson_gap_queue_20261018i.sh](../scripts/jetson_gap_queue_20261018i.sh) (as root via `sudo`, after the desktop session was stopped; MemAvailable 5949 MB at start). [run_flux_portable.py](../scripts/run_flux_portable.py) `--profile`: the unchanged config's full 1+3+10 protocol, then one extra generation under `torch.profiler` (CPU + CUDA activities, shapes, memory) with the tegrastats monitor already stopped; it is excluded from `runs.csv` and `summary.json`. Trace `profile/trace.json` (Chrome format, opens in Perfetto; git-ignored, kept locally); kernel tables from [analyze_profile.py](../scripts/analyze_profile.py) in `profile/stage_kernels.md`. Kernel times are from a profiled generation and are reported as shares and per-step means, not as baseline latency.

- **Without root** ([jetson-torch-flux-klein-004__profile__20261017-215429](../results/runs/jetson-torch-flux-klein-004__profile__20261017-215429/), queue H): `complete`, but Kineto reported `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` and the trace holds CPU operators only (no kernel events). Its protocol repeat: wall median 15051.5 ms.
- **As root** ([jetson-torch-flux-klein-004__profile__20261017-222734](../results/runs/jetson-torch-flux-klein-004__profile__20261017-222734/), queue I): `complete`; 9057 kernel events. Protocol repeat: wall median 14917.7 ms (14903.9–14930.0), first 94459.2 ms; profiled generation 15329.2 ms.

Per denoise step (mean of the 4 steps of the profiled generation, [stage_kernels.md](../results/runs/jetson-torch-flux-klein-004__profile__20261017-222734/profile/stage_kernels.md)): GPU span 3691.8 ms, busy 99%, 2347 kernels.

| Kernel group | ms / step | Share | Kernels / step |
|---|---:|---:|---:|
| gemm (cuBLAS/cutlass BF16) | 1374.4 | 37.7% | 109 |
| copy/cast | 785.6 | 21.6% | 624 |
| elementwise: mul | 713.1 | 19.6% | 451 |
| elementwise: other | 588.5 | 16.1% | 583 |
| attention (flash SDPA) | 79.2 | 2.2% | 25 |
| everything else | 104.0 | 2.9% | – |

By top-level op, `aten::to` (711.2 ms), `aten::mul` (652.6), `aten::__rshift__` (339.4), `aten::__and__` (83.5) and `aten::sub` (78.9) together take 1865.8 ms per step (51% of busy time): the diffusers GGUF Q4_0 path unpacks and rescales every weight to BF16 in separate kernels on each forward; `aten::linear` plus the cutlass kernels outside an aten op carry the matmuls.

## Observations

- Median wall drops from 80.31 s to 14.91 s; the measured generation is now 96.4% denoising.
- `other_ms` falls from 63.34 s to 30.6 ms: the per-generation encoder load and transformer reload no longer
  occur after the first generation (one swap recorded in `stage-residency.json`, 13 cache hits).
- Steady per-step denoise is unchanged (3593.1 vs 3590.8 ms). Step 0 is 0.42 s shorter than in 001 and now
  within 7 ms of the steady steps.
- The first generation still pays the full swap (70.29 s) and is 1.9 s slower than 001's first generation.
- The allocator peak (3.083 GiB, VAE decode) is unchanged; on hits the text-encode stage holds 2.486 GiB
  instead of reaching 3.067 GiB, so VAE decode sets the peak of every measured generation.
- Swap growth over the run is 375 MB (001: 957 MB) and the whole-capture RAM peak is 0.13 GiB lower.
- Output is bit-identical to 001 in all 14 generations.

## Interpretation

- For a fixed prompt, the latency of 001 is almost entirely an artefact of re-encoding: with exact reuse,
  this configuration's steady generation cost is the four transformer steps plus VAE decode. The result
  holds only while the prompt and every other encoding argument stay identical; a new prompt is a miss and
  pays the full swap (about 70 s here).
- Hypothesis: the 0.42 s step-0 excess in 001 is a first-forward cost of a freshly reloaded transformer
  (allocator or kernel warm-up after reload); it disappears here because the transformer is not reloaded.
  The data do not isolate the component.
- Hypothesis: lower swap growth follows from fewer weight-streaming events (one swap instead of 14), in
  line with the page-cache hypothesis in 001; swap-in/out counters were not recorded.
- `text_encode_ms` in this record is retrieval time on hits and includes the residency swap on the miss;
  it is not comparable with 001's `text_encode_ms` without that note.

## Next experiment

- `torch.compile` on top of this configuration: [jetson-torch-flux-klein-007](jetson-torch-flux-klein-007.md).
- Rerun `jetson-torch-flux-klein-q4-nf4-512-condreuse.json` with `/proc/vmstat` swap counters, to test the swap hypothesis.
