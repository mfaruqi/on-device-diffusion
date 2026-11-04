---
type: experiment-record
id: jetson-flux-klein-013
status: failed
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-013__attempt__20261015-120112]
updated: 2026-10-15
---

# jetson-flux-klein-013: automatic placement with VAE tiling fails during text encoding

Status: **failed** (2026-10-15); the correctness attempt failed in the first prompt encoding. No
generation completed and the full 1+3+10 protocol was not started.

## Question

Does sd.cpp's automatic placement (auto-fit) produce a feasible plan for the quantized 512×512 workload
when VAE tiling is on? The reference is the untiled auto-fit attempt
[jetson-flux-klein-004](jetson-flux-klein-004.md), which failed during text encoding. Week 3 “verify
target feasibility” and RQ1: “Which diffusion optimization choices transfer across devices and
workloads?” ([proposal overview](../wiki/project/overview.md)).

## Setup

- [Configuration](../configs/jetson-flux-klein-q4-512-autofit-vaetile.json) and
  [attempt configuration](../configs/jetson-flux-klein-q4-512-autofit-vaetile-smoke.json) (identical to the
  run's [config.json](../results/runs/jetson-flux-klein-013__attempt__20261015-120112/config.json)). One change
  against the 004 attempt configuration: VAE tiling on (`optimizations.vae_tiling: true`;
  `engine_settings.vae_tile_size: 128`, `vae_tile_overlap: 0.25`; harness `--vae-tiling 1
  --vae-tile-size 128 --vae-tile-overlap 0.25`). Auto-fit on, parameter backend explicitly unset (an
  explicit backend would disable auto-fit).
- Workload: pinned sd.cpp `19bbbca`; FLUX.2 klein 4B transformer Q4_0, Qwen3-4B Q4_K_M text encoder,
  BF16 VAE; 512×512, 4 Euler steps, flux2 scheduler, CFG 1.0, seed 0, batch 1, prompt “A cat holding a
  sign that says hello world”. CUDA0 compute, eager loading, segmentation allowed, diffusion flash
  attention on, prefetch on, mmap off, no conditioning or step cache, 4 CPU threads, 25 W mode.
- Harness: tiling-capable `sd-bench` (`~/tools/sd-bench-build-tiling-20261014`), binary SHA256
  `f9ed5788425f5ce53bf6da86182fdaeba2fa00e7c1c73a2c309f2f63808607f4`
  ([environment](../results/runs/jetson-flux-klein-013__attempt__20261015-120112/environment.json)). The 004
  attempt used an earlier harness build (`sd-bench-build-20261004`, SHA256 `6f0805c6…5d47`) with the same
  engine commit and submodule pins.
- Protocol: first + one measured generation, no warm-ups (correctness gate before the full protocol).
  Weight files SHA256-read before the run (warm filesystem cache, no flush).
- Memory guard (outside the config): kill if MemAvailable < 400 MB or swap growth > 768 MB, polled every 0.1 s.
- Metrics: [baseline-metrics.md, stable-diffusion.cpp runs](../wiki/methods/baseline-metrics.md#stable-diffusioncpp-runs).

## Results

[Run directory](../results/runs/jetson-flux-klein-013__attempt__20261015-120112/),
[status](../results/runs/jetson-flux-klein-013__attempt__20261015-120112/status.json) failed,
16:01:12–16:03:27 UTC, [engine log](../results/runs/jetson-flux-klein-013__attempt__20261015-120112/engine/sdcpp.log),
[callbacks](../results/runs/jetson-flux-klein-013__attempt__20261015-120112/engine/results.jsonl),
[memguard](../results/runs/jetson-flux-klein-013__attempt__20261015-120112/memguard.json). Power mode 25 W; CUDA 12.6.

### Auto-fit plan (engine-reported) and comparison with 004

| Plan line | 013 (tiling on) | 004 (untiled) |
|---|---|---|
| CUDA0 free / budget | 6601 MiB / 6089 MiB | 6523 MiB / 6011 MiB |
| RAM free / params budget | 6601 MiB / 4553 MiB | 6523 MiB / 4475 MiB |
| Weight cache priority | diffusion > te > vae | diffusion > te > vae |
| DiT | params 2346 MiB, compute reserve 2048 MiB → compute CUDA0, params CUDA0 | same |
| Conditioner | params 2375 MiB, compute reserve 2048 MiB → compute CUDA0, params cpu | same |
| VAE | params 160 MiB, compute reserve 1024 MiB → compute CUDA0, params cpu | same |
| Resolved arguments | `--backend "CUDA0" --params-backend "te=cpu,vae=cpu"` | same |
| Prepared parameter buffers | 2536.34 MB RAM on CUDA_Host (646 tensors); 2346.40 MB VRAM on CUDA0 (149 tensors) | same |

004 values are from its [engine audit](../results/runs/jetson-flux-klein-004__attempt__20261005-000226/engine-audit.json).
The plan differs only in the free-memory and budget figures (78 MiB higher in 013); the VAE compute
reserve stays at 1024 MiB with tiling on, and every placement decision is the same.

### Failure

- Model context creation 64.679 s (load event); engine tensor loading 61.32 s (read 59.77 s).
- The first generation began text encoding with `qwen3 using 29 segments`, staged 304.28 MB of compute
  parameters to CUDA0 and queued per-segment prefetches of 54.16–60.93 MB. Segments 1–11
  (`llm.text.prelude`, `llm.text.layers.0`–`.9`) executed.
- The engine then released the staged segment parameters and reported:
  - `model manager memory on CUDA0: reported free 515.91 MB / total 7619.96 MB, tracked weights 2400.57 MB / other runtime 0.00 MB / current runtime 61.00 MB`
  - `model manager cannot make enough memory available on CUDA0: need 517.00 MB device / 66.00 MB budget, available 515.91 MB device / 3688.64 MB budget`
  - `qwen3 segment 12/29 (llm.text.layers.10) failed during weight preparation`, `LLM prompt encoding failed`,
    `failed to encode prompt`.
- The engine's available-memory probe fell from 1625.95 MB at the start of text encoding to 515.91 MB at
  the failure (1.09 MB short of the request).
- The harness recorded generation 0 as `ok: false` after 2719.574 ms with no stage callbacks. The runner
  raised `RuntimeError('Expected one load and exactly the configured number of generations')`; no summary
  was written. No image exists; VAE decoding (and therefore tiling) was never reached.
- For comparison, 004 failed at Qwen3 segment 14/29 (`llm.text.layers.12`), needing 577.93 MB against
  557.17 MB available ([failure analysis](../results/runs/jetson-flux-klein-004__attempt__20261005-000226/failure-analysis.json)).

### Memory diagnostics

- Memory guard not triggered (exit code 1 is the engine failure): MemAvailable 5982 MB at start, minimum
  496 MB; swap 307 → 313 MB.
- tegrastats (68 one-second samples, whole capture including load): RAM peak 6887 MB (6.726 GiB) of
  7620 MB, at the last-but-one sample; swap 308–313 MB; maximum GPU temperature 48.0 °C. 004: 6.699 GiB
  (68 samples). Not stage-aligned and not an allocator peak.

## Observations

- With VAE tiling on, auto-fit chose the same placement as without it: transformer parameters on CUDA0,
  text encoder and VAE parameters in host memory, all compute on CUDA0.
- The plan's VAE compute reserve is 1024 MiB in both runs.
- Text encoding failed one memory request short (1.09 MB) at segment 12/29, two segments earlier than
  004's failure at 14/29 (20.76 MB short).
- The tegrastats RAM peak (6.726 GiB) is 0.027 GiB above 004's; it falls in the last two samples of the capture, which ends at the failure.

## Interpretation

- Turning on VAE tiling does not change the auto-fit plan, and the plan fails before the VAE is used,
  so this configuration answers the question negatively: auto-fit with tiling is infeasible under the
  recorded memory conditions, for the same stage and failure type as 004.
- **Hypothesis:** the plan's fixed compute reserves (2048 MiB for each of DiT and conditioner, 1024 MiB
  for VAE) and host-resident text-encoder parameters leave too little physical headroom on this shared-memory
  device for per-segment staging during text encoding; the tiling option cannot help because the planner
  does not reduce the VAE reserve and the failure is upstream of decode. Not tested: the log does not
  separate allocator behaviour, budget model and background pressure.
- The failure point differs from 004 (segment 12 vs 14), but one attempt per configuration on different
  days does not establish whether that difference is systematic.

## Next experiment

- Repeat `jetson-flux-klein-q4-512-autofit-vaetile-smoke.json` with the engine's per-segment free-memory
  probes exported, to test whether the failure segment (12 vs 004's 14) is reproducible.

The attempt is uploaded to W&B (`mfaruqi-purdue-university/on-device-diffusion`):
[failed attempt](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-013__attempt__20261015-120112).
