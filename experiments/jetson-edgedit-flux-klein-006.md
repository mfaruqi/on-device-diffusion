---
type: experiment-record
id: jetson-edgedit-flux-klein-006
status: complete
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-006__attempt__20261014-165850, jetson-edgedit-flux-klein-006__baseline__20261014-170149]
updated: 2026-10-14
---

# jetson-edgedit-flux-klein-006: edge-dit.cpp 1+3+10 baseline, Q8_0-embedding encoder, VAE tiling on, flash attention on

Status: **complete** (2026-10-14, 21:01:49–21:06:59 UTC; smoke 20:58:50–21:01:18 UTC). Full 1+3+10 protocol in one `ed-sample` process with all weights resident; no memory-guard intervention.

## Question

What are the latency and whole-system memory of the first feasible edge-dit.cpp placement on this device under the repeated protocol? This is the repeated baseline of the configuration admitted functionally in [005](jetson-edgedit-flux-klein-005.md) (which ran one generation, no config file). It is the reference for edge-dit.cpp variants on this device. Week 2 engine baselines (rolled over), RQ1.

## Setup

- **Config**: [jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json); smoke [jetson-edgedit-flux-klein-q4-embq8-512-vaetile-smoke.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-smoke.json) (first + one measured). Copies in each run directory ([config.json](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/config.json)). No `*.resolved.json`; components are pinned by SHA256 in the config and re-hashed by the runner before launch.
- **Engine**: edge-dit.cpp `97cc585d9dbd9ac5e30b78ef2e2c4a0d8bfbe9f2`, CUDA performance build ([build evidence](../results/runs/jetson-edgedit-build-20261011/)), `ed-sample` binary SHA256 `5936674c…ea358992`; the runner verifies the source commit, a clean tree and the binary hash ([environment.json](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/environment.json)).
- **Checkpoints**: transformer `flux-2-klein-4b-Q4_0.gguf` (leejet/FLUX.2-klein-4B-GGUF `3b1f5a9d`); VAE `flux2-vae.safetensors` (black-forest-labs/FLUX.2-klein-4B `e7b7dc27`), BF16; text encoder **Qwen3-4B Q4_K_M with Q8_0 token embedding** (`Qwen3-4B-Q4_K_M-embq8_0.gguf`, SHA256 `d83c059b…f867686`), a **compressed, separately labelled checkpoint** derived from unsloth/Qwen3-4B-GGUF `22c9fc8a` by converting only `token_embd.weight` Q6_K → Q8_0 ([conversion log](../results/runs/jetson-edgedit-convert-20261011/requant.log)). Engine-confirmed weight types: transformer `q4_0:80|bf16:69`, conditioner `f32:145|q8_0:1|q4_K:216|q6_K:36`, VAE `bf16:248`. Activation dtype f32 (`bf16_tensors=69/149`).
- **Workload**: 512×512, 4 Euler steps, scheduler auto (Flux2, `flux2_mu=2.031`; sigmas 1.000000 → 0.958085 → 0.883982 → 0.717497 → 0.000000), guidance 1.0, CFG 1.0, batch 1, seed 0, prompt "A cat holding a sign that says hello world". A lower resolution than the 1024² reference; separately labelled workload.
- **Placement and options** ([command.json](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/command.json)): `--backend cuda --type preserve --cache_method off --vae-tiling on --threads 4`, flash attention left at the engine default (on; `--no-flash-attention` absent). All three components resident in GPU-visible memory for the whole process: parameter buffers transformer 2346.40 MB, qwen3 2465.75 MB, VAE 94.72 MB, all VRAM (engine log). No offload, auto-fit or auto-allocate.
- **VAE tiling** (engine log, every pass): `VAE Tile size: 8x8`, `num tiles : 5, 5`, `optimal overlap : 0.250000, 0.250000 (targeting 0.250000)`, `processing 25 tiles`. The log does not state the unit of the tile size. Tiling is required for this placement to complete ([004](jetson-edgedit-flux-klein-004.md): untiled decode stopped by the memory guard).
- **Change vs reference**: none; this is the reference configuration for edge-dit.cpp on this device (`reference_config: null`).
- **Protocol**: one `ed-sample` process, one model load, `--warmup 0 --repeat 14`: pass 1 reported as first, passes 2–4 discarded warm-ups, passes 5–14 measured. Smoke: `--repeat 2` (first + one measured).
- **Runner**: [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) (hashes in [source-provenance.json](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/source-provenance.json); they match the committed-tree copy). Launched by [jetson_gap_queue_20261014.sh](../scripts/jetson_gap_queue_20261014.sh) under [jetson_memguard.sh](../scripts/jetson_memguard.sh) wrapping the whole runner process: kill if MemAvailable < 400 MB or swap growth > 768 MB, 0.1 s poll. Guard thresholds are set by the queue script, not by the config.
- **Device**: Jetson Orin Nano 8 GB, L4T R36.4.4, NV power mode 25W (checked by the runner), driver 540.4.0, `HF_HUB_OFFLINE=1`. `environment.json` `git_commit` is empty (the device copy is not a git checkout); provenance is the script hashes above.
- **Metric derivation** (runner docstring and code):
  - `wall_ms`: `ed-sample` per-pass stdout (`[ed-sample] pass i/N … X.XXXs`), upstream `steady_clock` around `ed_generate_image`, **rounded to 1 ms**. The runner checks the sum against `engine/timing.json` `e2e_time.total`.
  - `text_encode_ms`, `denoise_ms`, `vae_decode_ms`, `other_ms`: the engine's own `flux generate breakdown: total=… | text_encode=… sampling=… vae_decode=… other=…` line, printed in seconds with two decimals, so **10 ms rounding**. `denoise_ms` is the engine's `sampling` interval.
  - `stages.csv`: raw `[[phase]]` host `system_clock` marker intervals (`encode_setup`, `denoise`, `vae_decode`) plus `engine_breakdown_total`, kept unrounded as evidence; not GPU events. `encode_setup` includes conditioning plus latent/schedule setup and is wider than `text_encode`.
  - Memory: whole-system tegrastats at 200 ms only ([tegrastats-samples.csv](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/tegrastats-samples.csv)); printed MB treated as MiB and divided by 1024 ([jetson_device.py](../scripts/jetson_device.py)). No per-process, per-generation or allocator peaks.
  - Only the final pass's image is retained (upstream overwrites `img_000000.png` each pass), so determinism across passes is not established (`deterministic_output: null`).
  - Flash attention is evidenced only by the command line (flag absent → engine default); the engine log does not report the attention path.
  - The runner rejects a log with missing or reordered phase markers, a changed schedule, a changed conditioner weight-type stat, or a tile count that disagrees with the config.
- Shared definitions: [baseline metrics, edge-dit adapter](../wiki/methods/baseline-metrics.md#edge-dit-ed-sample-adapter) and [Jetson repeated baseline memory](../wiki/methods/baseline-metrics.md#jetson-repeated-baseline-memory). That page describes the A100 adapter, under which `text_encode_ms` and `other_ms` are unavailable; on this runner they come from the engine breakdown line as described above.

## Results

Full run [jetson-edgedit-flux-klein-006__baseline__20261014-170149](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/): [status.json](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/status.json) `complete`; [memguard.json](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/memguard.json) not killed, exit 0. 14 passes, 14 breakdown lines, 14 × 25 VAE tiles; schedule identical in every pass.

### Latency (measured n = 10; [summary.json](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/summary.json), [runs.csv](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/runs.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall (1 ms rounding) | 11962.5 | 11943.0–12026.0 | 100% |
| text_encode (engine, 10 ms rounding) | 930.0 | 910.0–930.0 | 7.8% |
| denoise (engine `sampling`, 4 steps) | 9185.0 | 9160.0–9230.0 | 76.8% |
| vae_decode (tiled, 25 tiles) | 1740.0 | 1720.0–1750.0 | 14.5% |
| other (engine) | 120.0 | 110.0–120.0 | 1.0% |
| First generation (wall) | 17395.0 | – | text 2280.0 / denoise 10470.0 / vae 4500.0 / other 140.0 |
| Model load (one context, s) | 64.705868 | – | [load.json](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/load.json) |

Wall stdev 25.6 ms. Warm-ups: 12069.0, 11996.0, 11958.0 ms. Medians of the raw host-marker intervals over the same 10 passes ([stages.csv](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/stages.csv)): encode_setup 1024.5 (1006.6–1027.9), denoise 9185.1 (9164.0–9229.1), vae_decode 1743.5 (1718.7–1752.1) ms. Load: the runner hashes all weight files immediately before launch, so the page cache is warm for this load; the engine reports `loading tensor data completed in 61.52s` inside it.

### Memory (whole-system tegrastats, 1171 samples at 200 ms, whole capture including load)

| Quantity | Value |
|---|---|
| Parameter buffers (engine log, VRAM) | transformer 2346.40 MB, qwen3 2465.75 MB, VAE 94.72 MB |
| RAM peak (tegrastats) | 6881 / 7620 MB = 6.72 / 7.44 GiB (`system_memory.ram_peak_gib` 6.7197) |
| Swap (tegrastats) | 297 MB throughout (0.29 GiB) |
| MemAvailable (guard, 0.1 s poll) | 5973 MB at start, minimum 584 MB (guard floor 400 MB) |
| Before / after (`free -h`) | available 6.4 Gi / 5.8 Gi ([free-before.txt](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/free-before.txt), [free-after.txt](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/free-after.txt)) |
| GPU temperature max | 61.5 °C |

RAM trajectory aligned to phase markers (tegrastats timestamps have 1 s resolution; [tegrastats.log](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/tegrastats.log)): maximum during load 6275 MB; during pass 0 up to about 6600 MB; a step to 6727 MB at the start of pass 1 and to 6862 MB at the start of pass 2; then 6860–6881 MB for passes 2–13. Allocated, reserved and per-generation device peaks are unavailable.

### Smoke run

[jetson-edgedit-flux-klein-006__attempt__20261014-165850](../results/runs/jetson-edgedit-flux-klein-006__attempt__20261014-165850/) ([summary.json](../results/runs/jetson-edgedit-flux-klein-006__attempt__20261014-165850/summary.json)): complete, guard not triggered. First 17368.0 ms (text 2270.0 / denoise 10670.0 / vae 4310.0 / other 120.0); one measured 12090.0 ms (940.0 / 9270.0 / 1750.0 / 120.0). Load 66.002942 s. RAM peak 6757 MB (6.60 GiB); MemAvailable minimum 697 MB; swap 297–298 MB. Not a baseline (n = 1).

### Repeatability and functional check

- Measured wall range 83 ms (0.7% of the median); the first generation is 5432.5 ms slower than the measured median, mostly in text encode (+1350 ms), denoise (+1285 ms) and VAE decode (+2760 ms).
- Final-pass image (run index 13) RGB SHA256 `7c7a84f3fe9988c8f43f76d656273ceb102b893cee89d4c14615116eeeaabd32` ([image-retention.json](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/image-retention.json)). The smoke run's final image has the same hash: the final pass reproduces across two processes. Earlier passes are overwritten, so within-run determinism is not established.
- `output.png` is not present in either local run directory (images are git-ignored and were not copied back), so no visual check was made for these runs; the hash is the only image evidence. The functional image check for this configuration is in [005](jetson-edgedit-flux-klein-005.md#functional-check). No quality metric (`formal_quality_eligible: false`).

## Observations

- The full 14-pass protocol completed with all weights resident; the lowest MemAvailable was 584 MB, 184 MB above the guard floor, and swap did not change.
- Denoising is 76.8% of the measured median wall; tiled VAE decode 14.5%; text encode 7.8%.
- The first generation pays about 1.35 s more text encode and 2.76 s more VAE decode than the measured passes.
- Measured wall is tight: 11943–12026 ms, stdev 25.6 ms; warm-ups were already within 107 ms of the median.
- System RAM rose in steps over the first three passes (consecutive samples 6562 → 6727 MB at the start of pass 1 and 6756 → 6862 MB at the start of pass 2) and then stayed within 21 MB to the end.
- Model load (64.7 s) is about 5.4 measured generations.
- `summary.json` `residency` reads "one load, 14 passes" in both runs; it is a fixed string in the runner, and the smoke ran 2 passes.

## Interpretation

- For this placement the steady-state cost is dominated by the four transformer steps (about 2.3 s per step from 9185 ms / 4); levers that reduce per-step cost or step count act on about three quarters of the time.
- Tiled decode costs 1.74 s per image once warm. Whether a larger tile would be faster here is open; this record cannot test it because untiled decode does not fit ([004](jetson-edgedit-flux-klein-004.md)).
- Hypothesis: the RAM steps over the first passes are engine compute-buffer or allocator growth after the first use of each graph, not weight loading (weights were loaded before pass 0). The plateau from pass 2 on sets the effective headroom for this configuration, about 0.57 GiB MemAvailable at minimum; a two-pass smoke does not reach that plateau.
- The text encoder output for this checkpoint is not validated against the reference Q4_K_M file; the converted encoder remains a separately labelled checkpoint.

## Next experiment

- Encoder-output comparison of the Q8_0-embedding encoder against the reference Q4_K_M file (still open from [005](jetson-edgedit-flux-klein-005.md#next-experiment)).
- Step-cache variant: copy of `jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json` with `step_cache` set as the single change.
- Flash attention off: [007](jetson-edgedit-flux-klein-007.md).
