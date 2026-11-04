---
type: experiment-record
id: jetson-edgedit-flux-klein-007
status: failed
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-007__attempt__20261014-170729, jetson-edgedit-flux-klein-007__baseline__20261014-171051]
updated: 2026-10-14
---

# jetson-edgedit-flux-klein-007: edge-dit.cpp flash attention off; full protocol stopped by memory guard in pass 3

Status: **failed**, with partial data (2026-10-14). The two-pass smoke completed (21:07:29–21:10:20 UTC). The full 1+3+10 run (started 21:10:51 UTC) was killed by the memory guard at about 21:13:40 UTC (MemAvailable 399 MB < 400 MB floor, exit 137) at the start of denoising in generation index 2, the first warm-up. Two passes completed; no measured-phase observation exists.

## Question

How does turning flash attention off change latency and whole-system memory for the edge-dit.cpp placement of [006](jetson-edgedit-flux-klein-006.md)? Single change against the reference config [jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json). Week 2 engine baselines (rolled over), RQ1.

## Setup

- **Config**: [jetson-edgedit-flux-klein-q4-embq8-512-vaetile-noflash.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-noflash.json); smoke [jetson-edgedit-flux-klein-q4-embq8-512-vaetile-noflash-smoke.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-noflash-smoke.json) (first + one measured). `reference_config` points to the 006 config.
- **Change vs reference (only one)**: `optimizations.flash_attention: false`, which the runner maps to `--no-flash-attention` ([command.json](../results/runs/jetson-edgedit-flux-klein-007__baseline__20261014-171051/command.json)). Everything else is identical to [006 setup](jetson-edgedit-flux-klein-006.md#setup): edge-dit.cpp `97cc585`, same binary hash; Q4_0 transformer, **Q4_K_M encoder with Q8_0 token embedding (compressed, separately labelled checkpoint)**, BF16 VAE, all weights resident in VRAM; `--backend cuda --type preserve --cache_method off --vae-tiling on --threads 4`; 512×512, 4 Euler steps, guidance 1.0, CFG 1.0, seed 0, one prompt; 25W.
- The engine log does not report the attention path in either setting; the only evidence that flash attention is off is the command-line flag. Apart from timings, output paths, the available-memory probe and load progress, the smoke engine log is line-for-line identical to the 006 smoke log.
- **VAE tiling** (engine log, every completed pass): `VAE Tile size: 8x8`, 5 × 5 = 25 tiles, overlap 0.25, as in 006.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`.
- **Runner and guard**: [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) (same hashes as 006 in [source-provenance.json](../results/runs/jetson-edgedit-flux-klein-007__baseline__20261014-171051/source-provenance.json)), launched by [jetson_gap_queue_20261014.sh](../scripts/jetson_gap_queue_20261014.sh) under [jetson_memguard.sh](../scripts/jetson_memguard.sh) around the whole runner process: SIGKILL to the process group if MemAvailable < 400 MB or swap growth > 768 MB, 0.1 s poll. The thresholds come from the queue script, not from the config. The queue runs the full protocol only after a complete smoke.
- Metric derivation (wall from `ed-sample` stdout at 1 ms rounding; stages from the engine `flux generate breakdown` line at 10 ms rounding; raw `[[phase]]` intervals in `stages.csv`; whole-system tegrastats only; final image only): [006 setup](jetson-edgedit-flux-klein-006.md#setup), [baseline metrics](../wiki/methods/baseline-metrics.md#edge-dit-ed-sample-adapter), [Jetson repeated baseline memory](../wiki/methods/baseline-metrics.md#jetson-repeated-baseline-memory).

## Results

### Full run: failure record

[jetson-edgedit-flux-klein-007__baseline__20261014-171051](../results/runs/jetson-edgedit-flux-klein-007__baseline__20261014-171051/):

- [memguard.json](../results/runs/jetson-edgedit-flux-klein-007__baseline__20261014-171051/memguard.json): `killed_by_guard: true`, reason `MemAvailable 399 MB < 400 MB`, exit code 137; MemAvailable 5996 MB at start, minimum 399 MB; swap 301 MB at start, maximum 302 MB.
- [status.json](../results/runs/jetson-edgedit-flux-klein-007__baseline__20261014-171051/status.json) still reads `running`: SIGKILL to the process group stopped the runner before it could write a final status. `summary.json`, `runs.csv`, `stages.csv`, `load.json`, `free-after.txt`, `engine/timing.json` and `output.png` were never written.
- [engine.log](../results/runs/jetson-edgedit-flux-klein-007__baseline__20261014-171051/engine.log) ends with `[[phase]] stage=denoise event=begin` for generation index 2 (21:13:38.558 UTC by the marker), with no `flux step` line after it; the last tegrastats sample is 17:13:40 local (21:13:40 UTC).

Partial measurements, from the engine log (no `runs.csv` exists; values are the engine's printed numbers):

| Pass (index) | Phase | Wall (ms) | Text / denoise / VAE / other (ms) |
|---|---|---:|---|
| 0 | first | 20393 | 2170 / 13830 / 4240 / 150 |
| 1 | warm-up | 15460 | 930 / 12680 / 1740 / 120 |
| 2 | warm-up | killed at denoise start | encode completed (condition graph 911 ms) |

Model load 62.607 s (`model load` line; `loading tensor data completed in 59.32s`). Raw marker intervals: pass 0 encode_setup 2290.4 / denoise 13830.8 / vae_decode 4239.5 ms; pass 1 1022.6 / 12677.9 / 1735.0 ms; pass 2 encode_setup 1006.2 ms.

Whole-system tegrastats ([tegrastats.log](../results/runs/jetson-edgedit-flux-klein-007__baseline__20261014-171051/tegrastats.log), 493 samples; summarized here because the runner did not reach its summary step): RAM peak 6996 / 7620 MB = 6.83 GiB at the last sample; swap 300–302 MB. Trajectory aligned to phase markers (1 s timestamp resolution): load maximum 6268 MB; pass 0 up to about 6720 MB; a step from 6607 to 6850 MB at the start of pass 1; 6871–6873 MB to the end of pass 1 and through pass 2 encode; a step from 6871 to 6994 MB at the start of pass 2 denoising; 6996 MB when killed.

### Smoke run (complete; first + one measured)

[jetson-edgedit-flux-klein-007__attempt__20261014-170729](../results/runs/jetson-edgedit-flux-klein-007__attempt__20261014-170729/) ([summary.json](../results/runs/jetson-edgedit-flux-klein-007__attempt__20261014-170729/summary.json), [runs.csv](../results/runs/jetson-edgedit-flux-klein-007__attempt__20261014-170729/runs.csv), [stages.csv](../results/runs/jetson-edgedit-flux-klein-007__attempt__20261014-170729/stages.csv)): status complete, guard not triggered, exit 0.

| Quantity | Value |
|---|---|
| First generation | wall 21433.0 ms; text 2700.0 / denoise 13910.0 / vae 4660.0 / other 160.0 |
| Measured (n = 1) | wall 15473.0 ms; text 930.0 / denoise 12690.0 / vae 1730.0 / other 120.0 |
| Model load | 63.960773 s ([load.json](../results/runs/jetson-edgedit-flux-klein-007__attempt__20261014-170729/load.json)) |
| RAM peak (tegrastats, 500 samples) | 6852 / 7620 MB = 6.69 GiB |
| Swap (tegrastats) | 297–398 MB (0.29–0.39 GiB) |
| MemAvailable (guard) | 5986 MB at start, **minimum 615 MB**; swap 296 → max 398 MB (growth 102 MB) ([memguard.json](../results/runs/jetson-edgedit-flux-klein-007__attempt__20261014-170729/memguard.json)) |

The smoke's RAM trajectory matches the first two passes of the full run (step 6679 → 6826 MB at the start of pass 1; peak 6852 MB). It ended after pass 1, before the point where the full run's step at pass 2 denoising occurred.

### Delta vs reference (006)

No measured-phase median exists for this configuration. The table uses single post-first observations against the 006 measured median ([006 summary.json](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/summary.json)); these are n = 1 indications, not a variant result.

| Stage | 006 median (ms) | 007 smoke measured, n = 1 (ms) | Δ | 007 full pass 1 (warm-up), n = 1 (ms) | Δ |
|---|---:|---:|---:|---:|---:|
| wall | 11962.5 | 15473.0 | +3510.5 (+29.3%) | 15460 | +3497.5 (+29.2%) |
| text_encode | 930.0 | 930.0 | 0 | 930 | 0 |
| denoise | 9185.0 | 12690.0 | +3505.0 (+38.2%) | 12680 | +3495.0 (+38.1%) |
| vae_decode | 1740.0 | 1730.0 | −10.0 (−0.6%) | 1740 | 0 |
| other | 120.0 | 120.0 | 0 | 120 | 0 |
| first generation (wall) | 17395.0 | 21433.0 | +4038.0 (+23.2%) | 20393 | +2998.0 (+17.2%) |

Memory against 006 ([006 memory](jetson-edgedit-flux-klein-006.md#memory-whole-system-tegrastats-1171-samples-at-200-ms-whole-capture-including-load)): MemAvailable minimum 615 MB (007 smoke) vs 697 MB (006 smoke), and 399 MB (007 full, killed in pass 2) vs 584 MB (006 full, 14 passes). During pass 1 tegrastats RAM was about 6850–6873 MB here against 6727–6756 MB in the 006 full run. At pass 2 encode both runs were at about 6862–6872 MB; 006 then stayed flat, while this configuration stepped to 6994 MB as denoising began.

### Functional check

Smoke final image (run index 1) RGB SHA256 `b7589892b8b6fc10727dd697dc4753ebea5e264b70c80044511a9b0527a699c9`, which differs from the 006 final image (`7c7a84f3…`): turning flash attention off changes output pixels. `output.png` is not present in the local run directory, so no visual check was made. The killed run retained no image. No quality metric.

## Observations

- With flash attention off, the smoke's measured pass took 15473 ms; nearly the whole increase over 006 is in denoising (+3.5 s over 4 steps), while text encode and tiled VAE decode are unchanged within 10 ms rounding.
- The full run was stopped by the memory guard 1 MB below the 400 MB floor, at the start of denoising in the third pass, after two complete passes.
- The smoke completed with a 615 MB MemAvailable minimum; it covered only the first two passes and so did not reach the pass-2 memory step.
- Swap stayed near its starting value in the full run (301–302 MB); the kill came from MemAvailable, not swap growth.
- `status.json` of the full run is stale (`running`) because of the process-group SIGKILL; `memguard.json` and `engine.log` are the failure evidence.

## Interpretation

- Flash attention on (006) is the faster and lower-memory setting for this placement: roughly 0.9 s more per denoise step without it in two single observations, and the full protocol does not fit under the guard.
- Hypothesis (not measured): without flash attention the engine materializes the attention-score tensor for each attention call. With 24 heads and an assumed 1536-token joint sequence (1024 image + 512 text tokens, from `image_seq_len=1024` and `cross_attn=[7680, 512]`), one f32 score tensor is about 216 MiB, the same order as the extra 100–250 MB steps seen here. The engine's compute-buffer sizes are not logged, so this cannot be confirmed from these runs.
- Hypothesis: the RAM step at pass 2 is the same kind of compute-buffer or allocator growth seen in 006's first passes, larger here; a two-pass smoke cannot show whether the run would have reached a plateau below the floor.
- The failure is a capacity result for this configuration under the stated guard; it is not evidence of an engine error.

## Next experiment

- None planned for flash attention off on this placement; [006](jetson-edgedit-flux-klein-006.md) remains the reference.
- If it is revisited: a copy of `jetson-edgedit-flux-klein-q4-embq8-512-vaetile-noflash-smoke.json` with at least four passes (`warmup_runs: 2`), so the memory step at pass 2 is seen before a full run.
