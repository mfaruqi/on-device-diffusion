---
type: experiment-record
id: jetson-edgedit-flux-klein-005
status: partial
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-005__attempt__20261011-embq8-vaetile, jetson-edgedit-convert-20261011]
updated: 2026-10-14
---

# jetson-edgedit-flux-klein-005: edge-dit.cpp, all-GPU, Q8_0-embedding text encoder, VAE tiling on

Status: **partial** (2026-10-11, 15:09:09–15:10:34 UTC). Functional admission only: one generation completed with exit code 0 and a 512×512 image. This is not a baseline; no 1+3+10 protocol was run, and the phase durations below are host-clock markers from a single first generation.

## Question

Does enabling VAE tiling admit the converted-encoder, all-GPU configuration of [004](jetson-edgedit-flux-klein-004.md) through a complete generation on this device? Single change against 004: `--vae-tiling on`. Week 2 engine baselines (rolled over), RQ1.

## Setup

- **Workload label**: compressed text-encoder checkpoint, **Qwen3-4B Q4_K_M with a Q8_0 token embedding** (`Qwen3-4B-Q4_K_M-embq8_0.gguf`, SHA256 `d83c059b…f867686`), separately labelled from the reference Q4_K_M file. Conversion and its error figures: [004 setup](jetson-edgedit-flux-klein-004.md#setup) and [requant.log](../results/runs/jetson-edgedit-convert-20261011/requant.log).
- Same build, transformer, VAE, workload and fixed options as [004](jetson-edgedit-flux-klein-004.md) ([model-sha256.txt](../results/runs/jetson-edgedit-flux-klein-005__attempt__20261011-embq8-vaetile/model-sha256.txt)): edge-dit.cpp `97cc585`, Q4_0 transformer, BF16 VAE; 512×512, 4 steps, guidance 1.0, CFG 1.0, seed 0, prompt "A cat holding a sign that says hello world"; `--backend cuda --type preserve --vae-tiling off --flash-attention --cache off --threads 4`.
- **Change vs 004**: `--vae-tiling on` appended after the fixed options ([extra-args.txt](../results/runs/jetson-edgedit-flux-klein-005__attempt__20261011-embq8-vaetile/extra-args.txt)). The command therefore carries both `--vae-tiling off` and `--vae-tiling on`; the engine log shows tiled decode, so the later value took effect. Tile size default (`VAE Tile size: 8x8`).
- Launched with [jetson_edgedit_smoke.sh](../scripts/jetson_edgedit_smoke.sh) under [jetson_memguard.sh](../scripts/jetson_memguard.sh) (MemAvailable < 400 MB or swap growth > 768 MB, 0.1 s poll). Tegrastats at 200 ms. One generation; no warm-ups, no repeats. No `configs/*.json` exists.
- Device: Jetson Orin Nano 8 GB, JetPack R36.4.4, 25W ([status.json](../results/runs/jetson-edgedit-flux-klein-005__attempt__20261011-embq8-vaetile/status.json)). Phase-marker semantics: [baseline metrics](../wiki/methods/baseline-metrics.md#edge-dit-ed-sample-adapter); memory: [Jetson diagnostics](../wiki/methods/baseline-metrics.md#jetson-cli-feasibility-diagnostics).

## Results

[exit_code.txt](../results/runs/jetson-edgedit-flux-klein-005__attempt__20261011-embq8-vaetile/exit_code.txt): `0`. [memguard.json](../results/runs/jetson-edgedit-flux-klein-005__attempt__20261011-embq8-vaetile/memguard.json): not killed; MemAvailable 6184 MB at start, minimum 762 MB; swap used 63 MB at start, maximum 110 MB. [kernel-oom.txt](../results/runs/jetson-edgedit-flux-klein-005__attempt__20261011-embq8-vaetile/kernel-oom.txt) is empty.

### Latency (single first generation, host-clock markers)

From [ed-cli.log](../results/runs/jetson-edgedit-flux-klein-005__attempt__20261011-embq8-vaetile/ed-cli.log) `system_clock` phase markers; not medians, not GPU-event durations.

| Phase | Duration (s) |
|---|---:|
| encode_setup | 2.564 |
| denoise (4 steps) | 10.630 |
| vae_decode (tiled) | 4.378 |
| first encode begin → decode end | 17.590 |

The engine's own breakdown line ([ed-cli.stderr](../results/runs/jetson-edgedit-flux-klein-005__attempt__20261011-embq8-vaetile/ed-cli.stderr)) reads `total=17.59s | text_encode=2.47s sampling=10.63s vae_decode=4.38s other=0.12s`; its `text_encode` is narrower than the `encode_setup` marker interval. Tiled decode: 5×5 = 25 tiles, overlap 0.25; the engine reports `computing vae decode graph completed, taking 4.37s`.

`[ED_WALL] load=63407ms gen=0ms save=0ms free=502ms`: `gen=0ms` contradicts the nonzero phase markers and is not used. `load=63407ms` is the engine's model-load interval for this single process; the filesystem cache state is not controlled (the smoke script hashes all three weight files immediately before launch).

### Memory

Whole-system [tegrastats](../results/runs/jetson-edgedit-flux-klein-005__attempt__20261011-embq8-vaetile/tegrastats.log) (408 samples, 200 ms; [peak-ram.txt](../results/runs/jetson-edgedit-flux-klein-005__attempt__20261011-embq8-vaetile/peak-ram.txt)): RAM peak 6701/7620 MB (11:10:28 local, near the denoise→decode boundary), swap peak 111/3810 MB. From 11:10:19 local (early in denoising) to the end of decode at 11:10:32, every sample was between 6635 and 6701 MB. Parameter buffers (engine log): transformer 2346.40 MB, qwen3 2465.75 MB, VAE 94.72 MB, all VRAM. Memory before launch ([free-before.txt](../results/runs/jetson-edgedit-flux-klein-005__attempt__20261011-embq8-vaetile/free-before.txt), MiB): available 6186, swap used 63. These are whole-capture diagnostics including load; no per-stage or allocator peak is available.

### Functional check

`output.png` (git-ignored, kept in the run directory) is 512×512 RGB, SHA256 `4043647867036b8b155aa4ed684cb550c7a8e0c383c0be0a32561994d074815f`. Informal inspection: a coherent cat holding a sign; the sign reads "helllo world" (one extra "l"). Single image; no determinism check and no quality metric.

Schedule (engine log): latent 32x32, image_seq_len 1024, flux2_mu 2.031, sigmas 1.000000 → 0.958085 → 0.883982 → 0.717497 → 0.000000. Activation dtype f32 (bf16_tensors=69/149).

## Observations

- The configuration completed one generation with all three components resident on the GPU and stayed above the guard threshold (minimum MemAvailable 762 MB).
- Peak RAM (6701 MB) is lower than the 6837 MB peak of the untiled decode attempt in [004](jetson-edgedit-flux-klein-004.md), and tiled decode did not raise RAM above the denoising plateau.
- Denoising took 10.630 s by markers and 10.63 s by the engine's sampling line; encode_setup 2.564 s; tiled decode 4.378 s.
- Model load (63407 ms) dominates the single-process duration of about 85 s (start to end).
- The image is coherent but misspells the sign text.

## Interpretation

- With the Q8_0-embedding encoder and VAE tiling on, the edge-dit build admits the quantized 512×512 workload on this device; this establishes feasibility, not performance.
- Denoising accounts for about 60% of the marked generation interval in this single observation (10.630 of 17.590 s); a repeated run is needed before treating that split as stable.
- The text error is a functional caveat for this checkpoint and engine; whether it persists across seeds or depends on the converted encoder is not established by one image.

## Next experiment

- Full 1+3+10 protocol on this configuration with a labelled config file and the edge-dit adapter.
- Encoder-output comparison of the converted Q8_0-embedding encoder against the reference Q4_K_M file, to isolate the "helllo" caveat.
