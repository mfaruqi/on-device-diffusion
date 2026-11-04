---
type: experiment-record
id: jetson-torch-flux-klein-base-001
status: complete
device: jetson-orin-nano
engine: pytorch-diffusers
runs: [jetson-torch-flux-klein-base-001__baseline__20261016-050136, jetson-torch-flux-klein-base-001__attempt__20261016-044214, jetson-torch-flux-klein-base-001__attempt__20261016-011604, jetson-torch-flux-klein-base-001__attempt__20261015-165022]
updated: 2026-10-16
---

# jetson-torch-flux-klein-base-001: PyTorch/diffusers klein Base reference, Q4_0 GGUF + NF4 encoder 512², 50 steps CFG 4, sequential stage residency; 494.6 s per image, 26% of it weight swaps

Status: **complete** (2026-10-16; full run 09:01:36–10:59:00 UTC, smoke 08:42:14–09:00:56 UTC; run directly on the device). Full 1+3+10 protocol; no memory-guard intervention. Two earlier attempts failed in the runner or the snapshot, not the device (Setup).

## Question

What does the PyTorch/diffusers pipeline cost per image on the Jetson for the klein Base workload (50 steps, CFG 4) under sequential stage residency (encoder loaded only for encoding, transformer released while it runs), with no reuse, cache or compiler? Reference for [base-002](jetson-torch-flux-klein-base-002.md) … [base-004](jetson-torch-flux-klein-base-004.md). Engine baseline (RQ1) and reference for residency/reuse levers under a memory budget (RQ2).

## Setup

- **Config**: [jetson-torch-flux-klein-base-q4-nf4-512.json](../configs/jetson-torch-flux-klein-base-q4-nf4-512.json); smoke [jetson-torch-flux-klein-base-q4-nf4-512-smoke.json](../configs/jetson-torch-flux-klein-base-q4-nf4-512-smoke.json) (first + one measured). Reference configuration.
- **Pipeline**: diffusers 0.40.0 `Flux2KleinPipeline` from `black-forest-labs/FLUX.2-klein-base-4B` at `a3b4f484…` (configs, tokenizer, scheduler and VAE only); transformer from the GGUF Q4_0 file of `leejet/FLUX.2-klein-base-4B-GGUF` (sha256 `c3a28545…`, BF16 compute); Qwen3 encoder from a pre-quantized bitsandbytes NF4 copy (sha256 `eaa2048d…`); BF16 VAE. **Compressed transformer and encoder (separately labelled workload).**
- **Residency** (`optimizations.stage_residency`): `text_encoder: release_after_encode`, `transformer: release_during_encode`. [run_flux_portable.py](../scripts/run_flux_portable.py) `apply_stage_residency()` wraps every `encode_prompt` call: free the transformer, load the encoder, encode, free the encoder, reload the transformer. Under CFG the pipeline calls `encode_prompt` twice per generation (prompt, then the empty negative prompt), so each generation performs **two** full swaps ([stage-residency.json](../results/runs/jetson-torch-flux-klein-base-001__baseline__20261016-050136/stage-residency.json): 28 entries for 14 generations). The swaps fall between timed stages and are counted in `other_ms`.
- **Workload**: 512×512, 50 steps (FlowMatch Euler), guidance 4.0, empty negative prompt, `max_sequence_length` 512, seed 0, batch 1, one prompt. SDPA default backend, no VAE tiling, no compiler.
- **Stack**: Jetson AI Lab jp6/cu126: torch 2.8.0, triton 3.4.0, bitsandbytes 0.48.0.dev0, diffusers 0.40.0, transformers 5.16.1; Python 3.10.12; `HF_HUB_OFFLINE=1`; 25W ([environment.json](../results/runs/jetson-torch-flux-klein-base-001__baseline__20261016-050136/environment.json)).
- **Protocol**: one process, one load; first, 3 warm-ups, 10 measured. Launched by [jetson_gap_queue_20261016g.sh](../scripts/jetson_gap_queue_20261016g.sh) under [jetson_memguard.sh](../scripts/jetson_memguard.sh) (400 MB floor, 1536 MB swap-growth limit); nothing else ran on the board.
- **Runner**: run_flux_portable.py sha256 `db0cfd9a…` ([source-provenance.json](../results/runs/jetson-torch-flux-klein-base-001__baseline__20261016-050136/source-provenance.json)). It differs from the version used for base-002 … base-004 (`4a254728…`) only in the post-run check of the number of residency swaps, which now expects two per generation under CFG.
- **Earlier attempts** (not measurements):
  - [jetson-torch-flux-klein-base-001__attempt__20261016-011604](../results/runs/jetson-torch-flux-klein-base-001__attempt__20261016-011604/) (2026-10-16 05:16 UTC): `failed`, `AssertionError('encoder reloads 4 != 2')`. Both generations completed (first 502.1 s, one measured 497.1 s; image hash `2a0803ab…`); the runner's check expected one swap per generation and stopped the run before writing `summary.json`.
  - [jetson-torch-flux-klein-base-001__attempt__20261015-165022](../results/runs/jetson-torch-flux-klein-base-001__attempt__20261015-165022/) (2026-10-15): `failed`, `IncompleteSnapshotError` for the pinned Base snapshot; the config's `allow_patterns` requested weight files not needed here and was narrowed to configs, tokenizer, scheduler and VAE.
- Metric definitions: [portable PyTorch runner](../wiki/methods/baseline-metrics.md) (stage recorder; allocator peaks from `torch.cuda`; whole-system tegrastats on Jetson).

## Results

Full run [jetson-torch-flux-klein-base-001__baseline__20261016-050136](../results/runs/jetson-torch-flux-klein-base-001__baseline__20261016-050136/): [status.json](../results/runs/jetson-torch-flux-klein-base-001__baseline__20261016-050136/status.json) `complete`; [memguard.json](../results/runs/jetson-torch-flux-klein-base-001__baseline__20261016-050136/memguard.json) not killed, exit 0. `transformer_calls_per_step` 2.

### Latency (measured n = 10; [summary.json](../results/runs/jetson-torch-flux-klein-base-001__baseline__20261016-050136/summary.json), [runs.csv](../results/runs/jetson-torch-flux-klein-base-001__baseline__20261016-050136/runs.csv), [stages.csv](../results/runs/jetson-torch-flux-klein-base-001__baseline__20261016-050136/stages.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall | **494634.4** | 492707.3–497133.7 (stdev 1509.1) | 100% |
| text_encode (2 calls) | 2742.1 | 2393.9–4343.9 | 0.6% |
| denoise (50 steps × 2 transformer calls) | 360239.1 | 357511.1–360988.1 | 72.8% |
| vae_decode | 1174.0 | 1103.0–1369.2 | 0.2% |
| postprocess | 47.7 | 31.3–86.2 | 0.0% |
| other (includes the two residency swaps) | 130418.8 | 128519.6–133051.7 | 26.4% |
| First generation: wall | 504094.0 | – | text 8139.4 / denoise 361857.1 / vae 3922.7 / other 130076.8 |
| Warm-ups (3, discarded): wall | – | 491934.5–500195.7 | – |
| Pipeline load (s) | 76.03 | – | page cache may be warm |

Steady step (mean of steps 1–49, median over measured runs): 7193.5 ms; first generation step 0: 8687.5 ms. One residency swap (first entry): transformer free 0.80 s, encoder load 29.61 s, encoder free 0.48 s, transformer reload 32.63 s; per-swap totals 63.5–68.3 s in the first six entries.

### Memory

| Quantity | Value |
|---|---:|
| Peak allocated / reserved (torch, measured median) | 3.09 / 3.44 GiB |
| RAM peak (tegrastats, whole capture) | 5.966 GiB of 7.44 GiB |
| Swap (tegrastats) min–max | 0.312–1.204 GiB |
| MemAvailable (guard), start → minimum | 5992 → 1350 MB |
| Swap (guard), start → maximum | 322 → 1233 MB |
| GPU temperature max | 65.8 °C |

### Smoke run

[jetson-torch-flux-klein-base-001__attempt__20261016-044214](../results/runs/jetson-torch-flux-klein-base-001__attempt__20261016-044214/): complete, guard not triggered. First 507047.6 ms; one measured 499232.0 ms (text 3748.5 / denoise 360367.9 / vae 1065.7 / other 133971.4). Load 91.17 s. Not a baseline (n = 1).

### Repeatability and functional check (visual, not a quality metric)

All 14 generations of the full run and both smoke generations have RGB SHA256 `2a0803ab87940098…`; `deterministic_output` true. Visual check of `images/first.png`: tabby cat holding a cardboard sign reading "HELLO WORLD" in bold capitals; no artefact. No quality metric.

## Observations

- Denoise is 72.8% of the median wall and the two residency swaps (in `other`) are 26.4%; text encode itself is 2.7 s.
- Each swap costs about 64–68 s, roughly half encoder load and half transformer reload; two swaps per image follow from CFG's two `encode_prompt` calls.
- Swap use grew by up to 0.91 GiB during the run while MemAvailable stayed at or above 1350 MB.
- Output bit-identical across all 16 generations in two processes.

## Interpretation

- Under this residency policy, a quarter of every Base image is spent moving weights, not computing; the policy swaps per `encode_prompt` call, so CFG doubles that cost. Reusing the encoding (or encoding both prompts within one swap) removes it; the transformer step (7.19 s for two calls) is the remaining floor.
- Allocator peak is 3.09 GiB, well under the 7.44 GiB board; the residency policy, not the denoise working set, is what keeps the encoder and transformer apart.

## Next experiment

- Copy of `jetson-torch-flux-klein-base-q4-nf4-512.json` with the residency wrapper encoding the prompt and negative prompt within one swap (one swap per generation; runner change).
