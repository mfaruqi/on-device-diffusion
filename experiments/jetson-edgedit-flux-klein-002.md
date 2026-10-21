---
type: experiment-record
id: jetson-edgedit-flux-klein-002
status: failed
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-002__attempt__20261011-autofit]
updated: 2026-10-14
---

# jetson-edgedit-flux-klein-002: edge-dit.cpp with `--auto-fit`, quantized 512×512

Status: **failed** (2026-10-11, started 06:36:31 UTC); the board stopped responding during text encoding and rebooted. No exit code, no image, no timing.

## Question

Does edge-dit.cpp's fully automatic fitting (`--auto-fit`, no `--max-vram`) admit the quantized distilled klein workload on this device? Single change against the all-GPU reference [001](jetson-edgedit-flux-klein-001.md) (gate E2, rung 2). Week 2 engine baselines (rolled over), RQ1.

## Setup

- Same build, weights, workload and fixed options as [001](jetson-edgedit-flux-klein-001.md): edge-dit.cpp `97cc585`, Q4_0 transformer, Qwen3-4B Q4_K_M text encoder, BF16 VAE ([model-sha256.txt](../results/runs/jetson-edgedit-flux-klein-002__attempt__20261011-autofit/model-sha256.txt)); 512×512, 4 steps, guidance 1.0, CFG 1.0, seed 0; `--backend cuda --type preserve --vae-tiling off --flash-attention --cache off --threads 4`.
- **Change vs reference**: `--auto-fit` ([extra-args.txt](../results/runs/jetson-edgedit-flux-klein-002__attempt__20261011-autofit/extra-args.txt)). Per the engine's help text this option chooses text-encoder/transformer quantization as well as placement; it is an engine option, not a pre-converted checkpoint.
- Launched with [jetson_edgedit_smoke.sh](../scripts/jetson_edgedit_smoke.sh)-style capture (`free -m` before, tegrastats at 200 ms). No memory guard was active. No `configs/*.json` exists; the script plus `extra-args.txt` define the configuration.
- Device: Jetson Orin Nano 8 GB, JetPack R36.4.4, 25W ([status.json](../results/runs/jetson-edgedit-flux-klein-002__attempt__20261011-autofit/status.json)). [Metric definitions](../wiki/methods/baseline-metrics.md#jetson-cli-feasibility-diagnostics).

## Results

Error evidence:

- No `exit_code.txt` or `end.txt` was written: the harness did not regain control.
- [ed-cli.log](../results/runs/jetson-edgedit-flux-klein-002__attempt__20261011-autofit/ed-cli.log) holds one marker, `stage=encode event=begin t=1791700669.409536` (06:37:49.41 UTC); no encode end.
- [previous-boot-kernel.txt](../results/runs/jetson-edgedit-flux-klein-002__attempt__20261011-autofit/previous-boot-kernel.txt) lists a single boot (index 0) ending 02:41:16 EDT; no previous-boot kernel log was available (status.json: journal not persistent).
- Memory before launch ([free-before.txt](../results/runs/jetson-edgedit-flux-klein-002__attempt__20261011-autofit/free-before.txt), MiB): available 6826, swap used 312.

Placement and precision decisions from [ed-cli.stderr](../results/runs/jetson-edgedit-flux-klein-002__attempt__20261011-autofit/ed-cli.stderr):

| Item | Value (engine log) |
|---|---|
| Available UMA memory | 6894596 kB, later 6883164 kB |
| auto-fit, text encoder | `TE quant set to q8_0 (superseding global --type for TE), 2.32 GB -> 2.32 GB` |
| auto-fit, transformer | `DiT quant set to q8_0 ... 2.17 GB`; `2.17 GB fits 2.58 GB budget -> resident` |
| Measured DiT compute buffer | 1.04 GB at latent 64x64 (fixed fallback 4.00 GB) |
| Transformer | RESIDENT; param buffer 2219.84 MB (VRAM) |
| Text encoder | OFFLOAD+segment; param buffer 3555.38 MB (RAM) |
| VAE | OFFLOAD+segment; param buffer 94.72 MB (RAM) |
| Segment budget | 2.40 GB (effective 6.56 GB − resident 2.17 GB − 2.00 GB staging slack) |
| Weight type stat, conditioner | q8_0:145, q4_K:216, q6_K:37 |
| Weight type stat, diffusion model | q4_0:80, q8_0:3, bf16:66 |
| Activation dtype | f32 (bf16_tensors=66/149) |
| qwen3 graph cut | max_vram=2454.01 MB, merged 29 segments → 12 segments (41 ms) |

The engine reported `EdgeDitEngine initialized successfully` before the encode marker.

Memory ([tegrastats.log](../results/runs/jetson-edgedit-flux-klein-002__attempt__20261011-autofit/tegrastats.log), 355 samples, 02:36:31–02:37:50 local): RAM peak 7478/7620 MB (02:36:55, swap 1500 MB in that sample); swap peak 1529/3810 MB (02:36:55). Last sample: RAM 7430/7620 MB, swap 673 MB, one CPU core at 100%.

## Observations

- Loading completed and the engine initialized; the failure occurred after the encode phase began.
- `--auto-fit` changed stored precision: the 145 f32 conditioner tensors and 3 of the 69 bf16 transformer tensors were reported as q8_0; the 80 q4_0 transformer tensors and the q4_K/q6_K encoder tensors were kept.
- With the text encoder offloaded, its parameter buffer was still 3555.38 MB, now in RAM.
- The resident transformer buffer was 2219.84 MB, down from 2346.40 MB in [001](jetson-edgedit-flux-klein-001.md).
- RAM stayed within about 140–190 MB of the 7620 MB total in the peak and last samples, with swap occupied.

## Interpretation

- Offloading on this device moves weights between two views of the same physical memory; the planned footprint (resident transformer plus RAM-held encoder and VAE plus segment budget) did not leave room for the system.
- Hypothesis: the board became unresponsive from memory pressure and swapping during segmented text encoding, not from a CUDA error; no kernel log survives to confirm it.
- Status.json flags the log line `DiT quant set to q8_0` with a Q4_0 source for checking against the documented rule that auto-fit does not raise an already-lower precision. The weight-type stat shows the q4_0 tensors kept and only 3 bf16 tensors converted; whether the reported 2.17 GB plan matches that is not resolved from these files.

## Next experiment

- `--auto-allocate` keeping stored precision ([003](jetson-edgedit-flux-klein-003.md)), run under a memory guard.
