---
type: experiment-record
id: jetson-edgedit-flux-klein-004
status: failed
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-004__attempt__20261011-embq8, jetson-edgedit-convert-20261011]
updated: 2026-10-14
---

# jetson-edgedit-flux-klein-004: edge-dit.cpp, all-GPU, text encoder with Q8_0 token embedding

Status: **failed** (2026-10-11, 15:06:24–15:07:46 UTC); text encoding and all four denoising steps completed, then the memory guard killed the run at the start of VAE decode, exit code 137. No image.

## Question

With a text-encoder checkpoint whose token-embedding table the engine can keep compressed, does the all-GPU placement of [001](jetson-edgedit-flux-klein-001.md) admit the workload on this device? Single change against 001: the text-encoder checkpoint. Week 2 engine baselines (rolled over), RQ1.

## Setup

- **Workload label**: compressed text-encoder checkpoint, **Qwen3-4B Q4_K_M with a Q8_0 token embedding** (`Qwen3-4B-Q4_K_M-embq8_0.gguf`, SHA256 `d83c059b…f867686`). This is a separately labelled compressed checkpoint, not the reference Q4_K_M file.
- **Conversion** ([convert evidence](../results/runs/jetson-edgedit-convert-20261011/)): [requant_gguf_tensor.py](../scripts/requant_gguf_tensor.py) rewrote the reference `Qwen3-4B-Q4_K_M.gguf` (SHA256 `f6f85177…`) with only `token_embd.weight` (shape 151936×2560) re-quantized from Q6_K to Q8_0; the other 397 tensors and the metadata were copied. Per [requant.log](../results/runs/jetson-edgedit-convert-20261011/requant.log): 319065600 → 413265920 bytes, max abs diff vs source 0.00095367431640625, relative RMS diff vs source 0.005288755985412679. The conversion ran under the memory guard without a kill ([memguard.json](../results/runs/jetson-edgedit-convert-20261011/memguard.json): exit 0, MemAvailable minimum 4451 MB). [convert.log](../results/runs/jetson-edgedit-convert-20261011/convert.log) holds one line, `unsupported weight type: preserve`, from a failed conversion attempt; the command for that attempt is not recorded, and the converted file comes from `requant_gguf_tensor.py`.
- Same build, transformer, VAE, workload and fixed options as [001](jetson-edgedit-flux-klein-001.md) ([model-sha256.txt](../results/runs/jetson-edgedit-flux-klein-004__attempt__20261011-embq8/model-sha256.txt)): edge-dit.cpp `97cc585`, Q4_0 transformer, BF16 VAE; 512×512, 4 steps, guidance 1.0, CFG 1.0, seed 0; `--backend cuda --type preserve --vae-tiling off --flash-attention --cache off --threads 4`; no extra arguments ([extra-args.txt](../results/runs/jetson-edgedit-flux-klein-004__attempt__20261011-embq8/extra-args.txt) holds `''`).
- Launched with [jetson_edgedit_smoke.sh](../scripts/jetson_edgedit_smoke.sh) under [jetson_memguard.sh](../scripts/jetson_memguard.sh) (MemAvailable < 400 MB or swap growth > 768 MB, 0.1 s poll). Tegrastats at 200 ms. One generation attempted. No `configs/*.json` exists.
- Device: Jetson Orin Nano 8 GB, JetPack R36.4.4, 25W ([status.json](../results/runs/jetson-edgedit-flux-klein-004__attempt__20261011-embq8/status.json)). [Metric definitions](../wiki/methods/baseline-metrics.md#edge-dit-ed-sample-adapter) (phase-marker semantics).

## Results

Error evidence:

- [memguard.json](../results/runs/jetson-edgedit-flux-klein-004__attempt__20261011-embq8/memguard.json): `killed_by_guard: true`, reason `MemAvailable 345 MB < 400 MB`; MemAvailable 6672 MB at start, minimum 345 MB; swap used 33 MB at start, maximum 63 MB.
- [exit_code.txt](../results/runs/jetson-edgedit-flux-klein-004__attempt__20261011-embq8/exit_code.txt): `137`. [kernel-oom.txt](../results/runs/jetson-edgedit-flux-klein-004__attempt__20261011-embq8/kernel-oom.txt) is empty.
- [ed-cli.log](../results/runs/jetson-edgedit-flux-klein-004__attempt__20261011-embq8/ed-cli.log): `decode event=begin` at t=1791731265.327786 (15:07:45.33 UTC) with no decode end.

Engine log ([ed-cli.stderr](../results/runs/jetson-edgedit-flux-klein-004__attempt__20261011-embq8/ed-cli.stderr)):

| Item | Value |
|---|---|
| Text-encoder tensors in file | 2465.75 MB (f32=145, q8_0=1, q4_K=216, q6_K=36) |
| Param buffer: flux (transformer) | 2346.40 MB (VRAM) |
| Param buffer: qwen3 (text encoder) | 2465.75 MB (VRAM) |
| Param buffer: VAE | 94.72 MB (VRAM) |
| Available UMA memory at start | 6730268 kB |
| Activation dtype | f32 (bf16_tensors=69/149) |
| Schedule | latent 32x32, image_seq_len 1024, flux2_mu 2.031; sigmas 1.000000 → 0.958085 → 0.883982 → 0.717497 → 0.000000 |
| Engine sampling line | `flux sampling completed, taking 10.57s` |

Host-clock phase markers ([ed-cli.log](../results/runs/jetson-edgedit-flux-klein-004__attempt__20261011-embq8/ed-cli.log)), single first generation, not a timing result:

| Phase | Duration (s) |
|---|---:|
| encode_setup | 2.840 |
| denoise (4 steps) | 10.574 |
| vae_decode | not completed |

Memory ([tegrastats.log](../results/runs/jetson-edgedit-flux-klein-004__attempt__20261011-embq8/tegrastats.log), 389 samples; [peak-ram.txt](../results/runs/jetson-edgedit-flux-klein-004__attempt__20261011-embq8/peak-ram.txt)): RAM peak 6837/7620 MB, swap peak 63/3810 MB. From 11:07:36 to 11:07:45 local (denoising) RAM held at 6708–6721 MB; the 6837 MB peak is in a sample at 11:07:45 local, the second the decode marker begins, followed by the drop at the kill.

## Observations

- The qwen3 parameter buffer equals the file's text-encoder tensor bytes (2465.75 MB), where in [001](jetson-edgedit-flux-klein-001.md) it exceeded them by 1179.47 MB.
- All three components loaded on the GPU and the engine completed text encoding and the four-step schedule.
- The guard fired at the start of VAE decode with VAE tiling off; swap barely changed.

## Interpretation

- Keeping the token embedding in a type the engine stores compressed removes the load-time expansion seen in 001, which is enough for loading, encoding and denoising to fit under the guard.
- Hypothesis: untiled VAE decode at 512×512 needs a compute buffer larger than the headroom left after resident weights and denoising (7620 − 6721 MB in tegrastats terms, before the guard's 400 MB margin); the 200 ms tegrastats sampling may have missed the actual peak.
- The memory-guard threshold (400 MB MemAvailable) is a protection margin, not the device limit; the run is failed under the guard as configured.

## Next experiment

- Same converted encoder with `--vae-tiling on` ([005](jetson-edgedit-flux-klein-005.md)).
