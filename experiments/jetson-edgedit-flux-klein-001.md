---
type: experiment-record
id: jetson-edgedit-flux-klein-001
status: failed
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-001__attempt__20261011-062302, jetson-edgedit-build-20261011]
updated: 2026-10-14
---

# jetson-edgedit-flux-klein-001: edge-dit.cpp, all components on GPU, quantized 512×512

Status: **failed** (2026-10-11, 06:23:02–06:24:20 UTC); kernel OOM kill during weight loading, exit code 137. No image, no timing.

## Question

Can edge-dit.cpp load and run the quantized distilled klein workload with all three components (transformer, text encoder, VAE) placed on the GPU by explicit options, with no automatic placement or offload? This is the reference placement for the edge-dit rungs on this device (gate E1). Week 2 engine baselines (rolled over), RQ1.

## Setup

- **Engine build** ([build evidence](../results/runs/jetson-edgedit-build-20261011/)): edge-dit.cpp `97cc585d9dbd9ac5e30b78ef2e2c4a0d8bfbe9f2`, ggml `f270346d688c7bec7c56bae2ae73343aba6817ad`, built on the device with [jetson_build_edgedit.sh](../scripts/jetson_build_edgedit.sh): Release, `performance` profile, CUDA architecture 87, CUDA compiler NVIDIA 12.6.68, GNU 11.4.0, cuDNN 9.3.0 with cuDNN SDPA, CUDA norm/RoPE/modulation and CUDA graphs ON, sage attention, MPI, NCCL and parallel OFF, benchmark markers ON ([build-config.txt](../results/runs/jetson-edgedit-build-20261011/build-config.txt), [binaries-and-flags.txt](../results/runs/jetson-edgedit-build-20261011/binaries-and-flags.txt)). The build job count was capped at 2 by a cmake wrapper. `ed-cli` SHA256 `271d8fa2…f995528`; the build log ends with `BUILD_OK` ([log](../results/runs/jetson-edgedit-build-20261011/edgedit-build-20261011.log)).
- **Weights** ([model-sha256.txt](../results/runs/jetson-edgedit-flux-klein-001__attempt__20261011-062302/model-sha256.txt)): the Jetson GGUF files used with sd.cpp on this device: transformer `flux-2-klein-4b-Q4_0.gguf`, text encoder `Qwen3-4B-Q4_K_M.gguf`, VAE `flux2-vae.safetensors` (BF16). The engine log reports component tensor bytes 2346.39 MB (transformer: q4_0=80, bf16=69), 2375.91 MB (text encoder: f32=145, q4_K=216, q6_K=37) and 160.31 MB (VAE: bf16=250, i32=1).
- **Workload**: 512×512, 4 steps, guidance 1.0, CFG 1.0, seed 0, prompt "A cat holding a sign that says hello world", batch 1.
- **Options** ([run.sh](../results/runs/jetson-edgedit-flux-klein-001__attempt__20261011-062302/run.sh)): `--backend cuda --type preserve --vae-tiling off --flash-attention --cache off --threads 4`; no `--auto-fit`, `--auto-allocate` or offload flags. No `configs/*.json` exists for this run; the command line in `run.sh` is the configuration.
- **Change vs reference**: none (reference placement for this engine on this device).
- **Protocol**: one generation attempted; no warm-ups, no repeats. Whole-system memory from `tegrastats --interval 200` (printed MB). No memory guard was active for this run.
- **Device**: Jetson Orin Nano 8 GB, JetPack R36.4.4, 25W mode ([status.json](../results/runs/jetson-edgedit-flux-klein-001__attempt__20261011-062302/status.json)). [Metric definitions](../wiki/methods/baseline-metrics.md#jetson-cli-feasibility-diagnostics).

## Results

Error evidence:

- [exit_code.txt](../results/runs/jetson-edgedit-flux-klein-001__attempt__20261011-062302/exit_code.txt): `137`.
- [kernel-oom.txt](../results/runs/jetson-edgedit-flux-klein-001__attempt__20261011-062302/kernel-oom.txt): `packagekitd invoked oom-killer` (global OOM, `constraint=CONSTRAINT_NONE`); `Killed process 8501 (ed-cli) total-vm:11150884kB, anon-rss:0kB, file-rss:3872kB`. Kernel timestamps are local time (02:24:18–02:24:19 EDT).
- [ed-cli.log](../results/runs/jetson-edgedit-flux-klein-001__attempt__20261011-062302/ed-cli.log) is empty: no phase marker was written, so no generation phase began.

Partial observations from [ed-cli.stderr](../results/runs/jetson-edgedit-flux-klein-001__attempt__20261011-062302/ed-cli.stderr):

| Item | Value |
|---|---|
| CUDA device | Orin, compute capability 8.7, VMM yes, VRAM 7619 MiB |
| Available UMA memory at start (engine) | 6731224 kB |
| Transformer spec check | 149 tensors declared; missing=0, shape_mismatch=0, unexpected=0 |
| Activation dtype (engine) | f32 (bf16_tensors=69/149) |
| Param buffer: flux (transformer) | 2346.40 MB (VRAM) |
| Param buffer: qwen3 (text encoder) | 3555.38 MB (VRAM) |
| Param buffer: VAE | 94.72 MB (VRAM) |
| Last tensor-loading progress line | 240/795 tensors, 2786.81 MB, 62.12 s |

Memory ([tegrastats.log](../results/runs/jetson-edgedit-flux-klein-001__attempt__20261011-062302/tegrastats.log), 297 samples): RAM peak 7424/7620 MB; swap peak 2346/3810 MB, both in the same sample (02:24:09 local). RAM at the first sample was 821 MB. No sample was recorded between 02:24:09 and 02:24:19, the time of the kill.

Status.json also records that a first launch failed before `ed-cli` ran because `/usr/bin/time` was not installed; the run was relaunched without it.

## Observations

- Both GGUF files and the VAE were accepted; the transformer tensor spec matched with no missing or mismatched tensors.
- The qwen3 parameter buffer (3555.38 MB) is larger than the text-encoder tensor bytes in the file (2375.91 MB) by 1179.47 MB.
- Requested buffers for the three components total 5996.50 MB against 6731224 kB reported available at start.
- Loading had reached 2786.81 MB of tensor data (240 of 795 tensors) after 62.12 s when progress output stopped.
- RAM approached the 7620 MB total with 2346 MB of swap in use; the kernel selected `ed-cli` for the OOM kill, reporting anon-rss 0 kB.

## Interpretation

- At this commit, explicit all-GPU placement of these files does not fit the device's shared memory: the requested parameter buffers alone leave little room for the CUDA context, compute buffers and the rest of the system.
- Hypothesis: the 1179.47 MB excess of the qwen3 buffer over the file comes from the engine expanding the Q6_K token-embedding table (`token_embd.weight`, 151936×2560) to F32 at load. 151936×2560×4 bytes is 1483.75 MiB and the Q6_K tensor is 319065600 bytes (304.28 MiB); the difference, 1179.47 MiB, equals the excess. The mechanism (the engine's embedding `get_rows` support list) is recorded in [status.json of 004](../results/runs/jetson-edgedit-flux-klein-004__attempt__20261011-embq8/status.json) from source inspection and is tested by [004](jetson-edgedit-flux-klein-004.md).
- Hypothesis: `anon-rss:0kB` at kill indicates the process memory was held as GPU/driver allocations not counted as anonymous RSS on this platform; not verified.

## Next experiment

- `--auto-fit` on the same workload ([002](jetson-edgedit-flux-klein-002.md)).
- `--auto-allocate` with stored precision kept ([003](jetson-edgedit-flux-klein-003.md)).
- Text encoder with a Q8_0 token embedding, all-GPU ([004](jetson-edgedit-flux-klein-004.md)).
