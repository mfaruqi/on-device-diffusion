---
type: experiment-record
id: jetson-edgedit-flux-klein-003
status: failed
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-003__attempt__20261011-autoalloc]
updated: 2026-10-14
---

# jetson-edgedit-flux-klein-003: edge-dit.cpp with `--auto-allocate`, quantized 512×512

Status: **failed** (2026-10-11, 14:44:25–14:45:02 UTC); killed by the memory guard during weight loading, exit code 137. The board stayed up. No image, no timing.

## Question

Does edge-dit.cpp's automatic per-component placement (`--auto-allocate`), which keeps the stored precision, admit the quantized distilled klein workload on this device? Single change against the all-GPU reference [001](jetson-edgedit-flux-klein-001.md) (gate E2, rung 3). Week 2 engine baselines (rolled over), RQ1.

## Setup

- Same build, weights, workload and fixed options as [001](jetson-edgedit-flux-klein-001.md): edge-dit.cpp `97cc585`, Q4_0 transformer, Qwen3-4B Q4_K_M text encoder, BF16 VAE ([model-sha256.txt](../results/runs/jetson-edgedit-flux-klein-003__attempt__20261011-autoalloc/model-sha256.txt)); 512×512, 4 steps, guidance 1.0, CFG 1.0, seed 0; `--backend cuda --type preserve --vae-tiling off --flash-attention --cache off --threads 4`.
- **Change vs reference**: `--auto-allocate` ([extra-args.txt](../results/runs/jetson-edgedit-flux-klein-003__attempt__20261011-autoalloc/extra-args.txt)).
- Launched with [jetson_edgedit_smoke.sh](../scripts/jetson_edgedit_smoke.sh) under [jetson_memguard.sh](../scripts/jetson_memguard.sh): SIGKILL to the process group if MemAvailable falls below 400 MB or swap grows by more than 768 MB, polled every 0.1 s. Tegrastats at 200 ms. No `configs/*.json` exists; the script plus `extra-args.txt` define the configuration.
- Device: Jetson Orin Nano 8 GB, JetPack R36.4.4, 25W ([status.json](../results/runs/jetson-edgedit-flux-klein-003__attempt__20261011-autoalloc/status.json)). [Metric definitions](../wiki/methods/baseline-metrics.md#jetson-cli-feasibility-diagnostics).

## Results

Error evidence:

- [memguard.json](../results/runs/jetson-edgedit-flux-klein-003__attempt__20261011-autoalloc/memguard.json): `killed_by_guard: true`, reason `MemAvailable 399 MB < 400 MB`; MemAvailable 6419 MB at start, minimum 399 MB; swap used 0 MB at start, maximum 39 MB.
- [exit_code.txt](../results/runs/jetson-edgedit-flux-klein-003__attempt__20261011-autoalloc/exit_code.txt): `137`. [kernel-oom.txt](../results/runs/jetson-edgedit-flux-klein-003__attempt__20261011-autoalloc/kernel-oom.txt) is empty (no kernel OOM kill).
- [ed-cli.log](../results/runs/jetson-edgedit-flux-klein-003__attempt__20261011-autoalloc/ed-cli.log) is empty: no generation phase began. The last loading line in [ed-cli.stderr](../results/runs/jetson-edgedit-flux-klein-003__attempt__20261011-autoalloc/ed-cli.stderr) is 152/795 tensors, 2346.39 MB, 31.00 s, followed by the guard's `Killed`.
- Memory before launch ([free-before.txt](../results/runs/jetson-edgedit-flux-klein-003__attempt__20261011-autoalloc/free-before.txt), MiB): available 6421, free 92, buff/cache 6582, swap used 0.

Placement decisions from the engine log:

| Component | Decision (engine log) | Param buffer |
|---|---|---|
| model.diffusion_model | `2.29 GB > 6.19 GB budget left -> OFFLOAD+segment` | 2346.40 MB (RAM) |
| text_encoders | `2.32 GB > 6.19 GB budget left -> OFFLOAD+segment` | 3555.38 MB (RAM) |
| first_stage_model (VAE) | `0.16 GB fits but would squeeze offloaded segment budget (2.04 GB left < 8.00 GB safe) -> OFFLOAD instead` | 94.72 MB (RAM) |

Segment budget: 4.19 GB (effective 6.19 GB − resident 0.00 GB − 2.00 GB staging slack). Available UMA memory at start: 6493528 kB. Activation dtype f32 (bf16_tensors=69/149); transformer spec missing=0, shape_mismatch=0, unexpected=0.

Memory ([tegrastats.log](../results/runs/jetson-edgedit-flux-klein-003__attempt__20261011-autoalloc/tegrastats.log), 176 samples; [peak-ram.txt](../results/runs/jetson-edgedit-flux-klein-003__attempt__20261011-autoalloc/peak-ram.txt)): RAM peak 6975/7620 MB, swap peak 39/3810 MB. RAM rose from 1118 MB to about 6960 MB within the first 5 s (by 10:44:30 local) and stayed between 6957 and 6975 MB until the kill.

## Observations

- All three components were placed off the GPU, so no component was resident; stored precision was kept (no auto-fit requantization lines).
- The log's first two decisions print a comparison that is false as written (2.29 GB and 2.32 GB are not greater than 6.19 GB) yet select OFFLOAD.
- RAM reached its plateau within about 5 s, well before tensor loading finished; at the kill, loading had reached 2346.39 MB, equal to the transformer's tensor bytes.
- The guard fired on MemAvailable, not on swap growth (39 MB maximum).

## Interpretation

- On this device CPU and GPU share one physical memory, so moving parameter buffers to RAM does not reduce the total footprint; the 3555.38 MB text-encoder buffer is present in either placement.
- Hypothesis: the early plateau reflects up-front allocation of all host parameter buffers (2346.40 + 3555.38 + 94.72 MB) before data is read. The guard records only the minimum MemAvailable, not its trajectory, so why it crossed 400 MB about 30 s later while tegrastats RAM stayed flat is not established.
- Hypothesis: the printed `>` comparison reflects a log-message or decision-rule issue in `model_runtime.cpp`; status.json flags it for source inspection, which these files do not include.
- Status.json notes that at this commit, phase-wise release of components exists only for MiniMax-H3 (`--minimax-h3-stage-lifecycle`) and there is no memory-mapped weight option; not re-checked here.

## Next experiment

- Text encoder with a Q8_0 token embedding, all-GPU ([004](jetson-edgedit-flux-klein-004.md)).
