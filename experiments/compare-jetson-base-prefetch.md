---
type: experiment-record
id: compare-jetson-base-prefetch
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-base-001__repeat__20261005-042014, jetson-flux-klein-base-004__baseline__20261005-074852]
updated: 2026-10-05
---

# Compare: Jetson Base parameter prefetch

## Question

Does disabling parameter prefetch improve repeated Base latency? Week2 image baseline evidence, RQ1.

## Runs and matching

| Role | Configuration | Record |
|---|---|---|
| Prefetch enabled | [reference](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/config.json) | [Base001](jetson-flux-klein-base-001.md) |
| Prefetch disabled | [variant](../results/runs/jetson-flux-klein-base-004__baseline__20261005-074852/config.json) | [Base004](jetson-flux-klein-base-004.md) |

| Held fixed | Value |
|---|---|
| Device | Jetson Orin Nano,25W,four CPU threads |
| Model | Same pinned Base Q4_0, Qwen3 Q4_K_M, BF16 VAE hashes |
| Workload | 512²,50Euler steps,flux2 scheduler,CFG4,batch1,seed0,same cat/sign prompt |
| Execution | Same harness binary and engine commit, eager disk-backed segmented CUDA, no reuse, flash attention |
| Protocol | One first + three warm-ups + ten measured, unprofiled, one-second tegrastats |
| Variable | Parameter prefetch enabled versus disabled |

Runs were sequential; filesystem cache, background activity and thermal history were not experimentally matched. Treat the observed difference as descriptive, not an isolated causal estimate.

## Results

Sources: [reference summary](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/summary.json), [variant summary](../results/runs/jetson-flux-klein-base-004__baseline__20261005-074852/summary.json).

| Metric | Prefetch enabled median (range), ms | Disabled median (range), ms |
|---|---:|---:|
| wall_ms | 328719.445 (313962.972–331504.639) | 320877.059 (315421.228–328175.862) |
| text_encode_ms | 24192.235 (16156.323–24600.976) | 21929.209 (16734.053–24684.082) |
| denoise_ms | 299599.784 (289221.908–301971.807) | 295679.244 (289598.117–299968.072) |
| vae_decode_ms | 4598.830 (3588.229–4890.237) | 4367.119 (4064.204–4589.003) |
| other_ms | 295.655 (280.565–306.708) | 283.556 (259.066–295.068) |

Observed median delta -7842.386ms (-2.386%); reference/variant ratio 1.024440. Whole-capture system RAM peak 6.014648 vs 6.045898GiB; swap 0.302734–0.316406 vs 0.313477–0.375000GiB. Memory is not isolated model allocation.

## Findings and selection

All fourteen variant images match the reference pixels; [diagnostics](../results/runs/jetson-flux-klein-base-004__baseline__20261005-074852/quality-diagnostics.json) report MSE0 and LPIPS0. The exact-option fidelity screen passes. The predeclared speed rule fails: variant maximum 328175.862ms is not below reference minimum 313962.972ms. Do not select prefetch-disabled combinations under that rule.

## Limits

One development prompt/seed; no held-out or formal quality evaluation. The requested context flag is audited, but physical storage traffic and overlap are not measured here. The lower median alone is insufficient to claim a reliable speedup.
