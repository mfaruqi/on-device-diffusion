---
type: experiment-record
id: jetson-nunchaku-flux-klein-001
status: complete
device: jetson-orin-nano
engine: nunchaku
runs: [jetson-nunchaku-flux-klein-001__baseline__20261102-120036, jetson-nunchaku-flux-klein-001__attempt__20261102-115600, jetson-nunchaku-layer-check-001__layer__20261102-165502]
updated: 2026-11-02
---

# jetson-nunchaku-flux-klein-001: SVDQuant INT4 W4A4/r32 transformer (Nunchaku, sm_87 source build) + NF4 text encoder, sequential stage residency, 512×512

Status: **complete** (2026-11-02, run `__baseline__20261102-120036`; run directly on the device, no Slurm job).
The smoke gate and the full 1+3+10 protocol completed. A single-layer kernel check on the same board, run
before the pipeline, gated both.

## Question

Do Nunchaku's SVDQuant W4A4 kernels compute correctly on the Jetson Orin Nano (sm_87) when built from the pinned
source, and what are the stage latencies and memory of distilled klein at 512×512 when the transformer is the
community INT4/r32 checkpoint? The reference is
[jetson-torch-flux-klein-001](jetson-torch-flux-klein-001.md) (GGUF Q4_0 transformer). This is a separately labelled
replacement-checkpoint workload. RQ1 (engine and precision coverage on the edge device); week 5 quantized variant.

## Setup

- [Configuration](../configs/jetson-nunchaku-flux-klein-int4-r32-512.json) (full protocol) and
  [smoke configuration](../configs/jetson-nunchaku-flux-klein-int4-r32-512-smoke.json) (1 first + 1 measured).
  `black-forest-labs/FLUX.2-klein-4B` @ `e7b7dc27f91deacad38e78976d1f2b499d76a294` for the scheduler,
  tokenizer, text encoder source and VAE.
- Workload: distilled klein, batch 1, 512×512, 4 steps, guidance 1.0, `max_sequence_length` 512, prompt
  "A cat holding a sign that says hello world", seed 0. Same as the reference.
- One change from the reference: `optimizations.quantization.transformer` is `nunchaku_svdq` instead of `gguf`.
  - Checkpoint: community `tonera/FLUX.2-klein-4B-Nunchaku` @ `8aa64445d8e45edc2ea431743b67ac7e2ca5f739`,
    `svdq-int4_r32-FLUX.2-klein-4B-Nunchaku.safetensors` (sha256 `8b0adf66…49311`, verified at load). Its
    ancestry to the pinned BF16 transformer revision is not established.
  - Kernels: `tonera/nunchaku` @ `0c8379d10b4b9017928d02fd0fdb2c26b991a8b6`, built in place for sm_87
    ([build probe](../wiki/methods/nunchaku-howto.md#sm87-build-probe-2026-11-02)). Two lines differ from the
    pinned source, both architecture allowlists: `setup.py get_sm_targets()` and
    `nunchaku/utils.py check_hardware_compatibility()`
    ([nunchaku-source-patch.diff](../results/runs/jetson-nunchaku-flux-klein-001__baseline__20261102-120036/nunchaku-source-patch.diff)).
    The runner refuses to load unless the tree is at that revision and its diff hashes to the config's
    `patch_sha256`. 120 `SVDQW4A4Linear` layers, all INT4 with rank 32
    ([load.json](../results/runs/jetson-nunchaku-flux-klein-001__baseline__20261102-120036/load.json)).
  - Unchanged from the reference: the pre-quantized bitsandbytes NF4 text encoder (sha256 `eaa2048d…`), the BF16 VAE,
    and the stage residency `text_encoder: release_after_encode`, `transformer: release_during_encode`. Each
    generation frees the transformer, loads the encoder, encodes, frees the encoder and reloads the transformer.
    Reloads fall between timed stages (inside `wall_ms` and `other_ms`).
  - No CPU offload, VAE tiling, compiler or attention-backend restriction.
- Protocol: 1 first + 3 warm-up (discarded) + 10 measured. Tegrastats every 200 ms.
- Runner: [scripts/run_flux_portable.py](../scripts/run_flux_portable.py) sha256 `e7337b57…`
  ([source-provenance.json](../results/runs/jetson-nunchaku-flux-klein-001__baseline__20261102-120036/source-provenance.json)).
  Launched by [queue N](../scripts/jetson_nunchaku_queue_20261102.sh) under
  [jetson_memguard.sh](../scripts/jetson_memguard.sh) with `MAX_SWAP_GROWTH_MB=1536` (launcher safety setting, not
  part of the configuration).
- Device: Jetson Orin Nano 8 GB (`mfaruqi-desktop`), R36.4.4, 25W, CUDA 12.6, cuDNN 9.3.0, sm_87. Software: torch
  2.8.0, diffusers 0.40.0, transformers 5.16.1, bitsandbytes 0.48.0.dev0+ff389db, Python 3.10.12
  ([environment.json](../results/runs/jetson-nunchaku-flux-klein-001__baseline__20261102-120036/environment.json)).
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md).

## Results

### Kernel check (gate)

[layer-check.json](../results/runs/jetson-nunchaku-layer-check-001__layer__20261102-165502/layer-check.json), script
[nunchaku_layer_check.py](../scripts/nunchaku_layer_check.py), method in the
[how-to](../wiki/methods/nunchaku-howto.md#single-layer-check-2026-11-02). Four layers of transformer block 0 run
through the build's own `SVDQW4A4Linear` on fixed synthetic N(0, 1) inputs and are compared with the same inputs on an
A100 (community wheel). The gate was every output finite and rel_l2 ≤ 1e-2 against the A100 outputs; it passed.

| Layer (in→out) | rel_l2 vs A100, 1024 tokens | rel_l2 vs BF16 | W4A4 / BF16 F.linear, 4096 tokens (ms, median of 20) |
|---|---:|---:|---:|
| attn.to_qkv (3072→9216) | 3.5e-7 | 0.1217 | 12.24 / 26.70 |
| attn.to_out.0 (3072→3072) | 4.1e-5 | 0.1398 | 4.82 / 8.40 |
| ff.linear_in (3072→18432) | 6.8e-5 | 0.1416 | 23.59 / 53.80 |
| ff.linear_out (9216→3072) | 3.6e-5 | 0.1524 | 13.07 / 36.99 |

The rel_l2 vs BF16 values match the A100's to four decimals. The inputs are synthetic, so they measure the kernels'
4-bit error on Gaussian data, not the checkpoint's quality.

### Latency: full protocol

[Run](../results/runs/jetson-nunchaku-flux-klein-001__baseline__20261102-120036/),
[summary.json](../results/runs/jetson-nunchaku-flux-klein-001__baseline__20261102-120036/summary.json),
[runs.csv](../results/runs/jetson-nunchaku-flux-klein-001__baseline__20261102-120036/runs.csv),
[stage-residency.json](../results/runs/jetson-nunchaku-flux-klein-001__baseline__20261102-120036/stage-residency.json).
Started 17:00:36Z, finished 17:17:36Z; `status.json` complete; guard not triggered.

| Metric | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall_ms | 64598.3 | 58458.7–72655.9 | – |
| text_encode_ms | 1104.1 | 1063.7–2205.5 | 1.7% |
| denoise_ms | 2953.5 | 2940.7–3249.9 | 4.6% |
| denoise_step_0_ms | 761.0 | 755.0–1055.9 | – |
| denoise_step_1_ms | 728.9 | 726.4–730.0 | – |
| denoise_step_2_ms | 729.0 | 726.5–730.0 | – |
| denoise_step_3_ms | 733.1 | 730.3–734.0 | – |
| vae_decode_ms | 486.6 | 485.3–487.2 | 0.8% |
| postprocess_ms | 48.6 | 32.1–96.9 | 0.1% |
| other_ms | 59910.9 | 53910.5–66815.0 | 92.7% |
| First run wall_ms | 77784.4 | – | – |

`other_ms` is almost entirely the stage-residency reloads. Over all 14 generations, the median text-encoder load is
33.11 s (27.95–36.10) and the median transformer load is 27.88 s (23.94–33.97). Frees take about 0.5 s each.
The first run's vae_decode_ms is 3597.3; every measured value is below 490.

Cold load (`load.json`): 62.6 s from_pretrained + placement. This is a cached read, not a cold read.

### Memory

| Metric | Value |
|---|---:|
| Transformer weights (INT8-packed INT4 + BF16) | 2.280 GiB |
| VAE weights | 0.157 GiB |
| Allocated after load | 2.504 GiB |
| Peak allocated (per generation, median) | 3.118 GiB |
| Peak reserved (per generation, median) | 3.650 GiB |
| Whole-capture sampled system RAM peak (tegrastats) | 5.53 GiB of 7.44 |
| Swap, whole capture | 0.28–0.96 GiB |
| Host peak RSS | 3.65 GiB |
| Memory guard: MemAvailable minimum | 1771 MB |

NVML is unavailable on the Jetson, so `device_used_peak_gib` is empty; system RAM comes from tegrastats over the
whole capture, including load.

### Repeatability and output

- Denoise steps 1–3 vary by less than 4 ms across the 10 measured generations. The wall-time spread (14.2 s) comes
  from the reload times.
- `deterministic_output` is false: all 14 generations have different pixel hashes despite the fixed seed
  ([runs.csv](../results/runs/jetson-nunchaku-flux-klein-001__baseline__20261102-120036/runs.csv)). Only `first.png`
  and `measured-0.png` are kept. Their mean absolute pixel difference is 15.57 on a 0–255 scale (max 223), a
  diagnostic, not a quality score.
- Both images show a coherent cat holding a sign. The lettering differs between them and misspells the prompt's
  text ("HOLLO WORLD", "HLLLO WORLD"). Quality is not measured.

### Smoke gate

[Run](../results/runs/jetson-nunchaku-flux-klein-001__attempt__20261102-115600/summary.json): complete. First wall
82212.0 ms, one measured wall 73388.6 ms, denoise 3238.7 ms.

### W&B

Uploaded and verified 2026-11-02 (state finished, medians match `summary.json`, two images, run-directory artifact): [full](https://forge.coreweave.com/wandb/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-nunchaku-flux-klein-001__baseline__20261102-120036), [smoke](https://forge.coreweave.com/wandb/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-nunchaku-flux-klein-001__attempt__20261102-115600). The layer checks are not uploaded; they have no generation protocol.

## Observations

- The pinned Nunchaku source, built for sm_87 with only the two allowlists changed, gives the same layer outputs on
  the Orin as the A100 community wheel to within rel_l2 7e-5.
- On this board, the four W4A4 layers take 0.35–0.57 of the BF16 `F.linear` time at 4096 tokens, and 0.37–0.63 at
  1024 tokens.
- A steady-state denoise step takes 729 ms. Step 0 adds about 30 ms in the median.
- Reloads are 92.7% of the median wall time. The transformer reload of the 2.28 GiB checkpoint takes 27.9 s
  (median).
- Generations are not bitwise reproducible.

## Interpretation

- The kernels compute the same function on sm_87 as on sm_80, so the Orin result is a valid execution of this
  checkpoint, not a numerically broken port. Correctness of the full model is supported only indirectly, by the
  coherent images; there is no quality measurement.
- Under sequential residency, end-to-end latency is bounded by component reloads, not by denoising. Under this
  policy, the faster transformer moves wall time much less than denoise time.
- Hypothesis: the run-to-run variation comes from non-deterministic reductions or atomics in the fused kernels, as on
  the A100 run of the same checkpoint. It is not tested here.

## Next experiment

- `jetson-nunchaku-flux-klein-int4-r32-512` + conditioning reuse (exact prompt-encoding reuse removes the per-generation encoder load and transformer reload).
- Quality: paired LPIPS/CLIP against the Jetson GGUF images on the calibration prompt set.
