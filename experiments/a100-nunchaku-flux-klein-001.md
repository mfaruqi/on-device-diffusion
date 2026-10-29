---
type: experiment-record
id: a100-nunchaku-flux-klein-001
status: complete
device: a100-pcie-40gb
engine: nunchaku
runs: [a100-nunchaku-flux-klein-001__attempt__20261011-113031, a100-nunchaku-flux-klein-001__attempt__20261011-123802, a100-nunchaku-flux-klein-001__baseline__20261011-124123, a100-nunchaku-flux-klein-001__counters__20261029-001046]
updated: 2026-10-29
---

# a100-nunchaku-flux-klein-001: distilled Klein SVDQuant INT4/r32 admission

## Question

Week 2 engine coverage, RQ1: can the pinned community Nunchaku Klein implementation complete the reference A100 workload with a W4A4 transformer and rank-32 high-precision branch? This replacement checkpoint plus kernel implementation is distinct from an execution-only optimization of the BF16 reference.

## Setup

[Recorded configuration](../results/runs/a100-nunchaku-flux-klein-001__attempt__20261011-113031/config.json), [environment and source snapshot](../results/runs/a100-nunchaku-flux-klein-001__attempt__20261011-113031/environment.json), [procedure](../wiki/methods/nunchaku-howto.md), [metric definitions](../wiki/methods/baseline-metrics.md).

Distilled Klein 4B, 1024×1024, four steps, guidance 1, seed 0, batch one; BF16 text encoder and VAE, no offload or cache. Pinned community INT4/r32 transformer and wheel hashes are in the config. Smoke requests one first and one measured image; full continuation requires a passing smoke and requests one first, three discarded warmups and ten measured images.

## Results

### Preserved initial failure

Job 11929279 on gilbreth-g005 failed during post-load validation ([status](../results/runs/a100-nunchaku-flux-klein-001__attempt__20261011-113031/status.json)). The error is `AssertionError('Unexpected dtype: text_encoder')`. No image generation or valid stage/total latency summary was produced; full continuation did not run. The failed config, environment and traceback are retained.

## Observations and interpretation

The original guard required every floating parameter and buffer to be BF16. Installed Transformers 5.19.0 Qwen3 creates nonpersistent `inv_freq` and `original_inv_freq` rotary buffers in FP32. Hypothesis: these expected buffers caused the guard failure; the original attempt did not capture named tensor dtypes, so the exact offending tensors are not established by its outputs.

The repaired runner saves a named tensor audit before validation, requires BF16 model parameters, and permits FP32 only for those two nonpersistent buffers owned by `Qwen3RotaryEmbedding`. It does not cast tensors, change weights, or modify generation settings. Admission tests confirm that FP32 parameters, other buffer owners/names, persistent FP32 buffers and FP16 substitutions remain rejected.

## Completed retry and full protocol

Job 11929464 completed on gilbreth-g009 (A100 PCIe 40 GB) with exit 0. The [smoke summary](../results/runs/a100-nunchaku-flux-klein-001__attempt__20261011-123802/summary.json) and [validation](../results/runs/a100-nunchaku-flux-klein-001__attempt__20261011-123802/validation.json) admit the unchanged full configuration. The full protocol has one first image, three discarded warm-ups and ten measured generations ([CSV](../results/runs/a100-nunchaku-flux-klein-001__baseline__20261011-124123/runs.csv), [summary](../results/runs/a100-nunchaku-flux-klein-001__baseline__20261011-124123/summary.json), [validation](../results/runs/a100-nunchaku-flux-klein-001__baseline__20261011-124123/validation.json)).

The [tensor audit](../results/runs/a100-nunchaku-flux-klein-001__baseline__20261011-124123/component-tensor-audit.json) confirms that the only FP32 text-encoder entries are `model.rotary_emb.inv_freq` and `model.rotary_emb.original_inv_freq`; text-encoder parameters and floating VAE parameters remain BF16. The [quantization audit](../results/runs/a100-nunchaku-flux-klein-001__baseline__20261011-124123/quantization-audit.json) records 120 INT4/rank-32 modules, pinned checkpoint hash and no offload. Four transformer calls per image and all expected pipeline stages are present.

### Latency

All values below are milliseconds from the full summary; stage shares use the stage median divided by the total median, so shares need not sum exactly to one.

| Scope | Median | Min–max | Share of total |
|---|---:|---:|---:|
| Total | 897.826 | 894.052–901.756 | 100.0% |
| Text encoding | 47.082 | 46.825–47.782 | 5.2% |
| Denoising | 663.223 | 660.012–665.387 | 73.9% |
| VAE decoding | 164.823 | 164.239–166.135 | 18.4% |
| Postprocess | 14.546 | 13.798–18.013 | 1.6% |
| Other GPU-span time | 7.681 | 7.103–8.668 | 0.9% |

Full-protocol first generation: 1520.947 ms. Smoke first generation: 29367.681 ms. Full cached model load: 5.317 s. Full ran after smoke in the same allocation; this is not a cold-cache or cold-storage measurement, and the lower full first-run cost does not establish a cold-start guarantee.

### Memory

[Load accounting](../results/runs/a100-nunchaku-flux-klein-001__baseline__20261011-124123/load.json) and full summary report transformer weights/buffers 2.280 GiB, text encoder 7.492 GiB and VAE 0.157 GiB. Measured per-generation peak allocation is 12.520 GiB, reservation 14.557 GiB and sampled device-wide usage 15.570 GiB. These scopes are separate and must not be added. PyTorch allocator accounting may omit native allocations; NVML samples include context. Peak host RSS across the process is 5.490 GiB.

### Repeatability and quality

Total latency standard deviation is 2.537 ms. Ten measured generations have ten different pixel hashes despite the fixed seed; `deterministic_output` is false. Only the first and first-measured images are retained as PNGs. Their mean absolute pixel difference is 14.481 on a 0–255 scale; this is a diagnostic, not a quality score. The inspected measured image shows a coherent cat holding the requested sign. Saved images are nonflat and 1024×1024, and their pixel hashes match the CSV.

No formal quality evaluation or matched-software BF16 reference was run. Quantized checkpoint ancestry to the exact pinned BF16 transformer revision remains unestablished. No controlled speedup claim is made from this run alone.

## W&B copy

[Full run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-nunchaku-flux-klein-001__baseline__20261011-124123) and [all attempts](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/groups/a100-nunchaku-flux-klein-001) include generation/stage tables, saved images and source evidence artifacts. Remote terminal states, metrics, image counts and artifact contents were [verified](../output/nunchaku-prep/wandb-verification.json).

## Next experiment

Before reporting a controlled speedup, repeat the BF16 reference with the same software environment and A100 class, then evaluate image quality separately.

## Hardware-counter capture (2026-10-28/29)

Measured DRAM traffic of one steady denoising pass of this configuration, for the measured A100 roofline
([procedure](../docs/a100-hardware-counters.md#engine-counter-captures), [metric definitions](../wiki/methods/baseline-metrics.md#nsight-compute-engine-counter-captures-a100)). [Counter plan](../configs/ncu/a100-nunchaku-flux-klein-001-counters.json),
[controller](../scripts/profile_engine_counters.py). The template run [a100-nunchaku-flux-klein-001__baseline__20261011-124123](../results/runs/a100-nunchaku-flux-klein-001__baseline__20261011-124123/)
is reused unchanged except the protocol (1 first + 0 warm-up + 1 measured generation). Job 12038730 on gilbreth-g007.
Run: [a100-nunchaku-flux-klein-001__counters__20261029-001046](../results/runs/a100-nunchaku-flux-klein-001__counters__20261029-001046/) ([status](../results/runs/a100-nunchaku-flux-klein-001__counters__20261029-001046/status.json), [counter-summary.json](../results/runs/a100-nunchaku-flux-klein-001__counters__20261029-001046/profile/counter-summary.json),
[pass-kernels.json](../results/runs/a100-nunchaku-flux-klein-001__counters__20261029-001046/profile/pass-kernels.json)).

Scope: Nsight Compute kernel replay, no cache flush, clocks not locked. The pass is the last full denoising
pass of the measured generation, found as the repeating kernel-name period with the most kernel time. FLOPs
per pass stay analytic (34.82 T); counted Tensor Core + FP32 work is the check. Replay kernel time is not
latency; the benchmark time per pass comes from the template run.

| Quantity | Value |
|---|---:|
| Kernels per pass | 711 |
| DRAM read + write per pass | 43.64 GB |
| Counted operations per pass (ratio to analytic) | 0.85 T (0.024) |
| Arithmetic intensity (analytic FLOPs / measured bytes) | 797.9 FLOP/B |
| Summed replay kernel time per pass | 148.3 ms |
| Benchmark denoise time per pass (template run) | 165.8 ms |
| Analytic bytes per pass (earlier roofline estimate) | 16.24 GB (2.7× measured / analytic) |

Kernel time and DRAM bytes by kernel class (kernel-name rules in
[plot_roofline_measured.py](../scripts/plot_roofline_measured.py)): GEMM 250 kernels, 58 % of time, 19.8 GB; attention 25, 27 %, 3.1 GB; elementwise/reduction 395, 13 %, 19.0 GB; norm 41, 1 %, 1.8 GB.

Observations:

- Counted operations are 2.4 % of the analytic FLOPs: the W4A4 GEMMs run on INT4 Tensor Cores, which these metrics do not count. Intensity therefore uses the analytic FLOPs (798 FLOP/B).
- GEMMs and elementwise kernels move similar traffic (19.8 and 19.0 GB); the largest single transfers are PyTorch elementwise kernels.
- Measured traffic is 2.7× the analytic estimate.
- Summed replay kernel time is 89 % of the benchmark time per pass.

Interpretation: with 4-bit weights and activations the GEMMs move little data; the remaining traffic is split between them and BF16 elementwise work outside the quantized kernels.
