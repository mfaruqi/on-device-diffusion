---
type: experiment-record
id: compare-a100-base-pytorch-vs-sdcpp
status: complete
device: a100-pcie-40gb
engine: [pytorch-diffusers, stable-diffusion-cpp]
runs: [a100-flux-klein-base-001__baseline__20261005-074425, a100-sdcpp-flux-klein-base-001__baseline__20261005-080118]
updated: 2026-10-05
---

# Compare: A100 klein Base with PyTorch and sd.cpp

## Question

How do the existing unoptimized engines behave on the50-step Base workload? This serves Week1–2 image baselines and RQ1. It is a descriptive engine comparison, not a matched-initial-noise or matched-quality performance claim.

## Runs

| Engine | Run and saved config | Record |
|---|---|---|
| PyTorch/diffusers | [a100-flux-klein-base-001__baseline__20261005-074425](../results/runs/a100-flux-klein-base-001__baseline__20261005-074425/config.json) | [PyTorch Base001](a100-flux-klein-base-001.md) |
| stable-diffusion.cpp | [a100-sdcpp-flux-klein-base-001__baseline__20261005-080118](../results/runs/a100-sdcpp-flux-klein-base-001__baseline__20261005-080118/config.json) | [sd.cpp Base001](a100-sdcpp-flux-klein-base-001.md) |

## Matching

| Property | Both runs / limitation |
|---|---|
| Checkpoint | black-forest-labs/FLUX.2-klein-base-4B, a3b4f4849157f664bdbc776fd7453c2783562f4d |
| Weights and residency | BF16, fixed CUDA placement, no quantization/offloading/reuse/tiling; internal arithmetic and kernels differ |
| Workload |1024×1024,batch1,50steps,guidance4,empty negative prompt,same cat/sign prompt,max sequence length512 |
| Seed |0in both configs; initial noise differs across RNG implementations ([D-006](../wiki/project/decisions.md#d-006-shared-initial-noise-from-sdcpps-philox-rng-proposed)) |
| Schedule | Each native Flux2 path; numerical equality of timestep/sigma trajectories across engines is not verified |
| Protocol | One first,three discarded warm-ups,ten measured; no profiler attached |
| Hardware | A100-PCIE-40GB on gilbreth-g007; separate sequential jobs, not interleaved |
| Timing | Both synchronized end-to-end wall time; PyTorch stages use CUDA events, sd.cpp stages use host callbacks |
| Memory | Compare NVML device-wide sampled peaks only; PyTorch allocated/reserved counters have no equivalent in sd.cpp |

The engine is the intended variable. Initial noise, implementation arithmetic, unverified schedule equivalence and sequential-job conditions remain confounds, so this comparison is qualitative. Saved-image repeatability within each engine does not establish cross-engine fidelity.

## Results

Sources: [PyTorch summary](../results/runs/a100-flux-klein-base-001__baseline__20261005-074425/summary.json), [sd.cpp summary](../results/runs/a100-sdcpp-flux-klein-base-001__baseline__20261005-080118/summary.json). Stage medians are independently aggregated and need not sum to the end-to-end median. Stage timers have different boundaries/semantics.

| Metric | PyTorch median (min–max), ms | sd.cpp median (min–max), ms |
|---|---:|---:|
| wall_ms | 25147.679 (25054.344–25285.104) | 43686.406 (43637.599–43893.339) |
| text_encode_ms | 94.795 (93.626–103.741) | 119.130 (116.471–143.104) |
| denoise_ms | 24836.578 (24725.643–24975.987) | 42923.012 (42880.496–43130.816) |
| vae_decode_ms | 167.762 (166.576–169.206) | 612.247 (609.133–618.391) |
| other_ms | 12.982 (12.225–15.932) | 28.112 (26.248–29.689) |

PyTorch's separately recorded postprocess median is39.709ms; sd.cpp has no separately timed postprocess. Measured device-wide peak is20.614685GiB for PyTorch and22.880310GiB for sd.cpp. This is total sampled device usage, not allocator memory.

The observed sd.cpp/PyTorch wall-time ratio is **1.7372**. Denoising dominates both runs. Each engine produced14matching recorded hashes within its run; both retained PNGs were independently checked. No cross-engine quality metric or alignment benchmark was applied.

## Interpretation and limits

For these recorded configurations, PyTorch has the lower observed end-to-end latency and device-wide peak. This does not isolate the source of the engine difference or establish equal-quality superiority. The per-stage numbers locate where time is spent, but different timer boundaries prevent attributing every stage delta to kernel speed. The separate sd.cpp trace failed at profiler startup, so a paired kernel-level explanation remains pending. These results do not alone determine compute versus memory-bandwidth limits.
