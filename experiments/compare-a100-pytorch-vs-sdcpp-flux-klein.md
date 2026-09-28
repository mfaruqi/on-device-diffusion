---
type: experiment-record
id: compare-a100-pytorch-vs-sdcpp-flux-klein
status: complete
device: a100-pcie-40gb
engine: [pytorch-diffusers, stable-diffusion-cpp]
runs: [a100-flux-klein-001__baseline__20260923-221530, a100-flux-klein-001__repeat__20260925-123908, a100-sdcpp-flux-klein-001__baseline__20260925-114328, a100-flux-klein-001__profile__20260923-221717, a100-sdcpp-flux-klein-001__profile__20260925-124703]
updated: 2026-09-25
---

# Compare: PyTorch/diffusers vs stable-diffusion.cpp, FLUX.2 [klein] 4B on A100-PCIE-40GB

## Question

With the same checkpoint, workload and protocol on the same GPU type, how do the PyTorch/diffusers
reference and stable-diffusion.cpp differ in end-to-end latency, per-stage latency, device memory and
kernel mix, and where does the difference come from? This serves RQ1 (which execution choices
transfer across engines and devices) and sets the engine baseline required by proposal §6.

## Runs

| Role | Run directory | Engine | Node | Job |
|---|---|---|---|---|
| Timing, PyTorch | `a100-flux-klein-001__baseline__20260923-221530` | diffusers 0.40.0, torch 2.5.1+cu121 | g008 | 11807390 |
| Timing, PyTorch (repeat) | `a100-flux-klein-001__repeat__20260925-123908` | same | g006 | 11817919 |
| Timing, sd.cpp | `a100-sdcpp-flux-klein-001__baseline__20260925-114328` | sd.cpp `master-919-19bbbca`, CUDA 12.6 | g004 | 11817906 |
| Profile, PyTorch | `a100-flux-klein-001__profile__20260923-221717` | torch.profiler | g008 | 11807396 |
| Profile, sd.cpp | `a100-sdcpp-flux-klein-001__profile__20260925-124703` | Nsight Systems | g005 | 11818087 |

Experiment notes: [a100-flux-klein-001.md](a100-flux-klein-001.md), [a100-sdcpp-flux-klein-001.md](a100-sdcpp-flux-klein-001.md).

## Matching

| Setting | PyTorch | sd.cpp | Matched? |
|---|---|---|---|
| GPU | A100-PCIE-40GB, 250 W, driver 590.48.01 | same (different g node) | yes |
| Transformer weights | diffusers `transformer/` @ `e7b7dc27` | `flux-2-klein-4b.safetensors` @ `e7b7dc27` | yes, value-identical (`configs/sdcpp-weights.check.json`) |
| Text encoder / VAE weights | `text_encoder/`, `vae/` @ `e7b7dc27` | same files | yes |
| Weight dtype | BF16 | BF16 (engine audit) | yes |
| Resolution / steps / guidance / batch | 1024², 4, 1.0, 1 | same | yes |
| Prompt, token padding | same prompt, padded to 512 | same prompt, min length 512 | yes |
| Sampler | FlowMatchEuler, dynamic shift | Euler, Flux2 scheduler (`mu=2.291`) | assumed equivalent; sigmas not compared value by value |
| Placement | all on GPU, no offload | all on GPU, auto-fit off, 1 graph segment | yes |
| Protocol | 1 first + 3 warm-up + 10 measured | same | yes |
| Initial noise | torch CUDA generator, seed 0 | sd.cpp `cuda` RNG, seed 0 | **no**: different noise, so images differ |

**Confounds** (differences in how each engine computes, part of what is compared, not settings):
- **Activation precision.** PyTorch keeps activations in BF16. ggml keeps them in FP32 between ops and
  converts to BF16 for GEMMs and to FP16 for flash attention. The sd.cpp VAE GEMMs run in FP16.
- **Timing method.** PyTorch stages use CUDA events. sd.cpp stages use host timestamps at synchronous
  callbacks ([baseline-metrics.md](../wiki/methods/baseline-metrics.md#stable-diffusioncpp-runs)). Stage boundaries
  differ slightly: PyTorch has a separate postprocess stage, sd.cpp folds it into VAE decode and "other".
- **Memory metric.** Only the NVML device-wide peak exists for both. Allocator metrics exist only for PyTorch.
- **Profilers.** torch.profiler vs Nsight Systems. Both give GPU kernel durations, but kernel groups
  differ in granularity (see below), so only totals are compared.

## Results

### End-to-end and stages (median of 10 measured runs, ms)

| Stage | PyTorch (g008) | PyTorch repeat (g006) | sd.cpp (g004) | sd.cpp / PyTorch |
|---|---|---|---|---|
| Text encode | 47.0 | 47.5 | 64.7 | 1.38× |
| Denoise (4 steps) | 980.8 | 979.5 | 1725.4 | 1.76× |
|   per step | 242–248 | – | 427–441 | 1.74× |
| VAE decode | 165.3 | 165.9 | 603.5 | 3.65× |
| Postprocess | 39.3 | 27.8 | (in decode / other) | – |
| Other | 7.8 | – | 28.7 | – |
| **End-to-end** | **1240.6** | 1229.6 | **2423.3** | **1.95×** |
| First run | 3163.2 | – | 2604.5 | 0.82× |
| Load (cached files → GPU, s) | 13.7 | – | 5.2 | 0.38× |

### Memory (GiB)

| Metric | PyTorch | sd.cpp |
|---|---|---|
| Weights | 14.87 | 14.87 (15,225 MB reported) |
| Device-wide after load | 15.85 (`load.json`) | 17.02 |
| **Device-wide peak during generation** | **20.61** | **22.88** |
| Stage that sets the peak | VAE decode | VAE decode (6.66 GB compute buffer) |

### Kernel time per denoise step (profiled runs, mean of 4 steps, ms)

| Group | PyTorch | sd.cpp | Δ |
|---|---|---|---|
| GEMM | 123.6 | 116.0 | −7.7 |
| Flash attention | 37.4 (~180 TFLOP/s) | 148.3 (~44 TFLOP/s) | **+110.9** |
| Everything else on the GPU (elementwise, norms, copies/casts, concat, memcpy) | 83.3 | 159.2 | **+75.9** |
|   of which copy/dtype conversion | 19.6 | 56.7 | +37.1 |
|   of which host↔device memcpy | 0.0 | 11.8 | +11.8 |
| GPU idle inside the step | 3.3 | 23.4 | +20.1 |
| **Step span** | **247.6** | **446.9** | **+199.3** |
| Kernels per step | 1431 | 1133 | |

Groups other than GEMM and attention aren't comparable one by one. For example, PyTorch runs
`rms_norm` as separate pow/mean/add/rsqrt/mul/cast kernels, while ggml has a fused `rms_norm_f32`.
So the table compares their total.

### VAE decode kernel time (profiled runs, ms)

| | PyTorch | sd.cpp |
|---|---|---|
| Decode GPU time | ~165 | 606.8 |
| Largest component | cuDNN conv + `group_norm` (whole-generation profile: conv 58, group_norm 75) | **`im2col_kernel` 403.7** (66.5%), `group_norm_f32` 56.2, FP16 GEMMs ~50 |

## Findings

1. **GEMMs transfer; attention doesn't.** With identical BF16 weights, the two engines' GEMM time per
   step is within 7% (sd.cpp slightly faster). Flash attention is 4× slower in ggml's F16 tensor-core
   kernel than in PyTorch's SDPA-flash for this shape (S = 4608, H = 24, D = 128). Attention accounts
   for 111 of the 199 ms per-step gap.
2. **ggml's FP32-activation design costs ~50 ms per step.** It adds 37 ms of extra dtype conversion
   and 12 ms of host↔device copies, and the GPU idles 20 ms more per step between graph launches.
3. **The VAE decode gap (3.65×) is almost entirely im2col.** Lowering convolutions to im2col + GEMM
   costs 404 ms and is the likely source of the 6.66 GB decode buffer. That buffer sets sd.cpp's device
   peak 2.3 GiB above PyTorch's.
4. **Startup favours sd.cpp.** It loads 2.6× faster (5.2 vs 13.7 s), and its first generation is
   only 7% slower than a warm one, versus 2.5× for PyTorch.
5. **Both engines are deterministic and stable** (spread 0.3–0.9%). The PyTorch reference repeats
   within 0.9% on a second node.

## Limits

- One GPU type, one workload (1024², 4 steps, batch 1), one prompt. Shares may shift at other resolutions
  or step counts.
- Output quality isn't compared: the initial noise differs between engines, so the images differ by
  construction. A quality comparison needs shared initial latents (planned reference mode).
- Sampler sigmas weren't compared value by value. Both use the FLUX.2 dynamic-shift schedule for 4096
  image tokens.
- The attention-throughput and im2col findings name the kernels responsible. Why ggml's FA kernel is
  slower at this shape isn't established.
- sd.cpp's own non-reference options (`--vae-conv-direct`, `--vae-tiling`, quantized weights, `--fa`
  for the text encoder) weren't run. Each is a separate labelled configuration.
