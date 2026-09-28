---
type: system
kind: engine
summary: stable-diffusion.cpp (ggml) — C/C++ diffusion engine with CUDA/Metal/Vulkan backends; a proposal baseline.
status: active
aliases: [sd.cpp, sdcpp, ggml]
updated: 2026-09-28
---

# stable-diffusion.cpp

- **Role**: engine baseline named in the proposal ([overview](../project/overview.md#evaluation-rules)).
- **Version used**: tag `master-919-19bbbca`, ggml `4bf5f600`, built for sm_80 ([how-to](../methods/sdcpp-howto.md)).
- **Runner**: load-once harness `engines/sdcpp/bench.cpp` driven by `scripts/run_sdcpp.py`
  ([D-004](../project/decisions.md#d-004-drive-sdcpp-through-a-load-once-harness-not-sd-cli)). Stage times come from host
  timestamps at sd.cpp's callbacks; memory is NVML device-wide only.
- **Reference settings**: all on `cuda0`, auto-fit off, no graph segmentation, eager load, no
  conditioning cache ([D-003](../project/decisions.md#d-003-stable-diffusioncpp-reference-settings)).
- **Computation**: FP32 activations between ops. GEMMs go through cuBLAS BF16; flash attention through
  ggml's own F16 kernel; VAE convolutions as im2col + FP16 GEMM ([record](../../experiments/a100-sdcpp-flux-klein-001.md#profiler-run-diagnostic-nsight-systems-job-11818087)).
- **Loads**: single-file diffusion model, sharded text encoder via its safetensors index, diffusers-format VAE.
- **Noise**: its own Philox RNG, seeded per image ([concept](../concepts/initial-noise.md)).
- **Engine options not yet measured**: `--vae-conv-direct`, `--vae-tiling`, `--offload-to-cpu`, auto-fit,
  `--cache-mode` (step caching), GGUF quantized weights.
