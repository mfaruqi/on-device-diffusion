---
type: system
kind: engine
summary: PyTorch + Hugging Face diffusers (Flux2KleinPipeline) — the reference engine for FLUX.2 klein on CUDA.
status: active
aliases: [diffusers, pytorch, torch]
updated: 2026-09-28
---

# PyTorch / diffusers

- **Role**: reference engine. Every other engine and variant is compared against its BF16 configuration ([D-001](../project/decisions.md#d-001-reference-configuration-and-one-change-per-labelled-configuration)).
- **Versions used**: torch 2.5.1+cu121, diffusers 0.40.0, transformers 5.16.1 ([record](../../experiments/a100-flux-klein-001.md#results)).
- **Runner**: `scripts/run_flux.py`. Stage times use CUDA events around `encode_prompt`, each transformer
  forward, `vae.decode` and `postprocess`; memory comes from allocator counters and NVML ([metrics](../methods/baseline-metrics.md)).
- **Computation**: BF16 weights and activations; attention via SDPA, which picks the flash backend;
  the VAE uses cuDNN convolutions ([record](../../experiments/a100-flux-klein-001.md#profiler-run-diagnostic-job-11807396)).
- **Noise**: `torch.Generator("cuda")` per run. Accepts external `latents=` ([concept](../concepts/initial-noise.md)).
- **Known behaviour**: large first-run cost, and a third of each denoise step in unfused ops
  ([finding](../findings/pytorch-denoise-step-third-in-unfused-memory-bound-ops.md)).
