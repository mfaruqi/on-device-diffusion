---
type: system
kind: model
summary: DreamLite-mobile (0.39B U-Net, 4-step) — bounded iPhone workload and deployment reference, not a third full study.
status: planned
updated: 2026-10-02
---

# DreamLite-mobile

- **Status**: [Source review](../papers/dreamlite.md) ingested; local export and device validation remain pending. Further work is temporarily deferred ([D-009](../project/decisions.md#d-009-prioritize-image-engine-baselines-and-profiling)); no local measurements.
- Its iOS reference splits the pipeline across a 4-bit MLX text encoder and FP16 Core ML U-Net and VAE ([deployment review](../papers/dreamlite.md)). Never presented as an optimized FLUX result ([proposal](../project/overview.md#fallbacks-and-scope-control)).
