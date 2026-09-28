---
type: system
kind: device
summary: NVIDIA Jetson Orin Nano (8 GB shared CPU/GPU memory) — CUDA edge target.
status: active
updated: 2026-09-28
---

# Jetson Orin Nano

- **Status**: CUDA execution verified; disk-backed quantized 512×512 generation reported one
  saved image that passed a visual smoke check. Repeated benchmark remains pending
  ([record](../../experiments/jetson-flux-klein-003.md)).
- **Software observed**: Ubuntu 22.04.5 LTS, Jetson Linux R36.4.4, kernel
  `5.15.148-tegra`, ARM64, CUDA 12.6 / nvcc V12.6.68
  ([setup evidence](../../raw/jetson-setup-2026-09-28.md)).
- **Access and procedures**: [Jetson how-to](../methods/jetson-howto.md), including the
  last known SSH address and the CUDA PATH fix.
- 8 GB of memory shared between CPU and GPU, so memory accounting must cover total physical memory and must not double-count mapped memory ([overview](../project/overview.md#targets)).
