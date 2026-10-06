---
type: system
kind: device
summary: MacBook Air 15-inch M5, 24 GB unified memory — Metal laptop target (replaces the M1 MacBook Pro, D-010).
status: planned
updated: 2026-10-06
---

# MacBook Air 15" M5 (24 GB)

- **Status**: planned, no measurements yet. Replaces the [M1 MacBook Pro](m1-macbook-pro.md) as the Apple laptop target in
  [D-010](../project/decisions.md#d-010-deliver-a-tvm-compiled-klein-pipeline-with-the-planner-on-top).
- **Hardware** ([Apple specs](https://www.apple.com/macbook-air/specs/)): M5 with a 10-core CPU, a 10-core GPU
  with Neural Accelerators, a 16-core Neural Engine, 153 GB/s memory bandwidth and 24 GB unified memory.
- **Fanless** ([Apple newsroom](https://www.apple.com/newsroom/2026/03/apple-introduces-the-new-macbook-air-with-m5/)):
  sustained runs may throttle, so each run records the thermal state.
- **Paths**: Metal for the compiled pipeline and stable-diffusion.cpp
  ([milestones](../project/milestones.md#revised-schedule)). Core AI, Apple's newer framework,
  requires macOS 27 ([coreai-models](https://github.com/apple/coreai-models)); it is a roll-over candidate.
- Memory accounting covers whole-system unified memory ([overview](../project/overview.md#targets)).
