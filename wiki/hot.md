---
type: project
summary: Current state of the project — this week, in progress, next actions, blockers. Read first in every session; rewritten, not appended.
status: active
updated: 2026-10-02
---

# Hot: current state

**Week 1** (Sep 28 – Oct 2): image baselines and bounded planning hypothesis ([milestones](project/milestones.md)).
October 1 priorities: deeper profiling and engine/option baselines; FastVideo and further DreamLite work temporarily deferred ([D-009](project/decisions.md#d-009-prioritize-image-engine-baselines-and-profiling)).

## Done
- A100-PCIE baselines for FLUX.2 klein with two engines, plus their comparison ([registry](experiments.md)).
- Wiki set up ([SCHEMA.md](SCHEMA.md)).
- Jetson disk-backed sd.cpp repeated baseline, targeted profile and full 14-generation profile validated/exported; ten measured observations ([record](../experiments/jetson-flux-klein-003.md)).
- MLC documentation reviewed: [compiler precedents and proposed distinction](papers/mlc-compiler-precedent.md).
- Core reuse, precision and search papers reviewed; [DreamLite](papers/dreamlite.md) timing scope checked ([reading list](papers/reading-list.md)).
- Measurement refactor complete: readable runners, monitors, analysis and derived events;
  40 CPU checks pass locally/on Gilbreth, both A100 smoke checks pass ([validation](../scripts/README.md#compatibility-and-checks)).

## In progress / next
1. Profile kernels, transfers, loading and host overhead more closely, especially Jetson ([meeting actions](../raw/meetings/2026-10-01-haoran-you.md#to-do)); explain the step-0 captured-activity gap ([profile results](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/groups/jetson-flux-klein-003)).
2. Screen edge-dit.cpp first; keep LightX2V and TensorRT as unverified additional candidates ([reading list](papers/reading-list.md), [memo](../raw/meetings/2026-10-01-haoran-you.md#to-do)).
3. Audit sd.cpp auto-fit's resolved decisions; test supported memory options and EasyCache/DBCache individually before combinations ([memo](../raw/meetings/2026-10-01-haoran-you.md#to-do)).
4. Extend engine/device/model stage charts and collect operation/traffic data for roofline-style plots; label estimates and timing bases ([memo](../raw/meetings/2026-10-01-haoran-you.md#to-do)).
5. Generalize configuration-driven runners/adapters with shared outputs, pinned revisions and explicit unsupported-option failures ([memo](../raw/meetings/2026-10-01-haoran-you.md#to-do)).
6. Write the bounded planning hypothesis ([RQ3](rq/rq3.md)); audit recipe/novelty overlap ([ExecuTorch review](papers/executorch-mlsys2026.md#what-it-means-for-this-project), [reading list](papers/reading-list.md)).
7. Shared initial noise for cross-engine image comparison ([D-006](project/decisions.md#d-006-shared-initial-noise-from-sdcpps-philox-rng-proposed)).

## Waiting on
- Advisor's recommendation for storing and viewing results ([D-007](project/decisions.md#d-007-results-stay-in-run-directories-viewer-undecided)).
- W&B viewer: 14 runs retain timings with explicit [scope/basis and profiler labels](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion); links in the [registry](experiments.md). Advisor's view still pending.
- Medians export to Summary only; generation curves retain all images. Existing uploads/panels stay unchanged ([results viewer](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion)).
- Stage-mean overview and A100 detail charts uploaded as a separate [W&B analysis run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/00321d37da374213).

## Open questions
4 unresolved ([open questions](open-questions.md)).
