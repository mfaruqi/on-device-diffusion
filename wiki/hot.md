---
type: project
summary: Current state of the project — this week, in progress, next actions, blockers. Read first in every session; rewritten, not appended.
status: active
updated: 2026-09-29
---

# Hot: current state

**Week 1** (Sep 28 – Oct 2): A100 and Jetson image baselines, DreamLite review, bounded planning
hypothesis ([milestones](project/milestones.md)).

## Done
- A100-PCIE baselines for FLUX.2 klein with two engines, plus their comparison ([registry](experiments.md)).
- Wiki set up ([SCHEMA.md](SCHEMA.md)).
- Jetson CUDA execution and sd.cpp verified; disk-backed quantized generation saved one image
  ([record](../experiments/jetson-flux-klein-003.md)); visual smoke check passed.
- Measurement refactor complete: readable runners, monitors, analysis and derived events;
  40 CPU checks pass locally/on Gilbreth, both A100 smoke checks pass ([validation](../scripts/README.md#compatibility-and-checks)).

## In progress / next
1. Jetson: corrected stage profile imported, reviewed and in W&B; next, prepare
   repeated unprofiled measurements ([record](../experiments/jetson-flux-klein-003.md#corrected-harness-profile)).
2. DreamLite review ([reading list](papers/reading-list.md)).
3. Write the bounded planning hypothesis ([RQ3](rq/rq3.md)).
4. Shared initial noise for cross-engine image comparison ([D-006](project/decisions.md#d-006-shared-initial-noise-from-sdcpps-philox-rng-proposed)).
5. Refactored Jetson capture still needs hardware validation; full repeated/profile
   validation remains separate from the small A100 smoke checks ([guide](../scripts/README.md#compatibility-and-checks)).

## Waiting on
- Advisor's recommendation for storing and viewing results ([D-007](project/decisions.md#d-007-results-stay-in-run-directories-viewer-undecided)).
- W&B viewer: 11 runs exported (Jetson 003 profile added), linked per experiment in the [registry](experiments.md). Advisor's view still pending.

## Open questions
4 unresolved ([open questions](open-questions.md)).
