---
type: project
summary: Current state of the project — this week, in progress, next actions, blockers. Read first in every session; rewritten, not appended.
status: active
updated: 2026-09-28
---

# Hot: current state

**Week 1** (Sep 28 – Oct 2): A100 and Jetson image baselines, DreamLite review, bounded planning
hypothesis ([milestones](project/milestones.md)).

## Done
- A100-PCIE baselines for FLUX.2 klein with two engines, plus their comparison ([registry](experiments.md)).
- Wiki set up ([SCHEMA.md](SCHEMA.md)).
- Jetson CUDA execution and sd.cpp verified; disk-backed quantized generation saved one image
  ([record](../experiments/jetson-flux-klein-003.md)); visual smoke check passed.

## In progress / next
1. Jetson: CUDA events confirmed in supplied Nsight stats; import/validate original trace, then
   rerun/validate the callback-corrected [stage-labelled harness](methods/jetson-howto.md#confirmed-installation-status)
   and instrument repeated timings ([record](../experiments/jetson-flux-klein-003.md)).
2. DreamLite review ([reading list](papers/reading-list.md)).
3. Write the bounded planning hypothesis ([RQ3](rq/rq3.md)).
4. Shared initial noise for cross-engine image comparison ([D-006](project/decisions.md#d-006-shared-initial-noise-from-sdcpps-philox-rng-proposed)).

## Waiting on
- Advisor's recommendation for storing and viewing results ([D-007](project/decisions.md#d-007-results-stay-in-run-directories-viewer-undecided)).
- W&B viewer: all 7 runs exported, linked per experiment in the [registry](experiments.md). Advisor's view still pending.

## Open questions
4 unresolved ([open questions](open-questions.md)).
