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

## In progress / next
1. Jetson Orin Nano image baseline: a configuration that fits in 8 GB shared memory ([device](systems/jetson-orin-nano.md)).
2. DreamLite review ([reading list](papers/reading-list.md)).
3. Write the bounded planning hypothesis ([RQ3](rq/rq3.md)).
4. Shared initial noise for cross-engine image comparison ([D-006](project/decisions.md#d-006-shared-initial-noise-from-sdcpps-philox-rng-proposed)).

## Waiting on
- Advisor's recommendation for storing and viewing results ([D-007](project/decisions.md#d-007-results-stay-in-run-directories-viewer-undecided)).
- Nothing committed since `771641e`. The wiki, the sd.cpp kit and today's runs are uncommitted on `a100-flux-klein-baseline`.

## Open questions
3 unresolved ([open questions](open-questions.md)).
