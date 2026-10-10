---
type: rq
summary: RQ3 — does joint planning with bounded runtime adaptation beat fixed or independently tuned policies. Jetson option evidence shows workload-dependent and non-additive effects; draft planning hypothesis and planner design written; no planner tested.
status: active
updated: 2026-10-06
---

# RQ3: Does joint planning with bounded runtime adaptation outperform fixed or independently tuned policies?

**From the proposal**: "Isolate region selection, memory planning, and runtime guards under matched
checkpoints and quality requirements. Identify where saved computation outweighs cache storage, probe
costs, and runtime overhead." ([overview](../project/overview.md#research-questions))

## Evidence so far
No planner has been tested. Fixed-configuration evidence that bears on the question:
- the same step cache is a large win at 50 steps and a net cost at 4 ([finding](../findings/step-caching-helps-at-50-steps-but-not-at-4-on-jetson.md));
- combined reuse savings overlap rather than add ([finding](../findings/combined-reuse-savings-overlap-on-jetson-base.md));
- on the Jetson, residency (reloading) dominates few-step generations ([finding](../findings/weight-reloading-dominates-disk-backed-jetson-generations.md)),
  and one component's reuse changes another's reload cost ([finding](../findings/exact-conditioning-reuse-removes-most-jetson-reload-time.md));
- the engine's rule-based automatic placement failed on the Jetson ([finding](../findings/sdcpp-auto-fit-fails-first-text-encoding-on-jetson.md));
- A100 levers: text-encoder residency ([finding](../findings/text-encoder-half-of-weights-little-of-time.md)), VAE peak
  ([finding](../findings/vae-decode-sets-peak-memory.md)), fusable glue work ([finding](../findings/pytorch-denoise-step-third-in-unfused-memory-bound-ops.md)).

## Bounded planning hypothesis (draft, 2026-10-06)
Hypothesis: on one device and one checkpoint, under several memory budgets and a frozen quality
criterion, a planner that jointly selects precision, reuse and residency from measured history achieves
lower median latency than (1) a tuned fixed configuration, (2) the engine's automatic fitting, (3) existing
adaptive caching alone, and (4) options tuned independently, with planning and calibration cost counted.
The search space is bounded to a few supported variants per lever (on the order of tens of plans)
before compatibility filtering ([proposal objective](../project/overview.md#planner-objective)).

## Planner design (proposed, not implemented)
Inputs: model, workload, memory budget, quality requirement. Steps: model adapter (stages, timestep loop,
tensor lifetimes, invariants) → registry of variants with contracts → reject incompatible plans → check
peak memory over the whole execution timeline, including loading transitions → rank with measured history
→ measure a few uncertain candidates → compile the chosen plan for the device → runtime with
pre-validated guards → append measurements to history ([system components](../project/overview.md#system-components),
[levels](../concepts/kernel-graph-plan-search.md)).

## Candidate contributions (hypotheses)
1. **Timeline-aware residency.** Model residency as a schedule over the denoising loop, including loading
   and file-cache interference between components. Motivated by the reload-cost interaction above.
2. **Schedule-aware lever choice.** Choose reuse or residency levers per workload and schedule length,
   since their value reverses between 4 and 50 steps.
3. **Error-budget allocation.** Split a declared quality budget between quantization and approximate
   reuse, measured jointly on calibration prompts. Overlaps with CacheQuant and Q&C, which must be
   reviewed before any novelty claim ([reading list](../papers/reading-list.md)).

## Gaps
- No memory budget defined per target device; no frozen quality criterion.
- Registry and history record specification ([RQ2](rq2.md)).
- The compiled pipeline that the planner would control ([D-010](../project/decisions.md#d-010-deliver-a-tvm-compiled-klein-pipeline-with-the-planner-on-top)).

## Next
Review CacheQuant and Q&C against the three candidate contributions. Build a planner over existing engine
options first ([D-011](../project/decisions.md#d-011-build-the-compiled-pipeline-mlc-style-on-tvm-with-an-engine-level-planner-fallback-proposed)),
then over compiled variants.

Related: [RQ1](rq1.md), [RQ2](rq2.md).
