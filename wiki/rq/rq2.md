---
type: rq
summary: RQ2 — can structure, hardware information and persistent history select effective plans at low cost. No evidence yet.
status: active
updated: 2026-09-28
---

# RQ2: Can structure, hardware information, and persistent history select effective plans at low cost?

**From the proposal**: "Compare rule-based selection, measured lookup, and a learned cost model using equal
profiling budgets. Test held-out workloads and report prediction error, search cost, constraint
violations, and latency relative to the best evaluated feasible plan." ([overview](../project/overview.md#research-questions))

## Evidence so far
None. The measurement format is the start of the history the planner needs: every run directory
records config, environment, per-stage latency and memory in the same schema for every engine
([metrics](../methods/baseline-metrics.md), [D-007](../project/decisions.md#d-007-results-stay-in-run-directories-viewer-undecided)).

## Gaps
- The registry and history record specification (week 2).
- Enough measured configurations to rank at all.

## Next
Specify the history record keys (checkpoint and graph fingerprint, workload, device, runtime version, policy version) from the existing run-directory fields.
