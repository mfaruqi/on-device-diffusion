---
type: rq
summary: RQ2 — can structure, hardware information and persistent history select effective plans at low cost. Seed measurements exist; selection untested.
status: active
updated: 2026-10-06
---

# RQ2: Can structure, hardware information, and persistent history select effective plans at low cost?

**From the proposal**: "Compare rule-based selection, measured lookup, and a learned cost model using equal
profiling budgets. Test held-out workloads and report prediction error, search cost, constraint
violations, and latency relative to the best evaluated feasible plan." ([overview](../project/overview.md#research-questions))

## Evidence so far
None for selection itself. Measured history now exists in the shared run-directory schema across two
devices and four engines, including single-option and combined configurations with no-cache controls
([registry](../experiments.md), [metrics](../methods/baseline-metrics.md)). The options measured on the Jetson show
effects that simple rules get wrong: a cache that helps at 50 steps hurts at 4, and the engine's
rule-based placement failed ([step caching](../findings/step-caching-helps-at-50-steps-but-not-at-4-on-jetson.md),
[auto-fit](../findings/sdcpp-auto-fit-fails-first-text-encoding-on-jetson.md)).

Interpretation: the learned cost model in RQ2 is the plan-level analogue of TVM's learned kernel cost
models (AutoTVM, MetaSchedule): it predicts which candidates are worth measuring. Plan measurements cost
whole generations, so the profiling budget matters more than at kernel level
([levels](../concepts/kernel-graph-plan-search.md)).

## Gaps
- The registry and history record specification.
- A defined candidate set per device and a quality criterion, so rankings can be compared.

## Next
Specify the history record keys (checkpoint and graph fingerprint, workload, device, runtime version, policy
version) from the existing run-directory fields, and seed them from the Jetson and A100 option campaigns.
