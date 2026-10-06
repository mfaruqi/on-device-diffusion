---
type: finding
summary: On Jetson, sd.cpp EasyCache cuts a 50-step klein Base generation by 44% (25 of 50 steps skipped) but makes the 4-step distilled generation slower (no steps skipped).
status: supported
confidence: high
rq: [RQ1, RQ3]
sources: [../../experiments/compare-jetson-base-easycache.md, ../../experiments/compare-jetson-distilled-easycache.md]
updated: 2026-10-06
---

# Step caching helps at 50 steps but not at 4 on the Jetson

**Claim.** With the same EasyCache settings (threshold 0.2, active from 15% to 95% of steps), a 50-step
klein Base generation takes **184192.8 ms**
([`measured.wall_ms.median`](../../results/runs/jetson-flux-klein-base-002__baseline__20261005-024556/summary.json)),
down from about 329 s without caching, while the 4-step distilled generation takes **68241.9 ms**
([`measured.wall_ms.median`](../../results/runs/jetson-flux-klein-006__baseline__20261005-111428/summary.json)), slower than its
no-cache control.

**Evidence.** Matched-binary comparisons with no-cache controls, Q4 512², one prompt and seed:
[Base](../../experiments/compare-jetson-base-easycache.md), [distilled](../../experiments/compare-jetson-distilled-easycache.md).
Base skipped 25 of 50 steps in every generation; the 4-step run skipped none. Base PSNR/LPIPS against the
no-cache image are diagnostics only; no formal quality eligibility.

**Interpretation.** The same policy can be a large win or a net cost depending on schedule length, so a
fixed recipe cannot be right for both workloads. This motivates per-workload selection ([RQ3](../rq/rq3.md)).

Related: [cross-step reuse](../concepts/cross-step-reuse.md), [kernel, graph and plan search](../concepts/kernel-graph-plan-search.md).
