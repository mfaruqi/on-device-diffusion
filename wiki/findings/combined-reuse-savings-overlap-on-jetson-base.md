---
type: finding
summary: On Jetson klein Base, EasyCache plus exact conditioning reuse is faster than either alone but saves less than the two separate savings added together.
status: supported
confidence: medium
rq: [RQ3]
sources: [../../experiments/compare-jetson-base-combined-reuse.md]
updated: 2026-10-06
---

# Combined reuse savings overlap on the Jetson Base workload

**Claim.** EasyCache and conditioning reuse together give **141012.3 ms**
([`measured.wall_ms.median`](../../results/runs/jetson-flux-klein-base-007__baseline__20261005-121955/summary.json)), against
**184192.8 ms** ([`measured.wall_ms.median`](../../results/runs/jetson-flux-klein-base-002__baseline__20261005-024556/summary.json))
for EasyCache alone. Adding the two single-option savings would predict about 135 s.

**Evidence.** [Combined comparison](../../experiments/compare-jetson-base-combined-reuse.md): same binary, workload and
protocol; the combined range lies below both single options. Sequential captures leave file-cache and
thermal history uncontrolled, so the ratios are descriptive.

**Interpretation.** Option savings are not additive, so combinations must be measured together, as the
proposal requires ([evaluation rules](../project/overview.md#evaluation-rules)).

Related: [RQ3](../rq/rq3.md), [step caching](step-caching-helps-at-50-steps-but-not-at-4-on-jetson.md).
