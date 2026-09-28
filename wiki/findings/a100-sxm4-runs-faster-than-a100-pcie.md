---
type: finding
summary: The same sd.cpp configuration runs 5.6% faster on Gilbreth's A100-SXM4-40GB nodes than on A100-PCIE-40GB nodes; the two must not share a device label.
status: supported
confidence: high
rq: [RQ1]
sources: [../../experiments/a100-sdcpp-flux-klein-001.md]
updated: 2026-09-28
---

# A100-SXM4 runs faster than A100-PCIE for the same configuration

**Claim.** The sd.cpp reference took **2294.8 ms**
([`measured.wall_ms.median`](../../results/runs/a100-sdcpp-flux-klein-001-20260925-114138/summary.json))
on an SXM4 node (400 W) and **2423.3 ms**
([`measured.wall_ms.median`](../../results/runs/a100-sdcpp-flux-klein-001-20260925-114328/summary.json))
on a PCIE node (250 W). Both runs have spreads under 0.4%.

**Evidence.** Same config, engine build and weights; only the node type differs
([record](../../experiments/a100-sdcpp-flux-klein-001.md#repeatability)).

**Consequence.** Every Gilbreth job is pinned to PCIE nodes ([D-002](../project/decisions.md#d-002-pin-gilbreth-jobs-to-a100-pcie-nodes)).

Related: [A100-PCIE-40GB](../systems/a100-pcie-40gb.md), [A100-SXM4-40GB](../systems/a100-sxm4-40gb.md).
