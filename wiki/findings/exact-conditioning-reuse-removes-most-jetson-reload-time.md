---
type: finding
summary: On Jetson with disk-backed weights, reusing the encoded prompt cuts a 4-step distilled generation about 3.9x, more than skipping text encoding alone explains.
status: supported
confidence: medium
rq: [RQ3]
sources: [../../experiments/compare-jetson-distilled-conditioning-reuse.md, ../../experiments/compare-jetson-base-conditioning-reuse.md]
updated: 2026-10-06
---

# Exact conditioning reuse removes most of the Jetson reload time

**Claim.** With sd.cpp's conditioning cache on, a repeated-prompt 4-step generation takes **15734.5 ms**
([`measured.wall_ms.median`](../../results/runs/jetson-flux-klein-007__baseline__20261005-113106/summary.json)) against
**60936.0 ms** ([`measured.wall_ms.median`](../../results/runs/jetson-flux-klein-003__baseline__20261005-105911/summary.json))
for the matched no-cache control.

**Evidence.** [Distilled comparison](../../experiments/compare-jetson-distilled-conditioning-reuse.md) and
[Base comparison](../../experiments/compare-jetson-base-conditioning-reuse.md). Text encoding drops to near zero, and the
engine-reported transformer reload in the denoising stage also becomes much shorter. Reuse is exact, but
it only applies when the same prompt repeats, which every run here does.

**Hypothesis.** Skipping the text encoder avoids re-reading its file, so the transformer's file stays in
the Linux page cache and reloads faster. File-cache residency was not measured
([open question](../open-questions.md#is-jetson-weight-reload-time-storage-bound-or-page-cache-bound)).

Related: [stage residency](../concepts/stage-residency.md), [weight reloading](weight-reloading-dominates-disk-backed-jetson-generations.md).
