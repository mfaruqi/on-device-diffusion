---
type: experiment-record
id: m5-flux-klein-shiftadd-022
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-022__attempt__20261103-mixed-fused]
updated: 2026-11-03
---

# m5-flux-klein-shiftadd-022: fit-selected mixed planes for the fused projections

## Setup

Week 5 precision variants, RQ1/RQ3. Assemble a mixed candidate for the existing
50-matrix attention scope: choose ten of the twenty single-stream fused QKV/MLP
matrices to use three upstream planes; keep the other ten fused matrices and
all thirty dual-stream Q/K/V matrices at four planes. Source fits are the
[four-plane fused](m5-flux-klein-shiftadd-016.md),
[three-plane fused](m5-flux-klein-shiftadd-021.md), and
[four-plane dual-stream](m5-flux-klein-shiftadd-015.md) records.

All sources share checkpoint revision, BF16-prefix calibration capture, group
size, scale format, damping and fitting iterations. Each artifact has independent
reload verification. Assembly copies complete fitted matrices without refitting.
[Selection config](../shiftadd/configs/upstream-mixed-fused-selection-001.json),
[raw report and candidate audits](../results/runs/m5-flux-klein-shiftadd-022__attempt__20261103-mixed-fused/),
[metric definitions](../wiki/methods/baseline-metrics.md).

## Selection rule

The lower-plane count is fixed at ten before evaluating the candidate. Every
fused matrix yields the same 11,280,384-byte file saving when
changed from four planes to three. Rank matrices by the increase in mean relative
output L2 on the eight **fitting calls only**, and select the ten smallest
increases, breaking ties by module name. The other eight calibration-validation
calls are used only for the reported evaluation proxy. No held-out data are used.
A synthetic test deliberately reverses the validation-error ranking and verifies
that it does not change the selected fitting-based choice.

Selected single-stream block indices: **0, 1, 2, 3, 4, 5, 6, 7, 8, 11**. This is a local-error
heuristic, not a trained policy or the project's joint latency/memory planner.

## Saved candidate and checks

The 50 packed files occupy **960,728,440 bytes (0.894748 GiB)**, compared
with 1,073,532,280 bytes for the all-four-plane source files:
**10.51%** less for this subset. Storage includes NPZ headers
and row indices, but excludes assembly reports, the BF16 remainder, text encoder
and VAE. This is not the size of a complete standalone transformer or runtime
memory. The local bundle is `shiftadd/artifacts/upstream-bcq-mixed-fused-p3p4-001`.

The independent candidate audit reproduces the fitting-only ranking and checks
all 50 file hashes, complete row coverage, and exactly ten three-plane plus
forty four-plane matrices. Numerical source errors were already independently
reproduced in the fitting records; no new image run is claimed for this assembly.

## Diagnostic proxy

These values are the arithmetic mean of the chosen source per-matrix errors
across the twenty fused matrices. They are **not** a measurement of the combined
model. Inputs came from a BF16 prefix; error propagation and cancellation are
not modeled. The fixed dual-stream subset is excluded from this error average.

| Allocation among fused matrices | Fitting-call mean proxy | Calibration-validation mean proxy |
|---|---:|---:|
| All four-plane | 0.00593220 | 0.06321579 |
| Selected ten three-plane, ten four-plane | 0.00847737 | 0.08518847 |
| All three-plane | 0.01185715 | 0.11046296 |

`summary.json` preserves the full fixed-count diagnostic curve. The local figure
can be reproduced with `python -m shiftadd_flux.plot_plane_tradeoff --run
results/runs/m5-flux-klein-shiftadd-022__attempt__20261103-mixed-fused --output <new-path.png>` using the plots dependencies.

## Limits

The candidate saves file storage while increasing the local error proxy. It has
not been tested in the image pipeline and is not quality-admitted. There is no
measured inference speed, resident-memory or energy benefit. Selecting by local
relative output error does not guarantee good global image behavior. The curve
is not an image-quality Pareto frontier.

## Next experiment
- Compare this mixed candidate with the all-four-plane reference on the same calibration prompts before broader quality testing.
