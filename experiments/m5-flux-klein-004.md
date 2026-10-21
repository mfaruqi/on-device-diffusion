---
type: experiment-record
id: m5-flux-klein-004
status: failed
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-004__attempt__20261015-131752, m5-flux-klein-004__attempt__20261015-132823]
updated: 2026-10-15
---

# m5-flux-klein-004

## Question

Week 2–3 Apple individual-option work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 BF16 Klein distilled: compile only versus configs/m5-flux-klein-bf16-512.resolved.json; no conditioning reuse, offload or tiling.

[Configuration](../configs/m5-flux-klein-bf16-512-compile.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-flux-klein-004__attempt__20261015-131752

[Status](../results/runs/m5-flux-klein-004__attempt__20261015-131752/status.json), [execution config](../results/runs/m5-flux-klein-004__attempt__20261015-131752/config.json). Status: complete.

The initial smoke completed its first and measured images with a compiled full graph. It did not include the same-input numerical diagnostic, so it is retained as execution evidence only. [Compiler receipt](../results/runs/m5-flux-klein-004__attempt__20261015-131752/compiler.json) and [summary](../results/runs/m5-flux-klein-004__attempt__20261015-131752/summary.json) retain kernel counters, stage timings and substantial swap pressure. No full protocol was admitted from this attempt.

### m5-flux-klein-004__attempt__20261015-132823

[Status](../results/runs/m5-flux-klein-004__attempt__20261015-132823/status.json), [execution config](../results/runs/m5-flux-klein-004__attempt__20261015-132823/config.json). Status: failed.

Execution or validation failed; the error and partial measurements are retained. No performance claim.

```text
AssertionError('Compiled transformer numerical gate failed')
```

## Interpretation

The feasibility gate did not establish a valid full-protocol result. Full runs behind a failed gate remain unexecuted; precision, resolution and offload are not changed automatically.

## Next experiment

Investigate the recorded failure before another full-protocol attempt.

## Reference and option audit

[Reference](m5-flux-klein-001.md). Compilation targets transformer.forward only, with fullgraph Inductor and no eager error fallback. The [identical-input check](../results/runs/m5-flux-klein-004__attempt__20261015-132823/compile-numerical-check.json) measured relative RMSE 0.02090166 against the 0.01 engineering threshold. The full protocol is blocked. This is a numerical compatibility gate, not a formal image-quality threshold. The diagnostic first generation includes an extra eager transformer forward.
