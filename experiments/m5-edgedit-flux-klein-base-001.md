---
type: experiment-record
id: m5-edgedit-flux-klein-base-001
status: failed
device: macbook-air-m5
engine: edge-dit-cpp
runs: [m5-edgedit-flux-klein-base-001__attempt__20261010-192312]
updated: 2026-10-11
---

# m5-edgedit-flux-klein-base-001

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 edge-dit Metal BF16 Base reference; 512x512, no cache, explicit placement and no automatic fitting.

[Configuration](../configs/m5-edgedit-flux-klein-base-bf16-512.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-edgedit-flux-klein-base-001__attempt__20261010-192312

[Status](../results/runs/m5-edgedit-flux-klein-base-001__attempt__20261010-192312/status.json), [execution config](../results/runs/m5-edgedit-flux-klein-base-001__attempt__20261010-192312/config.json). Status: failed.

Execution or validation failed; the error and partial measurements are retained. No performance claim.

The engine failed before generation: its parameter check reported missing and unexpected transformer tensors. The first missing tensor was `model.diffusion_model.double_blocks.0.img_mlp.0.weight`. The linked status file preserves the full error. This gate provides no inference latency measurement.

Interpretation: checkpoint tensor mapping needs investigation before another attempt; this failure does not establish that Metal inference is unsupported.

## Interpretation

The feasibility gate did not establish a valid full-protocol result. Full runs behind a failed gate remain unexecuted; precision, resolution and offload are not changed automatically.

## Next experiment

Investigate the recorded failure before another full-protocol attempt.

## Native checkpoint preparation

The [native BF16 configuration](../configs/m5-edgedit-flux-klein-base-bf16-512-native-smoke.json) retains the pinned Base revision. Its checkpoint preparation encountered repeated read timeouts and was stopped before timed optimization measurements ([preparation log](../output/m5-prep/native-weights-preparation.log)). The required verified native file was unavailable at the final queue check, so no native smoke or full protocol was launched. The original loading failure remains preserved; this preparation blocker is not a measured Base Metal execution failure.
