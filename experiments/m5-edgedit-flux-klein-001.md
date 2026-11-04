---
type: experiment-record
id: m5-edgedit-flux-klein-001
status: failed
device: macbook-air-m5
engine: edge-dit-cpp
runs: [m5-edgedit-flux-klein-001__attempt__20261010-183705, m5-edgedit-flux-klein-001__attempt__20261011-043503]
updated: 2026-10-11
---

# m5-edgedit-flux-klein-001

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 edge-dit Metal BF16 distilled reference; 512x512, no cache, explicit placement and no automatic fitting. Native BFL transformer file at the same checkpoint revision; required by the pinned loader.

[Configuration](../configs/m5-edgedit-flux-klein-bf16-512-native.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-edgedit-flux-klein-001__attempt__20261010-183705

[Status](../results/runs/m5-edgedit-flux-klein-001__attempt__20261010-183705/status.json), [execution config](../results/runs/m5-edgedit-flux-klein-001__attempt__20261010-183705/config.json). Status: failed.

Execution or validation failed; the error and partial measurements are retained. No performance claim.

The original Diffusers-layout attempt failed its parameter check: 65 missing and 65 unexpected tensors, with zero shape mismatches. Metal initialized, but model generation was not reached. The linked status file retains the full error.

### m5-edgedit-flux-klein-001__attempt__20261011-043503

[Status](../results/runs/m5-edgedit-flux-klein-001__attempt__20261011-043503/status.json), [execution config](../results/runs/m5-edgedit-flux-klein-001__attempt__20261011-043503/config.json). Status: failed.

Execution or validation failed; the error and partial measurements are retained. No performance claim.

The native-weight retry loaded 149 transformer tensors from the top-level Flux2 file, passed the earlier loading barrier and reached Metal graph execution. It aborted with exit code -6: `ggml_metal_op_encode_impl: error: unsupported op 'CUSTOM'`. The linked status file retains the full error and backtrace. No completed image or valid latency result was produced.

## Interpretation

The feasibility gate did not establish a valid full-protocol result. Full runs behind a failed gate remain unexecuted; precision, resolution and offload are not changed automatically.

## Next experiment

Blocked: the pinned Metal backend requires support for the encountered custom operation before the unchanged baseline can pass its gate. The bounded native smoke retry is exhausted.
