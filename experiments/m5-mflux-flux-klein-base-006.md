---
type: experiment-record
id: m5-mflux-flux-klein-base-006
status: failed
device: macbook-air-m5
engine: mflux
runs: [m5-mflux-flux-klein-base-006__attempt__20261015-163359]
updated: 2026-10-15
---

# m5-mflux-flux-klein-base-006

## Question

Week 2–3 Apple individual-option work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 MFLUX Klein-base q8: vae-tiled256 only versus configs/m5-mflux-flux-klein-base-q8-512.resolved.json; native MLX compilation retained.

[Configuration](../configs/m5-mflux-flux-klein-base-q8-512-vae-tiled256.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-mflux-flux-klein-base-006__attempt__20261015-163359

[Status](../results/runs/m5-mflux-flux-klein-base-006__attempt__20261015-163359/status.json), [execution config](../results/runs/m5-mflux-flux-klein-base-006__attempt__20261015-163359/config.json). Status: failed.

Execution or validation failed; the error and partial measurements are retained. No performance claim.

```text
AssertionError('AC power required')
```

## Interpretation

The power preflight stopped the smoke before model loading or inference because the Mac was on battery. This does not establish an engine or tiling failure. No timing samples or full protocol were produced.

## Next experiment

Retry the unchanged smoke after AC power is restored; a passing smoke gates the full protocol.

## Reference and option audit

[Reference](../experiments/m5-mflux-flux-klein-base-002.md). Only VAE tiling changes: 256-pixel tiles with 64-pixel overlap; native MLX compilation and quantization stay fixed. Tile-local normalization can change pixels and colors; no formal quality approval.
