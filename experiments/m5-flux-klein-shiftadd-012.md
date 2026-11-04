---
type: experiment-record
id: m5-flux-klein-shiftadd-012
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-012__attempt__20261102-kv-activation]
updated: 2026-11-02
---

# m5-flux-klein-shiftadd-012: full K/V activation-error diagnostic

## Setup

Week 5 precision variants, RQ1/RQ3. Local CPU numerical evaluation of the
complete first-block K/V matrices from [upstream fitting](m5-flux-klein-shiftadd-011.md).
The reference is each fixed weight matrix with original captured inputs; separate
variants change only activation precision to symmetric A8 or A4, group size 32.
BF16-weight controls isolate activation error. All products are FP32, with
BF16-origin weights and inputs; quantized activations are rounded and decoded.
No image generation or performance timing is part of this experiment.

Six cases per matrix: BF16 or fixed upstream weights, each with original,
A8 or A4 inputs. The pinned checkpoint, upstream artifact hashes and capture
hash are in [the summary](../results/runs/m5-flux-klein-shiftadd-012__attempt__20261102-kv-activation/summary.json).
The shared-input capture provenance is documented in the K/V fitting record.
First eight calls are fitting inputs; the remaining eight come from two separate
calibration prompts. Neither split is the formal held-out evaluation.

[Config](../shiftadd/configs/upstream-kv-activation-error-001.json),
[raw outputs](../results/runs/m5-flux-klein-shiftadd-012__attempt__20261102-kv-activation/), and
[metric definition](../wiki/methods/baseline-metrics.md).
Reproduce with `PYTHONPATH=shiftadd/src python -m shiftadd_flux.diagnostics.upstream_activation_probe
--config shiftadd/configs/upstream-kv-activation-error-001.json` from the repository root.

## Results

Mean per-call relative output L2 on the separate calibration-validation inputs,
always relative to the original inputs multiplied by original weights:

| Projection | Weights | Activations | Mean output L2 |
|---|---|---|---:|
| K | bf16 | a4 | 0.04214002 |
| K | bf16 | a8 | 0.00252516 |
| K | bf16 | original | 0.00000000 |
| K | upstream_u8 | a4 | 0.04603995 |
| K | upstream_u8 | a8 | 0.02014463 |
| K | upstream_u8 | original | 0.02000314 |
| V | bf16 | a4 | 0.06989618 |
| V | bf16 | a8 | 0.00409034 |
| V | bf16 | original | 0.00000000 |
| V | upstream_u8 | a4 | 0.07754892 |
| V | upstream_u8 | a8 | 0.03435017 |
| V | upstream_u8 | original | 0.03411849 |

All 192 comparisons are finite. All 32 original-input upstream errors reproduce
the prior fitting record within rtol 1e-5, atol 1e-8, and all 32 BF16/original
controls are exactly zero ([validation](../results/runs/m5-flux-klein-shiftadd-012__attempt__20261102-kv-activation/validation.json)).
Fitting-split values and every individual call are preserved in the summary.

## Interpretation

A8 group 32 adds little projection error to these fixed upstream weights on
these inputs. A4 group 32 adds substantially more. This supports testing A8
before A4 in a combined image diagnostic, but cannot establish an image-quality
threshold. Inputs come from the original BF16 prefix, and only two prompts
supply the separate validation split. The check neither fits activation-aware
weights nor measures a GPU integer kernel, resident memory, latency or energy.

## Next experiment
- Add grouped A8 to a paired first-block upstream Q/K/V image diagnostic after the weight-only check.
