---
type: experiment-record
id: a100-flux-klein-shiftadd-activation-capture-002
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-shiftadd-activation-capture-002__attempt__20261103-calibration]
updated: 2026-11-03
---

# a100-flux-klein-shiftadd-activation-capture-002: all-attention BF16 calibration capture

Status: **complete**, job 12140057, 66 seconds allocation wall time, not inference latency.

## Setup

Week 5 precision calibration, RQ1/RQ3. Expand the
[initial two-module capture](a100-flux-klein-shiftadd-005.md) to 50 attention-related
projections while keeping the BF16 workload and sampling policy fixed.
The scope includes 30 dual-stream image/text Q/K/V matrices across five blocks
and 20 single-stream fused QKV/MLP projections. The latter include MLP output
rows as well as Q/K/V rows.

Pinned distilled Klein revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`,
512×512, four denoising steps, guidance 1, batch one, on A100 SXM4 node
gilbreth-n012 with Torch 2.5.1+cu121. The registry uses the existing cluster
slug. Four original calibration prompts and seeds, unchanged from the reference;
no held-out prompts. Each call samples 32 evenly spaced token vectors.

[Executed config, environment, source checks and validation](../results/runs/a100-flux-klein-shiftadd-activation-capture-002__attempt__20261103-calibration/).
[Metric definitions](../wiki/methods/baseline-metrics.md).

## Results

All 50 modules yielded 16 calls, 800 arrays total. Each array has shape
32×3072, FP32 storage of BF16-origin values, with all values finite.
The 32 arrays shared with the initial capture are exactly equal. Direct Q/K/V
input equality also holds for every call in both branches of all five dual-stream
blocks. Four consecutive calls correspond to each prompt in configuration order.

The local activation archive contains 314,835,542 bytes.
Its hash and local/remote locations are recorded in
[the artifact manifest](../results/runs/a100-flux-klein-shiftadd-activation-capture-002__attempt__20261103-calibration/activation-artifact.json).
The archive remains outside Git. No model weights changed during capture.

## Interpretation

Exact reproduction of the earlier sampled inputs supports using this expanded
capture for the same calibration procedure. Inputs are from an unquantized prefix;
this is not sequential recalibration after preceding layers are quantized.
The first two prompts can supply fitting inputs while the other two remain
calibration-validation prompts. CPU copies in the hooks invalidate timing and
energy interpretation. This run establishes neither compressed-model quality nor
runtime memory savings.

## Next experiment
- Fit all dual-stream attention weights with the existing four-plane upstream procedure.
