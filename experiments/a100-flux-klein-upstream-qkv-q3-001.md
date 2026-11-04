---
type: experiment-record
id: a100-flux-klein-upstream-qkv-q3-001
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-qkv-q3-001__attempt__20261103-calibration]
updated: 2026-11-03
---

# a100-flux-klein-upstream-qkv-q3-001: three-plane Q with four-plane K/V

Status: **complete**, job 12140128, 53 seconds allocation wall time, not inference latency.

## Setup

Week 5 precision variants, RQ1/RQ3. Reduce only first-block Q from four to three
upstream BCQ planes relative to the
[four-plane Q/K/V reference](a100-flux-klein-upstream-qkv-calibration-001.md).
K and V remain four-plane upstream BCQ. Both candidates replace
`transformer_blocks.0.attn.to_q.weight`, `to_k.weight` and `to_v.weight`.
Remaining weights and all activations stay BF16. The bundle combines
[the three-plane query fit](m5-flux-klein-shiftadd-014.md) with existing K/V
artifacts, without refitting. Transferred file hashes and case selection passed
preflight checks.

A100 SXM4 node gilbreth-n012, Torch 2.5.1+cu121, with the existing cluster
registry device slug. Distilled Klein revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`, 512×512, four steps, guidance 1,
batch one. The same four calibration prompts/seeds as the reference are used:
teapot (17), lighthouse (23), three lemons (31), ECO AI robot (47).
The first two supplied fitting inputs; the latter two are calibration-validation
prompts. No held-out prompts were used.

[Executed config, environment, provenance and measurements](../results/runs/a100-flux-klein-upstream-qkv-q3-001__attempt__20261103-calibration/).
[Metric definitions](../wiki/methods/baseline-metrics.md).
One generation per case/prompt, without repeated latency measurements.
Packed weights reconstruct to dense BF16 for execution.

## Results

Relative latent L2 against paired BF16 (zero for BF16):

| Prompt | Q4/K4/V4 | Q3/K4/V4 |
|---|---:|---:|
| cal01 | 0.0354433 | 0.0435952 |
| cal02 | 0.0898966 | 0.0764791 |
| cal03 | 0.1460473 | 0.1346028 |
| cal04 | 0.1423134 | 0.1034672 |

The three-matrix Q3 bundle occupies 14,085,204 file bytes, versus 15,338,580
for Q4/K4/V4: 1,253,376 bytes saved, or 8.17%. This is packed file storage,
not whole-model or resident GPU memory savings.

All 12 outputs are finite and images have the configured dimensions. Both
reference cases exactly reproduce earlier latents, and their eight PNGs are
byte-identical to the prior comparison. Each candidate audit confirms exactly
three matrix replacements. CPU norm recomputation passes rtol 1e-5, atol 1e-8.

## Observations

The four Q3 images were visually inspected alongside the previously inspected,
byte-identical reference images. All remain recognizable: red teapot, watercolor
lighthouse, lemons with green bottle, robot with readable ECO AI sign. Each
lemon case shows two visible lemons rather than the requested three.

## Interpretation

Q3 is closer to BF16 than Q4 on three of four prompts, including both separate
calibration-validation prompts. This non-monotonic image-level error shows that
increasing local projection precision need not decrease final latent distance
for every prompt. It does not establish better image quality or generalize beyond
this first-block subset and small prompt set. There is no measured latency,
resident-memory or energy advantage from dense BF16 reconstruction.

## Next experiment
- Compare Q3, K3 and V3 at equal file storage using the completed matched image runs.

## Published evidence

[Verified W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-upstream-qkv-q3-001__attempt__20261103-calibration) contains all 12 images, metrics and the evidence artifact. [Receipt](../results/runs/a100-flux-klein-upstream-qkv-q3-001__attempt__20261103-calibration/wandb-verification.json).
