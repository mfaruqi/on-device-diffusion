---
type: experiment-record
id: a100-flux-klein-upstream-all-attention-001
status: complete
device: a100-sxm4-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-all-attention-001__attempt__20261103-calibration]
updated: 2026-11-03
---

# a100-flux-klein-upstream-all-attention-001: all attention and fused-projection image comparison

Status: **complete**, job 12140980, 242 seconds allocation wall time, not inference latency.

## Setup

Week 5 precision variants, RQ1/RQ3. Compare BF16 with radix p4, conventional
symmetric INT4 and upstream BCQ on the same 50 matrices: image/text Q/K/V across
all five dual-stream blocks plus all 20 single-stream fused QKV/MLP projections.
Remaining transformer weights and all activations stay BF16. The upstream
artifacts come from the verified [dual-stream](m5-flux-klein-shiftadd-015.md)
and [single-stream](m5-flux-klein-shiftadd-016.md) fits.
Packed weights reconstruct to dense BF16 before execution.

A100 SXM4 on gilbreth-n009, Torch 2.5.1+cu121. Placement was broadened from
n012 to matching N-feature SXM4 nodes while pending; the update receipt is retained. Distilled Klein revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`, 512×512, four steps, guidance 1,
batch one. Four calibration prompts/seeds: teapot (17), lighthouse (23), three
lemons (31), ECO AI robot (47). The first two supplied fitting inputs; the
latter two are calibration-validation prompts. No held-out prompts were used.

[Executed configuration, environment, audits and measurements](../results/runs/a100-flux-klein-upstream-all-attention-001__attempt__20261103-calibration/).
[Metric definitions](../wiki/methods/baseline-metrics.md).
One generation per case/prompt, without repeated timing measurements.
Upstream uses 4.3125 payload bits/weight; radix and uniform INT4 use 4.25.
The fitting objective and scale family also differ, so the comparison does not
isolate activation calibration from representation changes.

## Results

Relative latent L2 against paired BF16 (BF16 is zero for each prompt):

| Prompt | Radix p4 | Uniform INT4 | Upstream BCQ |
|---|---:|---:|---:|
| cal01 | 0.2667292 | 0.2970542 | 0.1956372 |
| cal02 | 0.4933472 | 0.4596097 | 0.3091596 |
| cal03 | 0.4770473 | 0.4785375 | 0.3365283 |
| cal04 | 0.5646527 | 0.4224983 | 0.4791622 |

All 16 outputs are finite and have the configured dimensions. Each candidate
audit lists exactly the same 50 requested replacements. All four BF16 latents
exactly reproduce the earlier calibration run and its PNGs are byte-identical.
Independent CPU norms pass rtol 1e-5, atol 1e-8, allowing FP32 reduction variation.

## Observations

The 12 candidate images were inspected alongside the previously inspected,
byte-identical BF16 references. Every case retains recognizable teapot,
lighthouse, lemon/bottle and robot scenes; all robot signs read ECO AI.
All lemon images show two visible lemons rather than the requested three.
The radix lighthouse loses the attached building and changes the tower
geometry; the INT4 lighthouse includes a prominent foreground rock. Robot body
and face details, reflections and object geometry differ across candidates.

## Interpretation

Upstream BCQ has lower BF16 latent distance than radix on all four prompts
and lower distance than uniform INT4 on three. Uniform INT4 is closer on the
robot prompt. This supports upstream fitting as a candidate across the tested
50 matrices, but does not establish formal matched quality, full-transformer
superiority, or an advantage at equal bit budgets. The remaining 59 matrix
weights are unchanged BF16.
Dense reconstruction establishes no GPU resident-memory, latency or energy savings.

## Next experiment
- Add grouped A8/A4 inputs to these 50 upstream projections against the same weight-only reference.

## Published evidence

[Verified W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-upstream-all-attention-001__attempt__20261103-calibration) contains all 16 images, metrics and the evidence artifact. [Receipt](../results/runs/a100-flux-klein-upstream-all-attention-001__attempt__20261103-calibration/wandb-verification.json).
