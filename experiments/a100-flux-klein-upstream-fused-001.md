---
type: experiment-record
id: a100-flux-klein-upstream-fused-001
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-fused-001__attempt__20261102-fused]
updated: 2026-11-02
---

# a100-flux-klein-upstream-fused-001: fused-projection upstream BCQ image diagnostic

Status: **complete** (2026-11-02, job 12134626).

## Question
Does replacing one fused QKV/MLP projection with upstream calibration-fitted BCQ
preserve the BF16 output more closely than radix and conventional INT4 controls?

## Setup
Week 5 precision variants, RQ1/RQ3. A100 SXM4 on gilbreth-n012; the device slug
follows the existing cluster registry. Distilled Klein revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`, 512×512, four steps, guidance 1,
batch one, seed 0, cat holding a hello-world sign calibration prompt.
Each candidate changes only `single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight`
(27648×3072); all other parameters remain BF16. Weights are reconstructed to dense
BF16 before execution. No activation quantization is applied.
Upstream u8 BCQ uses 4.3125 payload bits/weight; radix and conventional INT4 use
4.25. This is not a matched-storage comparison. The upstream artifact and fitting
protocol are described in the [conversion record](m5-flux-klein-shiftadd-010.md).

[Executed configuration, environment, audits and measurements](../results/runs/a100-flux-klein-upstream-fused-001__attempt__20261102-fused/).
[Metric definitions](../wiki/methods/baseline-metrics.md).
One generation per case, without discarded warm-ups or repeated measurements.
The reservation was reduced from ten to three minutes before execution; the
job completed successfully in 75 seconds. The frozen source and cases were unchanged.

## Results

| Case | Relative latent L2 vs paired BF16 |
|---|---:|
| bf16 | 0.0000000 |
| radix-p4-fused | 0.1231375 |
| uniform-int4-fused | 0.1213876 |
| upstream-u8-fused | 0.1334340 |

Values above come from `summary.json`; independently recomputed validation values
agree within floating-point reduction tolerance. Validation confirms exact BF16
latent agreement with the prior reference, finite outputs, and exactly one
replaced matrix per candidate. All four decoded images are 512×512.

Visual inspection: all four images depict the cat and legible “hello world” sign,
with small differences in facial details and framing. This is a single calibration
image, not formal quality admission or a held-out evaluation.

## Interpretation
For this projection and prompt, upstream BCQ has the largest latent deviation
among the compressed candidates despite its slightly larger weight payload.
Latent closeness does not by itself establish perceptual or prompt-following quality.
No packed-runtime memory, latency or energy result is measured; single-pass dense
BF16 timings must not be interpreted as low-bit inference speedups.

## Next experiment
- Extend the fused-projection comparison to the separate calibration-validation prompts before selecting a fitting method.

## Published evidence

[Verified W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-upstream-fused-001__attempt__20261102-fused) includes four images, matching quality metrics
and the evidence artifact. [Verification receipt](../results/runs/a100-flux-klein-upstream-fused-001__attempt__20261102-fused/wandb-verification.json).
