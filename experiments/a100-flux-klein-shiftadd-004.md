---
type: experiment-record
id: a100-flux-klein-shiftadd-004
status: complete
device: a100-sxm4-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-shiftadd-004__attempt__20261102-182000]
updated: 2026-11-02
---

# a100-flux-klein-shiftadd-004: four-prompt calibration smoke

## Question

Do uniform and mixed signed-radix weights retain recognizable output beyond the
initial cat/sign example? Week 5 precision variants, RQ1/RQ3.

## Setup

Job 12133739, node gilbreth-n012, A100 SXM4 40GB. Distilled Klein source revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`. BF16 reference, uniform p4 and mixed
p4/p5; packed weights reconstructed into dense BF16 parameters. Text encoder and
VAE remain BF16. No activation quantization in this run. 512-square, four steps,
guidance 1, batch one; matched seeds within each prompt. Four new calibration-only
prompts (teapot, lighthouse, lemons, robot/sign); held-out prompt sets untouched.
[Config](../results/runs/a100-flux-klein-shiftadd-004__attempt__20261102-182000/config.json).
Archival directory stamp is not a measured start time. No warmups or repeated timing.

## Results

All 12 outputs are finite. Relative latent L2 to each prompt's BF16 reference,
from [summary](../results/runs/a100-flux-klein-shiftadd-004__attempt__20261102-182000/summary.json):

| Prompt | Uniform p4 | Mixed p4/p5 |
|---|---:|---:|
| cal01 | 0.427306 | 0.531186 |
| cal02 | 0.778353 | 0.708644 |
| cal03 | 0.820249 | 0.681429 |
| cal04 | 0.903983 | 0.782929 |

Visual inspection: all images retain recognizable requested subjects. Teapot
shape, lighthouse composition, bottle size and robot proportions change. The
three-lemon prompt produces two visible lemons in BF16 and mixed precision,
but three in uniform p4. The robot's ECO AI text is readable in all three cases.
These observations are not a benchmark score, blinded review or quality approval.
Images: `shiftadd/runs/a100-calibration-001/`; comparison figure:
`shiftadd/runs/figures/calibration-comparison.png`.

## Interpretation

The compressed representations remain functional across this small calibration
sample. Distance to BF16 cannot substitute for prompt correctness. No packed
execution memory, speed or energy benefit was measured. Generation-to-latent
observations omit weight installation and VAE decode and include cold/warm effects;
they must not be presented as speedups.

## Next experiment

- Activation-aware calibration with frozen quality criteria before choosing a deployment format.
