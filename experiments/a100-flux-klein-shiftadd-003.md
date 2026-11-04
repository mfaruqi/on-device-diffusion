---
type: experiment-record
id: a100-flux-klein-shiftadd-003
status: complete
device: a100-sxm4-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-shiftadd-003__attempt__20261102-181000]
updated: 2026-11-02
---

# a100-flux-klein-shiftadd-003: mixed p4/p5 reconstructed-weight smoke

## Question

Does a mixed four/five-plane transformer preserve a recognizable output on the
initial calibration prompt? Week 5 precision variants, RQ1/RQ3.

## Setup

Job 12133710, node gilbreth-n012, A100 SXM4 40GB. Distilled Klein 4B revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`, 512-square, four steps, guidance 1,
batch one and seed 0. Prompt: “A cat holding a sign that says hello world”.
[Config](../results/runs/a100-flux-klein-shiftadd-003__attempt__20261102-181000/config.json).
The archival directory stamp is not a measured start time.

The signed-radix representation uses groups of 128. Twenty-one matrices receive
five planes; remaining matrices receive four. Selection sorts four-plane matrix
relative reconstruction error, admitting at most 15% of matrix elements into
five-plane storage. This is a weight-error heuristic, not activation-calibrated
sensitivity. All weights are reconstructed into dense BF16 parameters before
inference; text encoder and VAE stay BF16. No activation quantization is applied.
One BF16 reference image and one mixed-weight image, with no warmups or repeats.

## Results

[Summary](../results/runs/a100-flux-klein-shiftadd-003__attempt__20261102-181000/summary.json):
mixed-weight relative latent L2 versus BF16 is 0.595130; outputs are finite.
The image shows a recognizable cat holding a readable “hello world” sign, with
visible differences from the BF16 reference. One prompt cannot establish quality
eligibility. Local images: `shiftadd/runs/a100-mixed-paired-001/`.

The logical mixed weight bundle uses 2,135,621,820 bytes (about 1.989 GiB), including
unchanged tensors, according to the local mixed storage report. This is storage,
not live inference memory. Both p4 and p5 experimental artifacts remain preserved.
No packed-kernel speed or energy is measured. The recorded generation-to-latent
observations exclude installation and VAE decode and are cold/warm-confounded.

## Interpretation

Mixed signed-radix reconstruction passes this initial functional visual check.
Matched-quality claims require additional calibration prompts and declared
criteria. The representation's storage savings do not imply execution savings.

## Next experiment

- Broaden calibration prompts with frozen BF16, uniform-p4 and mixed-p4/p5 settings.
