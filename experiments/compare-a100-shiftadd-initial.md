---
type: experiment-record
id: compare-a100-shiftadd-initial
status: complete
device: a100-sxm4-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-shiftadd-001__smoke__20261102, a100-flux-klein-shiftadd-002__attempt__20261102-180000, a100-flux-klein-shiftadd-003__attempt__20261102-181000]
updated: 2026-11-02
---

# Compare: initial A100 ShiftAdd reconstructed-weight smoke tests

## Question

How do fitting method and mixed precision affect one calibration image?
Week 5 precision variants supporting RQ1/RQ3.

## Runs

| Record | Variable |
|---|---|
| [Greedy p4](a100-flux-klein-shiftadd-001.md) | Greedy residual power-of-two fitting |
| [Joint radix p4](a100-flux-klein-shiftadd-002.md) | Joint power-of-two signed-radix fit; separate A8/A4 single-projection diagnostics |
| [Mixed p4/p5](a100-flux-klein-shiftadd-003.md) | Same radix fit, with 21 matrices raised to five planes |

## Matching

Same source revision, A100 SXM4 node n012, Torch 2.5.1+cu121, 512-square,
four steps, guidance 1, seed 0, batch one and cat/sign prompt. Text encoder
and VAE remain BF16. All compressed transformer weights execute after dense
BF16 reconstruction. Packed representations differ as explicitly labelled.
The comparison uses weight-only cases; activation cases are separate changes.

One image per case, no warmups or repeats. Cold/warm state is unmatched for
timing, so timing comparisons are invalid. These are calibration diagnostics,
not held-out quality results. Seed matching does not by itself establish a
portable RNG protocol; this comparison stays within the same software/device.

## Results

Values from each linked record's summary.json:

| Whole-transformer representation | Relative latent L2 vs its BF16 reference | Visual inspection |
|---|---:|---|
| Greedy p4 | 1.113309 | Noise-like; requested subject absent |
| Joint radix p4 | 0.768493 | Recognizable cat/sign; changed lettering |
| Mixed radix p4/p5 | 0.595130 | Recognizable cat; readable sign |

## Findings

The initial greedy whole-transformer fit fails this visual smoke. Joint radix
fitting improves this example. Increasing precision on a weight-error-selected
subset further reduces the latent discrepancy on this example. None of these
observations establishes general image quality or the optimal allocation.

## Limits

No perceptual metric, alignment benchmark, blinded review or declared quality
threshold was evaluated. Latent L2 is a diagnostic, not a quality ranking.
The mixed allocation is based on weight error rather than measured activation
sensitivity. Signed-radix fitting is a restricted quantization-like baseline,
not a reproduction of output-aware ShiftAddLLM/ShiftAddQwen fitting.
No live compressed-memory, packed-kernel speed or energy comparison is available.
