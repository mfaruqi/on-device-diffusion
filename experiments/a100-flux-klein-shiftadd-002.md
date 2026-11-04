---
type: experiment-record
id: a100-flux-klein-shiftadd-002
status: complete
device: a100-sxm4-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-shiftadd-002__attempt__20261102-180000]
updated: 2026-11-02
---

# a100-flux-klein-shiftadd-002: signed-radix p4 and single-projection activation smoke

## Question

Does joint signed-radix fitting yield a recognizable image with all 109 transformer
matrix weights reconstructed, and can one query projection tolerate activation
quantization? Week 5 precision variants, RQ1/RQ3.

## Setup

Job 12133607, A100 SXM4 node gilbreth-n012. Distilled Klein 4B pinned to
`e7b7dc27f91deacad38e78976d1f2b499d76a294`, 512-square, four steps, guidance 1,
batch one, seed 0. Prompt: “A cat holding a sign that says hello world”.
[Exact config](../results/runs/a100-flux-klein-shiftadd-002__attempt__20261102-180000/config.json).
The run-directory stamp is an archival identifier, not a measured start time.

Four sign planes with groupwise power-of-two units are fitted jointly by searching
five candidate exponents. This is a restricted signed-radix representation,
not a reproduction of output-aware ShiftAddQwen. Packed weights are reconstructed
into dense BF16 parameters. Text encoder and VAE remain BF16.

Cases: BF16 reference; all transformer matrices reconstructed; then per-token
symmetric fake INT8 or INT4 activation quantization on
`transformer_blocks.0.attn.to_q` only. Each activation hook ran four times.
Floating-point GEMMs remain in use. One image per case; no warmups or repeats.

## Results

[Summary](../results/runs/a100-flux-klein-shiftadd-002__attempt__20261102-180000/summary.json):

| Case | Relative latent L2 vs BF16 | Finite |
|---|---:|---|
| p4 reconstructed weights | 0.768493 | yes |
| p4 + selected-projection A8 | 0.770522 | yes |
| p4 + selected-projection A4 | 0.784832 | yes |

All three compressed-weight cases produce a recognizable cat and sign. Composition
and lettering differ from BF16. The weight-only and A8 images show distorted
lettering; the A4 image reads “hello world” in this individual sample. This is
qualitative inspection of one calibration image, not an accuracy estimate or an
activation-precision ranking. Images remain under
`shiftadd/runs/a100-radix-paired-001/` locally.

No packed-kernel latency, live compressed-model memory, or energy was measured.
The generation-to-latent timings exclude weight installation and VAE decoding;
the first reference is cold and later cases are warm. They are not speedup evidence.

## Interpretation

Joint signed-radix p4 is functionally viable for this calibration example.
Large latent differences and a single prompt preclude matched-quality approval.
The activation result applies only to the named projection. Broader calibration,
matched source/software, and repeated performance measurements remain necessary.

## Next experiment

- `shiftadd/configs/a100-mixed-smoke-001.json`: mixed p4/p5 weight allocation against BF16.
