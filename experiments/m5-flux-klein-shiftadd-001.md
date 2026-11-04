---
type: experiment-record
id: m5-flux-klein-shiftadd-001
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-001__conversion__20261102]
updated: 2026-11-02
---

# m5-flux-klein-shiftadd-001: four-plane transformer storage conversion

## Question

How much local weight storage does a four-plane power-of-two representation save
against the pinned distilled Klein BF16 transformer? Week 5 precision variant,
RQ1/RQ3; this is compressed-checkpoint preparation, not inference benchmarking.

## Setup

Source revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`. All 109 matrix
weights are converted with greedy residual sign fitting, groups of 128 along
input channels, four planes, and one signed int8 exponent per group/plane.
Other tensors are retained in `unchanged.safetensors`. A zero scale uses exponent
-128. This is not the output-aware GPTQ fit of ShiftAddQwen.

[Config and roundtrip evidence](../results/runs/m5-flux-klein-shiftadd-001__conversion__20261102/config.json).
Each packed NPZ was reloaded, reconstructed, checked for finite values and
shape agreement, and hashed. Files remain locally in
`shiftadd/artifacts/transformer-p4-001/` and are excluded from git.

## Results

[Storage and checksum report](../results/runs/m5-flux-klein-shiftadd-001__conversion__20261102/storage-report.json):

- Source transformer file: 7,751,109,744 bytes.
- Packed matrix files plus unchanged tensors: 2,058,956,988 bytes.
- Weight-file reduction: 73.4366%.
- Per-matrix relative L2 reconstruction error: median 0.214634,
  minimum 0.096357, maximum 0.452466.

Relative L2 is `norm(original - reconstructed) / norm(original)` per matrix.
These are weight diagnostics, not image quality metrics. Payload storage is
4.25 bits per converted weight before file headers. Timings in the summary
cover conversion plus artifact verification, not generation.

## Interpretation

The representation reduces stored transformer weights, but the greedy fit has
substantial numerical error. No measured inference-memory, speed, image-quality
or energy benefit is established. Dense reconstruction retains BF16 parameter
storage. A full pipeline includes additional text-encoder and VAE weights.

## Next experiment

- `shiftadd/configs/a100-paired-smoke-001.json`: paired reconstructed-weight latent and image smoke.
