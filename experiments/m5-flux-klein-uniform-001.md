---
type: experiment-record
id: m5-flux-klein-uniform-001
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-uniform-001__conversion__20261102]
updated: 2026-11-02
---

# m5-flux-klein-uniform-001: full conventional INT4 transformer control

Week 5 execution variants, RQ1/RQ3. Convert all 109 matrix weights of distilled
Klein revision e7b7dc27f91deacad38e78976d1f2b499d76a294 to symmetric INT4,
with group size 128 and FP32 scales. Weight-only squared error selects among
five clipping fractions, as in the [slice control](m5-flux-klein-shiftadd-005.md).
Unchanged tensors are saved separately. Source weights and all prior bundles
are preserved. Config, per-matrix errors and hashes are in the
[run evidence](../results/runs/m5-flux-klein-uniform-001__conversion__20261102/).

Completed conversion, file reload/reconstruction, and independent SHA-256
verification of all 109 matrix artifacts. The bundle contains
**2,058,984,238 bytes (1.917579 GiB)**,
a **73.4363%** reduction from the source transformer file.
This counts packed files and unchanged tensors, excluding reports/configs.
It is not a measurement of live inference memory. Conversion timing is not
inference latency. No full-transformer images have been generated for this
conventional bundle; slice results do not establish full-model quality.

Local bundle: `shiftadd/artifacts/transformer-uniform-int4-001/`.
The loader and prepared paired image config support dense BF16 reconstruction;
no compressed GPU execution or performance claim is made.

Next experiment: run the prepared same-source paired image calibration control.

A full 3072×3072 query matrix also passed the chunked model-loader check: the
loaded BF16 parameter exactly equals independently decoded and BF16-rounded
weights. Evidence: `uniform-loader-real-layer.json` in the conversion run.
