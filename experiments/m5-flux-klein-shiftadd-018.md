---
type: experiment-record
id: m5-flux-klein-shiftadd-018
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-018__attempt__20261103-image-q-p3]
updated: 2026-11-03
---

# m5-flux-klein-shiftadd-018: three-plane image-query fits across five blocks

## Setup

Week 5 precision variants, RQ1/RQ3. Reduce the image-query `attn.to_q` matrices
in all five dual-stream blocks from four to three upstream BCQ planes. The
[four-plane reference](m5-flux-klein-shiftadd-015.md) uses the same source,
BF16 activation capture, group 128, u8/shared-FP16 scales, superblock 256,
15 iterations, act-order and damping 0.01. No other fitting setting changes.

[Config](../shiftadd/configs/upstream-bcq-all-image-q-p3-001.json),
[raw evidence](../results/runs/m5-flux-klein-shiftadd-018__attempt__20261103-image-q-p3/),
[metric definitions](../wiki/methods/baseline-metrics.md).
Each matrix uses eight fitting calls (256 sampled tokens from two prompts)
and eight calibration-validation calls from two other prompts. No held-out
prompts are used. This is CPU fitting and numerical evaluation only.

## Results

| Block | Three-plane mean fit L2 | Three-plane mean validation L2 | Four-plane mean validation L2 |
|---|---:|---:|---:|
| 0 | 0.00395669 | 0.02888011 | 0.01713898 |
| 1 | 0.00616564 | 0.05679617 | 0.03316742 |
| 2 | 0.00630517 | 0.06084703 | 0.03573075 |
| 3 | 0.00699756 | 0.06686012 | 0.03908318 |
| 4 | 0.00688043 | 0.06687740 | 0.03916380 |

Each file uses 3,859,484 bytes, with 3.25 payload bits/weight. Total file size is
19,297,420 bytes versus 25,564,300 for the five four-plane files: 6,266,880
bytes saved, 24.51%. In a 30-matrix dual-stream bundle with
all other matrices fixed at four planes, this would save 4.09% of packed file
storage. This is not resident memory or whole-model savings.

All 80 per-call errors reproduce after independent reload, with matching hashes.
The first-block Q3 file is byte-identical to the earlier
[query fit](m5-flux-klein-shiftadd-014.md). Packed decoding matches the upstream
fitter's dense output; no extra damping retry occurred.

## Interpretation

Three planes increase local validation projection error in every query matrix.
The smaller files provide a candidate for broader mixed precision, but final
image behavior cannot be inferred from these local errors. The capture comes
from a BF16 prefix and does not model accumulated quantization effects.
No GPU quality, latency or energy result is established by this fitting run.

## Next experiment
- Compare the 30-matrix bundle with all five image-query matrices at three planes against the all-four-plane reference.
