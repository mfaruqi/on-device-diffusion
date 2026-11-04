---
type: experiment-record
id: m5-flux-klein-shiftadd-006
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-006__conversion__20261102]
updated: 2026-11-02
---

# m5-flux-klein-shiftadd-006: lossless radix exponent metadata compaction

Week 5 precision execution variants, RQ1/RQ3. Change only stored exponent
metadata for the existing radix-p4 transformer bundle. For each nonzero group,
plane exponents are base + plane index; zero groups use −128 in every plane.
The converter verifies this relationship for every group before storing one
int8 base instead of four int8 exponents. Sign planes remain unchanged.
The original bundle is preserved. Source hashes, resulting hashes and per-matrix
sizes are in [evidence](../results/runs/m5-flux-klein-shiftadd-006__conversion__20261102/).

## Results

All 109 matrices pass exact sign-plane and expanded-exponent round-trip checks.
The new bundle has **1,968,125,182 bytes (1.832959 GiB)**, versus 2,058,956,988
bytes for the original radix bundle: **90,831,806 bytes saved**, or
**4.4115%** additional reduction. Relative to the same BF16 source
file, storage is reduced by **74.6085%**. Payload is now
4 + 8/128 = **4.0625 bits per weight**, before file headers and unchanged tensors.
A full 3072×3072 query matrix loaded in 128-row chunks exactly matches the
original radix bundle's BF16 reconstruction.

## Interpretation and limits

This is lossless metadata compression of a restricted radix representation,
not a better weight fit. Prior fitting errors are inherited unchanged. Per-row
`conversion_and_verification_s` entries in the derived summary are inherited
source fitting times, not compaction times; they must not be used to time this
operation. No inference latency, peak memory or energy is measured. The same
base-exponent technique does not apply to arbitrary independently fitted
ShiftAdd planes; the converter rejects exponent arrays without the relationship.
Decoded weights are identical, but no new full-pipeline runtime claim is made.

The full-source conventional INT4 bundle uses unrestricted FP32 scales, so its
4.25-bit payload is no longer exactly matched to this compact representation.
Future equal-storage comparisons must account for this change explicitly.
