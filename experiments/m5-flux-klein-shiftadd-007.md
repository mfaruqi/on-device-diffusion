---
type: experiment-record
id: m5-flux-klein-shiftadd-007
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-007__conversion__20261102]
updated: 2026-11-02
---

# m5-flux-klein-shiftadd-007: lossless mixed-radix metadata compaction

Week 5 execution variants, RQ1/RQ3. Change only the exponent serialization of
the existing mixed p4/p5 transformer bundle. The verified base-exponent method
is defined in [the uniform-radix compaction record](m5-flux-klein-shiftadd-006.md).
No weights are refitted; the original mixed bundle is preserved.

All 109 matrices pass exact sign-plane and expanded-exponent round-trip checks,
including all 21 five-plane matrices. A selected full five-plane `proj_out.weight`
also passes the chunked BF16 loader comparison exactly with `planes=None`.
[Raw evidence and verification](../results/runs/m5-flux-klein-shiftadd-007__conversion__20261102/).

The saved weight bundle is **2,040,280,318 bytes (1.900159 GiB)**, versus
2,135,621,820 bytes for the original mixed bundle. This saves **95,341,502 bytes**
(**4.4643%**) without changing decoded weights. Relative to the
same BF16 source transformer file, the reduction is **73.6776%**.
Sizes include weight files and unchanged tensors, excluding reports/configs.

This is storage compression, not a new quality result or a runtime memory,
speed or energy measurement. Per-row fitting-error and conversion-time fields
are inherited from the original fits; they do not measure metadata compaction.
The format preserves the mixed policy rather than changing the 21 selected layers.
Local bundle: `shiftadd/artifacts/transformer-radix-compact-mixed-001/`.
