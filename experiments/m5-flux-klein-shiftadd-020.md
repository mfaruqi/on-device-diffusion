---
type: experiment-record
id: m5-flux-klein-shiftadd-020
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-020__attempt__20261103-package]
updated: 2026-11-03
---

# m5-flux-klein-shiftadd-020: complete transformer package with an upstream subset

## Setup

Week 5 precision variants, RQ1/RQ3. Change the saved layout of the pinned BF16
transformer by substituting the existing 50 upstream matrix artifacts and
preserving all remaining tensors in BF16. No additional fitting or quantization
is performed. The packed matrices are the verified
[30 dual-stream Q/K/V](m5-flux-klein-shiftadd-015.md) and
[20 fused single-stream QKV/MLP](m5-flux-klein-shiftadd-016.md) conversions.

[Packaging configuration](../shiftadd/configs/upstream-transformer-package-001.json),
[raw evidence](../results/runs/m5-flux-klein-shiftadd-020__attempt__20261103-package/),
[format and loading instructions](../shiftadd/docs/upstream-package.md).
Source revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`; source SHA256 is
verified before writing. CPU packaging and loading use the environment in the
raw record. The local package is
`shiftadd/artifacts/upstream-bcq-transformer-package-001` and is not public.

## Saved storage

Figures are logical bytes, not allocated disk blocks or runtime memory.

| Measurement | Bytes |
|---|---:|
| Original transformer safetensors file | 7,751,109,744 |
| Packed matrices plus BF16 remainder file | 4,861,018,408 |
| Complete package including metadata | 4,861,217,420 |

The complete saved package is **4.527362 GiB**,
**37.2836%** smaller than the original
transformer file. It contains 50 packed tensors and 119 unchanged BF16 tensors,
covering all 169 transformer tensors. The remainder includes 59 unconverted
matrix weights and the other non-matrix tensors. Text encoder, VAE, tokenizer,
scheduler and pipeline configuration are excluded. The original source and all
other candidates remain on disk, so this does not measure total workspace use.

## Verification

All packed file hashes match their source artifacts; every tensor in the saved
BF16 remainder exactly matches the original checkpoint. The complete tensor
inventory, shapes, representations, file sizes and hashes are in `manifest.json`.

A full `Flux2Transformer2DModel` was constructed from the packaged architecture
configuration with real CPU BF16 parameter storage. The new loader read only
the package: a guarded safetensors call accepted its fallback file and would
reject paths outside the bundle. All 169 loaded parameters are finite. The 119
fallback parameters match the saved BF16 tensors exactly; the other 50 match
chunkwise BF16 reconstruction of their packed artifacts exactly.

The synthetic loading test additionally points provenance at a nonexistent
source and successfully loads the bundle. It checks known decoded values and
rejects a corrupted packed file before changing model parameters.

## Limits

This is a complete transformer **weight package**, not a Hugging Face pipeline
directory. Loading reconstructs dense BF16 parameters; no GPU image run,
resident-memory reduction, inference latency or energy benefit is measured.
Packaging temporarily holds the unconverted source tensors in CPU memory.
The package does not change the decoded weights relative to the source fits.

## Next experiment
- Use the complete package in a separately labelled paired image-loading check.
