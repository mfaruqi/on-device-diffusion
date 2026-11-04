---
type: experiment-record
id: m5-flux-klein-shiftadd-load-002
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-load-002__attempt__20261103-uncached, m5-flux-klein-shiftadd-load-002__attempt__20261103-cached]
updated: 2026-11-03
---

# m5-flux-klein-shiftadd-load-002: packed-array caching during upstream CPU loading

## Setup

Week 5 precision execution variants, RQ1/RQ3. Compare the same upstream fused
matrix, `single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight`, with packed
archive arrays read per chunk versus cached once per matrix. This is the only
variant. Both paths use 128-row reconstruction chunks and four CPU threads,
load into a 27,648×3,072 BF16 parameter, and verify the packed file hash.
The matrix is from the [verified upstream fit](m5-flux-klein-shiftadd-016.md).

Each configuration runs five fresh child processes: first reported separately,
one discarded warm-up, then three measured children. The uncached configuration
ran before the cached configuration. OS file cache and system pressure are not
controlled, so this is not cold-disk or sustained-performance evidence.

[Uncached config](../shiftadd/configs/upstream-load-uncached-001.json),
[cached config](../shiftadd/configs/upstream-load-cached-001.json),
[uncached raw evidence](../results/runs/m5-flux-klein-shiftadd-load-002__attempt__20261103-uncached/),
[cached raw evidence](../results/runs/m5-flux-klein-shiftadd-load-002__attempt__20261103-cached/),
[metric definitions](../wiki/methods/baseline-metrics.md).

`loader_wall_s` includes artifact/configuration reads, hashing, shape checks,
reconstruction and CPU parameter copies. It excludes imports, initial parameter
allocation and output hashing. Peak RSS covers the whole child lifetime,
including those excluded timing operations; it is not incremental decode memory.

## Results

| Packed-array policy | First load s | Median load s | Load min–max s | Median process peak GiB | Peak min–max GiB |
|---|---:|---:|---:|---:|---:|
| uncached | 1.062269 | 1.031287 | 1.031229–1.031549 | 0.449478 | 0.449341–0.449493 |
| cached | 0.292635 | 0.257474 | 0.255721–0.276330 | 0.449432 | 0.449066–0.450012 |

Caching reduces median loader time by a factor of **4.01** in this
single-matrix CPU diagnostic. Process peak RSS is approximately unchanged at
the reported scale; this does not mean caching allocates no extra arrays.
The fixed BF16 parameter contains 169,869,312 bytes.

Every child produces the identical BF16 parameter SHA256:
`1b2e3ebc551397fdbec78a78f32557b15160d7e1975c93518ab6dda23748e42a`.

A separate full-package replay with caching also passes all 169 parameter
checks: every tensor is finite, the BF16 remainder is unchanged, and all 50
packed matrices exactly match their expected BF16 reconstructions. This replay
uses the [complete package](m5-flux-klein-shiftadd-020.md) and is a numerical
check, not an additional timing sample. Its receipt is in the cached run.

## Interpretation and limits

The optional cache avoids repeatedly materializing complete arrays from NPZ
archives for each row chunk. It retains the packed arrays only for the current
matrix; dense reconstruction remains chunked. The default stays uncached so
previous experiment behavior is unchanged. No inference, GPU memory, GPU load
time, energy or full-transformer loading speedup is established by this test.
Three measurements in one ordering are insufficient for a general performance
claim across runtime conditions.

## Next experiment
- Measure the same explicit cache option during GPU pipeline loading separately from denoising.
