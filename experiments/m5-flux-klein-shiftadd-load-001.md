---
type: experiment-record
id: m5-flux-klein-shiftadd-load-001
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-load-001__attempt__20261102-whole, m5-flux-klein-shiftadd-load-001__attempt__20261102-chunk128]
updated: 2026-11-02
---

# m5-flux-klein-shiftadd-load-001: bounded-row reconstruction host memory

## Setup

Week 5 execution variants, RQ1/RQ3. Compare whole-matrix decoding against
128-row chunks for the same compact radix-p4 matrix:
`single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight`, shape 27648×3072.
Only `decode_chunk_rows` changes. Destination parameters are BF16 on CPU,
169,869,312 bytes. This is a single-matrix loading diagnostic, not inference.

Each configuration runs four fresh Python child processes, with the first
reported separately and three subsequent observations summarized. There are
no warmups inside a child: the quantity is fresh-process lifetime peak RSS,
so pre-running the loader would contaminate the memory high-water mark.
OS file caches are uncontrolled; no loading-time claim is made. Configurations
execute sequentially, whole first then chunked. Environments and child results
are archived with the runs below. No GPU job is used.

`resource.ru_maxrss` is converted from bytes on Darwin (KiB on Linux in the
portable utility). It covers the entire child including imports, the zero-filled
destination matrix, packed input, decoding and output hashing; it is not a
measured delta or a stage-aligned allocation metric.

## Results

| Decode policy | First peak RSS (GiB) | Subsequent median (GiB) | Subsequent min–max (GiB) |
|---|---:|---:|---:|
| whole | 3.244629 | 3.424957 | 3.424820–3.425018 |
| chunk128 | 0.404739 | 0.404770 | 0.404694–0.404907 |

All eight processes produce identical BF16 parameter SHA-256:
`a139d96d8184d086b1fd107f8a33038510725bbb16b0c75012f7f7886e79dd18`. The loader verifies the selected packed file before applying it.

Evidence:
- [m5-flux-klein-shiftadd-load-001__attempt__20261102-whole](../results/runs/m5-flux-klein-shiftadd-load-001__attempt__20261102-whole/)
- [m5-flux-klein-shiftadd-load-001__attempt__20261102-chunk128](../results/runs/m5-flux-klein-shiftadd-load-001__attempt__20261102-chunk128/)

## Interpretation and limits

Row chunks reduce temporary dense plane/weight arrays and the process high-water
RSS for this matrix. This supports bounded reconstruction as a loading preparation.
The destination remains dense BF16. These numbers do not measure a full-model
peak, GPU memory, Jetson memory, steady inference latency or energy. No assumption
is made that the difference transfers quantitatively to another device or loader.
