---
type: experiment-record
id: compare-upstream-attention-storage
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-015__attempt__20261103-all-dual, m5-flux-klein-shiftadd-016__attempt__20261103-all-single]
updated: 2026-11-03
---

# Compare: upstream attention storage and whole-transformer estimate

## Question and evidence

Week 5 precision variants, RQ1/RQ3. How much storage is saved by the verified
50-matrix upstream conversion, and what would it imply if the remaining tensors
were stored in BF16?

| Completed source | Configuration | Scope |
|---|---|---|
| [Dual-stream fit](m5-flux-klein-shiftadd-015.md), [run](../results/runs/m5-flux-klein-shiftadd-015__attempt__20261103-all-dual/) | [config](../shiftadd/configs/upstream-bcq-all-dual-attention-001.json) | 30 image/text Q/K/V matrices |
| [Single-stream fit](m5-flux-klein-shiftadd-016.md), [run](../results/runs/m5-flux-klein-shiftadd-016__attempt__20261103-all-single/) | [config](../shiftadd/configs/upstream-bcq-all-single-attention-001.json) | 20 fused QKV/MLP matrices |

Both use the same source revision, four planes, group 128, u8/shared-FP16 scales
and complete matrix rows. Their matrix sets are disjoint. The variable is the
stored representation of each selected tensor. No timing or image-quality
comparison is made here. Source file hash, shapes and byte offsets were checked
against the pinned checkpoint and preserved in the
[source inventory](../results/runs/m5-flux-klein-shiftadd-016__attempt__20261103-all-single/source-checkpoint-inventory.json).

## Measured subset storage

Packed file bytes include NPZ headers, plane data, scales and row indices;
BF16 bytes are the corresponding tensor payload, excluding its container header.
Reports, calibration captures, duplicate saved candidates, text encoder and VAE
are excluded. Figures are logical file sizes, not filesystem allocation or RAM.

| Matrix subset | BF16 payload bytes | Packed artifact file bytes | Reduction |
|---|---:|---:|---:|
| 30 dual-stream Q/K/V | 566,231,040 | 153,385,800 | 72.91% |
| 20 fused QKV/MLP | 3,397,386,240 | 920,146,480 | 72.92% |
| Combined 50 | 3,963,617,280 | 1,073,532,280 | 72.92% |

The combined files occupy 0.999805 GiB versus 3.691406 GiB
for those BF16 weights. The selected matrices account for
51.14% of the original transformer's tensor payload.

## Whole-transformer deployment estimate

The original transformer has 7,751,089,152 tensor-payload bytes;
its actual safetensors file is 7,751,109,744 bytes, including the header.
Unconverted tensors account for 3,787,471,872 payload bytes. Keeping those in
BF16 and substituting the packed 50-matrix files gives:

`remaining BF16 payload + selected packed files = 3,787,471,872 + 1,073,532,280 = 4,861,004,152 bytes`

That is approximately **4.527 GiB**, or **37.29%** below
the original transformer payload. This is a deployment-layout estimate, not a
new standalone checkpoint that has been saved. It excludes the future fallback
container's header and manifest. The original full source file remains on disk
for the current loader; workspace disk usage is therefore larger.

## Limits

The image runner reconstructs dense BF16 parameters, so the estimate does not
describe its runtime memory. Text encoder, VAE, activations, workspaces and
loading transitions add further costs. No latency or energy advantage follows
from this accounting. Full-transformer radix/INT4 bundles have different
conversion coverage and quality behavior and are reported in the
[separate storage comparison](compare-shiftadd-transformer-storage.md).

## Subsequent saved package

The estimate above is now complemented by a separate
[complete-package record](m5-flux-klein-shiftadd-020.md), which measures actual
saved files including the BF16 remainder and metadata and verifies source-independent
loading. The original estimate remains labelled as such.
