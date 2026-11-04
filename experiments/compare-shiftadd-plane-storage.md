---
type: experiment-record
id: compare-shiftadd-plane-storage
status: complete
device: m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-015__attempt__20261103-all-dual, m5-flux-klein-shiftadd-016__attempt__20261103-all-single, m5-flux-klein-shiftadd-020__attempt__20261103-package, m5-flux-klein-shiftadd-022__attempt__20261103-mixed-fused, m5-flux-klein-shiftadd-023__attempt__20261103-selected-p2, m5-flux-klein-shiftadd-024__attempt__20261103-selected-p1]
updated: 2026-11-04
---

# Compare: plane-allocation storage and reconstructed weight footprint

## Question

Week 5 precision variants, RQ1/RQ3. How do the recorded p4, p3p4, p2p4 and
p1p4 allocations change stored transformer weights, and what remains dense in
the current image runner?

## Runs and matching

- [m5-flux-klein-shiftadd-015__attempt__20261103-all-dual](../results/runs/m5-flux-klein-shiftadd-015__attempt__20261103-all-dual/).
- [m5-flux-klein-shiftadd-016__attempt__20261103-all-single](../results/runs/m5-flux-klein-shiftadd-016__attempt__20261103-all-single/).
- [m5-flux-klein-shiftadd-020__attempt__20261103-package](../results/runs/m5-flux-klein-shiftadd-020__attempt__20261103-package/).
- [m5-flux-klein-shiftadd-022__attempt__20261103-mixed-fused](../results/runs/m5-flux-klein-shiftadd-022__attempt__20261103-mixed-fused/).
- [m5-flux-klein-shiftadd-023__attempt__20261103-selected-p2](../results/runs/m5-flux-klein-shiftadd-023__attempt__20261103-selected-p2/).
- [m5-flux-klein-shiftadd-024__attempt__20261103-selected-p1](../results/runs/m5-flux-klein-shiftadd-024__attempt__20261103-selected-p1/).

All fit candidates share the pinned Klein checkpoint, group 128, u8/shared-FP16
scale format and selected fifty matrices. The mixed cases change only the plane
count of the same ten fused matrices (single-stream blocks 0–8 and 11).
The BF16 row is an uncompressed reference. The other 119 tensors remain BF16.
The chart compares logical file/payload bytes, not timing or peak memory.

## Results

| Allocation | Selected stored GiB | Transformer accounting GiB | Dense BF16 parameter GiB |
|---|---:|---:|---:|
| BF16 | 3.691406 | 7.218764 | 7.218764 |
| p4 | 0.999805 | 4.527163 | 7.218764 |
| p3p4 | 0.894748 | 4.422106 | 7.218764 |
| p2p4 | 0.789691 | 4.317049 | 7.218764 |
| p1p4 | 0.684635 | 4.211993 | 7.218764 |

[Machine-readable values and source hashes](../output/shiftadd-plane-storage-20261104/data.json),
[CSV](../output/shiftadd-plane-storage-20261104/sizes.csv),
[reproduction](../output/shiftadd-plane-storage-20261104/README.md).

Selected packed sizes include NPZ headers, row indices and scale metadata.
The BF16 counterpart is tensor payload without its container header.
Whole-transformer accounting adds the untouched BF16 payload to the selected
files; it excludes the remainder-file header and package manifest. Mixed-case
whole-transformer values are estimates of that layout, not measurements of new
standalone packages. The actual saved all-four-plane package is 4,861,217,420
bytes including its metadata, measured separately in
[package 020](m5-flux-klein-shiftadd-020.md).

## Interpretation and limits

Stored size decreases with fewer planes. The current
[image runner](../shiftadd/src/shiftadd_flux/calibration_smoke.py) loads the BF16
pipeline on CUDA, reconstructs weights in CPU chunks and copies them into existing
BF16 parameters before each candidate's images. Therefore all allocations retain
the same calculated transformer parameter payload: 7,751,089,152 bytes.
This is not a measurement of peak allocated/reserved/device-wide memory.
Text encoder, VAE, activations, workspaces and temporary loading buffers are excluded.

Lower-plane files do not by themselves establish useful quality or speed.
The [image diagnostic](a100-flux-klein-upstream-lowplane-fused-001.md) records
recognizable two-plane scenes and severe one-plane semantic failures for this
fixed selection. No matched-quality claim or packed execution advantage is made.

## Published charts

[W&B charts](https://forge.coreweave.com/wandb/mfaruqi-purdue-university/on-device-diffusion/runs/shiftadd-plane-storage-charts-20261104) contain separate stored-size and calculated dense-weight
panels, the exact-byte table and a downloadable source artifact. Run completion,
labels, metrics and both panels were verified; downloaded source files and charts
match local SHA256 hashes. [Receipt](../output/shiftadd-plane-storage-20261104/wandb-verification.json).
