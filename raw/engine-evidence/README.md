# Pinned engine source evidence

These source excerpts support the measurement definitions and experiment records:

- [Conditioning reuse](conditioning-source-evidence.txt): sd.cpp conditioning-cache behavior; [Jetson Base record](../../experiments/jetson-flux-klein-base-005.md).
- [edge-dit phase markers](edge-phase-source.txt): timer and stage boundaries; [A100 record](../../experiments/a100-edgedit-flux-klein-001.md).
- [Mapped loading](mmap-source-evidence.txt): sd.cpp mmap behavior; [metric definitions](../../wiki/methods/baseline-metrics.md#sdcpp-mapped-io-receipt).

Preserved byte-for-byte from the October campaign's local scratch directory. Run measurements,
validation and overlap reports remain in `results/runs/`; temporary launch and upload scripts
are local operational files, not the supported benchmark interface in `scripts/`.
