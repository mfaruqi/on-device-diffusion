---
type: experiment-record
id: jetson-torch-flux-klein-base-004
status: complete
device: jetson-orin-nano
engine: pytorch-diffusers
runs: [jetson-torch-flux-klein-base-004__baseline__20261016-033850, jetson-torch-flux-klein-base-004__attempt__20261016-032541, jetson-torch-flux-klein-base-004__attempt__20261015-165259]
updated: 2026-10-16
---

# jetson-torch-flux-klein-base-004: torch.compile on top of exact conditioning reuse, PyTorch/diffusers klein Base Q4_0 + NF4 512²; 183.9 s per image (−48.8% vs reuse), steady step 3.67 s

Status: **complete** (2026-10-16; full run 07:38:50–08:27:58 UTC, smoke 07:25:41–07:38:05 UTC; run directly on the device). Full 1+3+10 protocol; no memory-guard intervention.

## Question

What does `torch.compile` (Inductor, default mode) change in per-image time, first-generation cost, memory and output when applied to the transformer of [jetson-torch-flux-klein-base-002](jetson-torch-flux-klein-base-002.md) (exact conditioning reuse)? Single change against [jetson-torch-flux-klein-base-q4-nf4-512-condreuse.json](../configs/jetson-torch-flux-klein-base-q4-nf4-512-condreuse.json). Compiler lever (RQ1/RQ2).

## Setup

- **Config**: [jetson-torch-flux-klein-base-q4-nf4-512-condreuse-compile.json](../configs/jetson-torch-flux-klein-base-q4-nf4-512-condreuse-compile.json); smoke [jetson-torch-flux-klein-base-q4-nf4-512-condreuse-compile-smoke.json](../configs/jetson-torch-flux-klein-base-q4-nf4-512-condreuse-compile-smoke.json).
- **Change vs reference (only one)**: `optimizations.compiler: {"name": "torch_compile", "mode": "default"}`, applied to the transformer at load and re-applied after each residency reload (`on_reload`). With reuse, reloads happen only during the first generation's two misses, so the transformer is compiled three times in all ([compiler.json](../results/runs/jetson-torch-flux-klein-base-004__baseline__20261016-033850/compiler.json): `compilations_applied` 3), all inside the first generation. `TORCHINDUCTOR_COMPILE_THREADS=1` set by the queue; triton 3.4.0.
- Everything else as in [base-002](jetson-torch-flux-klein-base-002.md#setup). Runner sha256 `4a254728…`.
- **Earlier attempt** [jetson-torch-flux-klein-base-004__attempt__20261015-165259](../results/runs/jetson-torch-flux-klein-base-004__attempt__20261015-165259/) (2026-10-15): `failed`, `IncompleteSnapshotError` (allow_patterns, as in base-001). Not a measurement.

## Results

Full run [jetson-torch-flux-klein-base-004__baseline__20261016-033850](../results/runs/jetson-torch-flux-klein-base-004__baseline__20261016-033850/): `complete`; guard not killed. Conditioning cache hits 26, misses 2; 2 swaps (68.8 and 69.5 s) in the first generation.

### Latency (measured n = 10; [summary.json](../results/runs/jetson-torch-flux-klein-base-004__baseline__20261016-033850/summary.json), [runs.csv](../results/runs/jetson-torch-flux-klein-base-004__baseline__20261016-033850/runs.csv))

| Stage | Median (ms) | Min–max (ms) | Share | base-002 median (ms) | Δ |
|---|---:|---:|---:|---:|---:|
| wall | **183926.8** | 183875.5–183977.3 (stdev 30.6) | 100% | 359440.7 | −175514.0 (−48.8%) |
| text_encode (2 cache hits) | 1.41 | 1.06–1.56 | 0.0% | 1.34 | +0.07 |
| denoise | 183379.9 | 183337.8–183428.6 | 99.7% | 358891.9 | −175511.9 (−48.9%) |
| vae_decode | 486.0 | 485.4–486.8 | 0.3% | 484.2 | +1.7 |
| other | 38.0 | 32.7–42.1 | 0.0% | 37.7 | +0.4 |
| First generation: wall | 449911.8 | text 145747.7 / denoise 300083.4 (step 0: 120805.7, includes compilation) / vae 3085.7 / other 813.8 | – | 510586.6 | −60674.8 (−11.9%) |
| Pipeline load (s) | 82.24 | – | – | 79.99 | – |

Steady step (mean of steps 1–49): 3667.6 ms (base-002: 7177.4 ms, −48.9%). Warm-ups 183838.8–183863.0 ms. Against [base-001](jetson-torch-flux-klein-base-001.md) (494634.4 ms) the wall is −62.8%.

### Memory

| Quantity | 004 | base-002 |
|---|---:|---:|
| Peak allocated / reserved (median) | 3.09 / 3.74 GiB | 3.09 / 3.44 GiB |
| RAM peak (tegrastats) | 6.385 GiB | 5.835 GiB |
| Swap (tegrastats) min–max | 0.313–0.809 GiB | 0.314–1.001 GiB |
| MemAvailable (guard) start → min | 5965 → 832 MB | 5981 → 1472 MB |
| GPU temperature max | 69.7 °C | 66.7 °C |

### Smoke run

[jetson-torch-flux-klein-base-004__attempt__20261016-032541](../results/runs/jetson-torch-flux-klein-base-004__attempt__20261016-032541/): complete. First 450759.7 ms (step 0: 121708.9 ms); one measured 184075.1 ms. Not a baseline.

### Repeatability and functional check (visual, not a quality metric)

All 14 full-run generations and both smoke generations: RGB SHA256 `f61087dd8e6da613…`, so the compiled output reproduces across two processes. It differs from the eager output of base-002 (`2a0803ab…`): mean absolute RGB difference 2.5 levels, 7.6% of pixels differ by more than 8 levels in any channel (first images). Visual: same cat, sign and lettering ("HELLO WORLD"); indistinguishable at a glance. No quality metric.

## Observations

- Compilation halved the transformer step (7177.4 → 3667.6 ms) and the measured wall (−48.8%); other stages are unchanged.
- The first generation includes about 121 s of compilation in step 0 and is still 60.7 s faster than base-002's, because its 49 remaining steps are compiled.
- Peak reserved +0.30 GiB, RAM peak +0.55 GiB and MemAvailable minimum −640 MB against base-002; peak allocated unchanged.
- Output deterministic across processes but not bit-identical to eager.

## Interpretation

- On this device the Q4_0 GGUF transformer in eager mode is dominated by overheads that Inductor removes (hypothesis: fusing the per-layer dequantize and the elementwise work around the matmuls); a kernel profile of one eager and one compiled step would test it.
- The compiled pipeline pays for itself within the first image at 50 steps; the cost is about 0.6 GiB less headroom, which matters on an 8 GB board.

## Next experiment

- Profile one eager and one compiled denoise step of `jetson-torch-flux-klein-base-q4-nf4-512-condreuse(-compile).json` with the torch profiler (kernel shares only).
