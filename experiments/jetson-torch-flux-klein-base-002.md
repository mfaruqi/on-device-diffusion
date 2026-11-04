---
type: experiment-record
id: jetson-torch-flux-klein-base-002
status: complete
device: jetson-orin-nano
engine: pytorch-diffusers
runs: [jetson-torch-flux-klein-base-002__baseline__20261016-015207, jetson-torch-flux-klein-base-002__attempt__20261016-013540, jetson-torch-flux-klein-base-002__attempt__20261015-165125]
updated: 2026-10-16
---

# jetson-torch-flux-klein-base-002: exact conditioning reuse (prompt and negative prompt), PyTorch/diffusers klein Base Q4_0 + NF4 512², sequential residency; 359.4 s per image (−27.3%), output bit-identical

Status: **complete** (2026-10-16; full run 05:52:07–07:20:15 UTC, smoke 05:35:40–05:51:28 UTC; run directly on the device). Full 1+3+10 protocol; no memory-guard intervention.

## Question

How much of the per-image time of [jetson-torch-flux-klein-base-001](jetson-torch-flux-klein-base-001.md) (two residency swaps per CFG generation) disappears when the prompt and negative-prompt encodings of a fixed workload are computed once and reused? Single change against [jetson-torch-flux-klein-base-q4-nf4-512.json](../configs/jetson-torch-flux-klein-base-q4-nf4-512.json). Execution-policy lever under a memory budget (RQ2).

## Setup

- **Config**: [jetson-torch-flux-klein-base-q4-nf4-512-condreuse.json](../configs/jetson-torch-flux-klein-base-q4-nf4-512-condreuse.json); smoke [jetson-torch-flux-klein-base-q4-nf4-512-condreuse-smoke.json](../configs/jetson-torch-flux-klein-base-q4-nf4-512-condreuse-smoke.json).
- **Change vs reference (only one)**: `optimizations.conditioning_reuse: true`. `ConditioningCache` ([mps_conditioning.py](../scripts/mps_conditioning.py)) wraps `encode_prompt` outside the residency swaps and inside the stage recorder, keyed on the full bound argument list, capacity 2 (prompt and CFG negative prompt kept separately). The first generation misses twice and performs both swaps inside its timed `text_encode`; every later call is a hit that skips the swaps, so the transformer loaded at the first generation stays resident.
- Everything else as in the [base-001 setup](jetson-torch-flux-klein-base-001.md#setup). Runner sha256 `4a254728…` (before the swap-count check was corrected; this config takes the conditioning branch of that check, which was already correct).
- **Earlier attempt** [jetson-torch-flux-klein-base-002__attempt__20261015-165125](../results/runs/jetson-torch-flux-klein-base-002__attempt__20261015-165125/) (2026-10-15): `failed`, `IncompleteSnapshotError` (allow_patterns, as in base-001). Not a measurement.

## Results

Full run [jetson-torch-flux-klein-base-002__baseline__20261016-015207](../results/runs/jetson-torch-flux-klein-base-002__baseline__20261016-015207/): `complete`; guard not killed. [conditioning-cache.json](../results/runs/jetson-torch-flux-klein-base-002__baseline__20261016-015207/conditioning-cache.json): hits 26, misses 2, capacity 2; [stage-residency.json](../results/runs/jetson-torch-flux-klein-base-002__baseline__20261016-015207/stage-residency.json): 2 swaps (68.5 and 69.3 s), both in the first generation.

### Latency (measured n = 10; [summary.json](../results/runs/jetson-torch-flux-klein-base-002__baseline__20261016-015207/summary.json), [runs.csv](../results/runs/jetson-torch-flux-klein-base-002__baseline__20261016-015207/runs.csv))

| Stage | Median (ms) | Min–max (ms) | Share | base-001 median (ms) | Δ |
|---|---:|---:|---:|---:|---:|
| wall | **359440.7** | 359333.5–359562.1 (stdev 73.8) | 100% | 494634.4 | −135193.7 (−27.3%) |
| text_encode (2 cache hits) | 1.34 | 0.96–1.53 | 0.0% | 2742.1 | −2740.8 |
| denoise | 358891.9 | 358790.7–359017.8 | 99.8% | 360239.1 | −1347.2 (−0.4%) |
| vae_decode | 484.2 | 483.6–485.4 | 0.1% | 1174.0 | −689.8 |
| other | 37.7 | 29.9–41.0 | 0.0% | 130418.8 | −130381.1 |
| First generation: wall | 510586.6 | text 145493.3 (2 misses, both swaps) / denoise 360318.1 / vae 3846.8 / other 826.7 | – | 504094.0 | +6492.6 (+1.3%) |
| Pipeline load (s) | 79.99 | – | – | 76.03 | – |

Steady step (mean of steps 1–49): 7177.4 ms (base-001: 7193.5 ms). Warm-ups 359374.7–359403.9 ms.

### Memory

| Quantity | 002 | base-001 |
|---|---:|---:|
| Peak allocated / reserved (median) | 3.09 / 3.44 GiB | 3.09 / 3.44 GiB |
| RAM peak (tegrastats) | 5.835 GiB | 5.966 GiB |
| Swap (tegrastats) min–max | 0.314–1.001 GiB | 0.312–1.204 GiB |
| MemAvailable (guard) start → min | 5981 → 1472 MB | 5992 → 1350 MB |

### Smoke run

[jetson-torch-flux-klein-base-002__attempt__20261016-013540](../results/runs/jetson-torch-flux-klein-base-002__attempt__20261016-013540/): complete; hits 2, misses 2. First 492325.2 ms (text 126735.7); one measured 359142.1 ms. Not a baseline.

### Repeatability and functional check

All 14 generations: RGB SHA256 `2a0803ab87940098…`, identical to every base-001 generation; `deterministic_output` true. Visual: same "HELLO WORLD" image as base-001.

## Observations

- Reuse removed both swaps and the encoder work from every generation after the first: wall −135.2 s (−27.3%), almost all from `other` (−130.4 s) and `text_encode` (−2.7 s).
- Denoise and the steady step are within 0.4% of base-001; the output is bit-identical.
- The first generation is 6.5 s slower than base-001's because both swaps are now timed inside it.

## Interpretation

- For a fixed prompt, exact conditioning reuse turns this residency policy into "transformer resident after the first image" at no change in output; what remains is the transformer itself (99.8% of wall), so further gains must come from cheaper or fewer transformer calls.

## Next experiment

- `torch.compile` on top of this configuration: [jetson-torch-flux-klein-base-004](jetson-torch-flux-klein-base-004.md).
