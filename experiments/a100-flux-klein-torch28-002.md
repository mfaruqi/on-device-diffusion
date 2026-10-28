---
type: experiment-record
id: a100-flux-klein-torch28-002
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-torch28-002__baseline__20261015-151840, a100-flux-klein-torch28-002__attempt__20261015-151820]
updated: 2026-10-15
---

# a100-flux-klein-torch28-002: exact conditioning reuse, BF16 1024² klein, torch 2.8.0, all weights resident

Status: **complete** (2026-10-15, job 11959806 on gilbreth-g008; full run 19:18:40–19:19:14 UTC, smoke
19:18:20–19:18:39 UTC).

## Question

How much generation latency is removed when the prompt encoding of a fixed workload is computed once and
reused, with all weights resident and everything else unchanged? Reference:
[a100-flux-klein-torch28-001](a100-flux-klein-torch28-001.md), baseline
[`__baseline__20261015-151548`](../results/runs/a100-flux-klein-torch28-001__baseline__20261015-151548/).
Execution-policy lever (RQ1).

## Setup

- Config [a100-flux-klein-torch28-bf16-condreuse.json](../configs/a100-flux-klein-torch28-bf16-condreuse.json);
  smoke [a100-flux-klein-torch28-bf16-condreuse-smoke.json](../configs/a100-flux-klein-torch28-bf16-condreuse-smoke.json).
  The run-directory `config.json` copies equal these files.
- **One change vs [a100-flux-klein-torch28-bf16.json](../configs/a100-flux-klein-torch28-bf16.json):**
  `optimizations.conditioning_reuse: true`; otherwise only `id`, `description` and `reference_config` differ.
- How it is applied ([scripts/run_flux_portable.py](../scripts/run_flux_portable.py) `benchmark()`,
  [scripts/mps_conditioning.py](../scripts/mps_conditioning.py)): `ConditioningCache` wraps
  `pipe.encode_prompt`, keyed on the full bound argument list, capacity 1 (guidance 1.0, no negative prompt);
  the stage recorder wraps the cache. A hit returns the stored embeddings without running the text encoder, so
  `text_encode_ms` on a hit is retrieval time; the single miss (first generation) runs the encoder. This
  configuration has no stage residency, so no weights move on a miss or hit. The runner fails the run unless
  there is exactly 1 miss and 13 hits.
- Workload, protocol (1 + 3 + 10; smoke 1 + 1), device and environment as in
  [torch28-001 setup](a100-flux-klein-torch28-001.md#setup). Runner sha256 `44e44608…`, `mps_conditioning.py`
  `80a2bc49…` ([source-provenance.json](../results/runs/a100-flux-klein-torch28-002__baseline__20261015-151840/source-provenance.json)).
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md); steady per-step as in
  torch28-001.

## Results

[Run](../results/runs/a100-flux-klein-torch28-002__baseline__20261015-151840/),
[summary.json](../results/runs/a100-flux-klein-torch28-002__baseline__20261015-151840/summary.json),
[runs.csv](../results/runs/a100-flux-klein-torch28-002__baseline__20261015-151840/runs.csv),
[stages.csv](../results/runs/a100-flux-klein-torch28-002__baseline__20261015-151840/stages.csv),
[conditioning-cache.json](../results/runs/a100-flux-klein-torch28-002__baseline__20261015-151840/conditioning-cache.json)
(`exact_prompt_encoding`, hits 13, misses 1, entries 1, capacity 1). `status.json` `complete`.

### Latency

| Stage | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| text_encode_ms (cache retrieval) | 0.19 | 0.18–0.22 | 0.0% |
| denoise_ms | 1041.7 | 1036.8–1047.4 | 82.6% |
| denoise_step_0 / 1 / 2 / 3_ms | 255.3 / 260.2 / 259.8 / 265.8 | 253.5–269.4 (all four) | – |
| Steady per-step (mean of steps 1–3) | 262.2 | 260.9–263.6 | – |
| vae_decode_ms | 169.3 | 168.5–170.9 | 13.4% |
| postprocess_ms | 42.0 | 32.9–44.6 | 3.3% |
| other_ms | 8.7 | 7.5–9.5 | 0.7% |
| **wall_ms** | **1261.5** | 1252.3–1269.7 (stdev 5.9) | 100% |
| First generation (cache miss): wall_ms | 1848.7 | – | text 387.4 / denoise 1087.7 (step 0: 310.0) / VAE 306.9 / post 46.6 / other 20.1 |
| Warm-ups (3, discarded): wall_ms | – | 1245.1–1260.6 | – |
| Load (cached, s) | 8.80 | – | `from_pretrained` 2.75 + `.to("cuda")` 6.05 |

### Memory (GiB)

| Quantity | Value |
|---|---:|
| peak_alloc_gib (all measured equal) | 17.32, set by `vae_decode` |
| Hit `text_encode` stage: allocated at start / peak | 14.94 / 14.94 (no encoder activations) |
| peak_reserved_gib / device_used_peak_gib | 19.61 / 20.62 |
| host_rss_gib, measured median | 1.68 (1.68–1.69) |
| host_peak_rss_gib (includes load) | 8.06 |

The text encoder (7.49 GiB) stays resident; this configuration does not release it.

### Repeatability and functional check (visual, not a quality metric)

All 14 generations have `image_sha256` `f11220f630cce576da295977e5f575a1c1b391c8d117dc5db263cf5ab04624dd`
(`deterministic_output: true`), identical to every generation of torch28-001; the smoke run has the same hash.
The retained `images/measured-0.png` is pixel-identical to torch28-001's (RGB difference 0), so the torch28-001
visual check (cat holding a sign reading "hellword") applies unchanged.

### Smoke gate

[`__attempt__20261015-151820`](../results/runs/a100-flux-klein-torch28-002__attempt__20261015-151820/):
complete; hits 1, misses 1. First 1850.4 ms; one measured 1243.3 ms (text 0.20 / denoise 1032.9 / VAE 168.4 ms).
Load 8.76 s. Memory 17.32 / 19.61 / 20.62 GiB. Not a baseline (n = 1).

### Delta vs reference (torch28-001 baseline)

| Metric | torch28-001 median | torch28-002 median | Δ |
|---|---:|---:|---:|
| wall_ms | 1290.5 | 1261.5 | −29.0 (−2.2%) |
| text_encode_ms | 46.2 | 0.19 (retrieval) | −46.0 |
| denoise_ms | 1037.8 | 1041.7 | +4.0 (+0.4%) |
| Steady per-step | 260.3 | 262.2 | +1.9 (+0.7%) |
| vae_decode_ms | 168.2 | 169.3 | +1.1 |
| postprocess_ms | 29.8 | 42.0 | +12.2 |
| other_ms | 7.5 | 8.7 | +1.2 |
| First generation wall_ms | 1830.9 | 1848.7 | +17.8 (+1.0%) |
| peak_alloc / peak_reserved / device_used_peak (GiB) | 17.32 / 19.61 / 20.62 | 17.32 / 19.61 / 20.62 | 0 |

## Observations

- Text encoding (46.2 ms, 3.6% of the reference wall) is replaced by a 0.19 ms retrieval on every hit.
- The end-to-end median falls by 29.0 ms (−2.2%), less than the 46.0 ms removed from text encode: postprocess
  is 12.2 ms higher at the median and the other stages are within a few ms.
- Memory peaks are unchanged; VAE decode sets the allocator peak in every generation.
- Output is bit-identical to torch28-001 in all 14 generations.

## Interpretation

- With all weights resident, conditioning reuse removes only the text-encoder compute; on this configuration
  that is a small share of generation time, and denoising remains about 83% of wall. The saving holds only while
  the prompt and every encoding argument stay identical; a new prompt is a miss and pays the full encode.
- Reuse does not lower memory here because the encoder weights stay resident; a memory effect would require
  releasing the encoder after the miss (a separate stage-residency change).
- Hypothesis: the postprocess difference (host-bound GPU→CPU copy and PIL conversion) is run-to-run host
  variation, not an effect of the cache; the measured ranges overlap (27.1–34.6 vs 32.9–44.6 ms).

## Next experiment

- Copy of `a100-flux-klein-torch28-bf16-condreuse.json` adding text-encoder release after the miss (stage residency, separately labelled), to measure the memory side.
