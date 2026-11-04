---
type: experiment-record
id: a100-flux-klein-base-torch28-002
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-base-torch28-002__baseline__20261015-160117, a100-flux-klein-base-torch28-002__attempt__20261015-160007]
updated: 2026-10-15
---

# a100-flux-klein-base-torch28-002: exact conditioning reuse (positive + negative), BF16 1024² klein Base, torch 2.8.0, all weights resident

Status: **complete** (2026-10-15, job 11959986 on gilbreth-g009; full run 20:01:17–20:07:42 UTC, smoke
20:00:07–20:01:16 UTC).

## Question

How much generation latency is removed when both prompt encodings of the klein Base CFG workload (positive and empty
negative) are computed once and reused, with all weights resident and everything else unchanged? Reference:
[a100-flux-klein-base-torch28-001](a100-flux-klein-base-torch28-001.md), baseline
[`__baseline__20261015-155338`](../results/runs/a100-flux-klein-base-torch28-001__baseline__20261015-155338/).
Execution-policy lever (RQ1).

## Setup

- Config [a100-flux-klein-base-torch28-bf16-condreuse.json](../configs/a100-flux-klein-base-torch28-bf16-condreuse.json);
  smoke [a100-flux-klein-base-torch28-bf16-condreuse-smoke.json](../configs/a100-flux-klein-base-torch28-bf16-condreuse-smoke.json).
  The run-directory `config.json` copies equal these files.
- **One change vs [a100-flux-klein-base-torch28-bf16.json](../configs/a100-flux-klein-base-torch28-bf16.json):**
  `optimizations.conditioning_reuse: true`; otherwise only `id`, `description` and `reference_config` differ.
- How it is applied ([scripts/run_flux_portable.py](../scripts/run_flux_portable.py),
  [scripts/mps_conditioning.py](../scripts/mps_conditioning.py);
  [definition](../wiki/methods/baseline-metrics.md#portable-pytorch-runner-options)): `ConditioningCache` wraps
  `pipe.encode_prompt`, keyed on the full bound argument list, capacity 2 (CFG: positive and empty negative prompt as
  separate entries); the stage recorder wraps the cache. On a hit `text_encode_ms` is retrieval time. The run fails
  unless misses = 2 and hits = (generations − 1) × 2. No stage residency is configured, so the text encoder stays
  resident.
- Workload, protocol (1 + 3 + 10; smoke 1 + 1), device and environment as in
  [base-torch28-001 setup](a100-flux-klein-base-torch28-001.md#setup). Runner sha256 `06f0a949…`,
  `mps_conditioning.py` `80a2bc49…` ([source-provenance.json](../results/runs/a100-flux-klein-base-torch28-002__baseline__20261015-160117/source-provenance.json)).
  Two transformer calls per step and two `text_encode` calls per generation (from `stages.csv`; the field
  `transformer_calls_per_step` is absent from these `summary.json` copies, see the reference record).
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md); steady per-step as in base-torch28-001.

## Results

[Run](../results/runs/a100-flux-klein-base-torch28-002__baseline__20261015-160117/),
[summary.json](../results/runs/a100-flux-klein-base-torch28-002__baseline__20261015-160117/summary.json),
[runs.csv](../results/runs/a100-flux-klein-base-torch28-002__baseline__20261015-160117/runs.csv),
[stages.csv](../results/runs/a100-flux-klein-base-torch28-002__baseline__20261015-160117/stages.csv),
[conditioning-cache.json](../results/runs/a100-flux-klein-base-torch28-002__baseline__20261015-160117/conditioning-cache.json)
(`exact_prompt_encoding`, hits 26, misses 2, entries 2, capacity 2). `status.json` `complete`.

### Latency

| Stage | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| text_encode_ms (2 cache retrievals) | 0.38 | 0.36–0.82 | 0.0% |
| denoise_ms | 26069.8 | 26043.8–26089.9 | 99.2% |
| Steady per-step (mean of steps 1–49) | 521.5 | 521.1–522.0 | – |
| vae_decode_ms | 169.7 | 169.2–170.5 | 0.6% |
| postprocess_ms | 37.8 | 31.9–43.5 | 0.1% |
| other_ms | 12.5 | 11.8–13.8 | 0.0% |
| **wall_ms** | **26290.4** | 26259.8–26318.0 (stdev 15.9) | 100% |
| First generation (2 misses): wall_ms | 26562.5 | – | text 411.7 / denoise 25782.6 / VAE 299.8 / post 43.9 / other 24.0 |
| Warm-ups (3, discarded): wall_ms | – | 26147.1–26241.5 | – |
| Load (cached, s) | 9.74 | – | `from_pretrained` 2.81 + `.to("cuda")` 6.93 |

### Memory (GiB)

| Quantity | Value |
|---|---:|
| peak_alloc_gib (all measured equal) | 17.33, set by `vae_decode` |
| Hit `text_encode` stage peak | 14.95 (no encoder activations; 15.19 in the reference) |
| peak_reserved_gib / device_used_peak_gib | 19.61 / 20.62 |
| host_rss_gib, measured median | 1.69 (1.69–1.71) |
| host_peak_rss_gib (includes load) | 8.06 |

The text encoder (7.49 GiB) stays resident; this configuration does not release it.

### Repeatability and functional check

All 14 generations have `image_sha256` `7357668ff45cc26fdb530a90375447ae761d0243a6972a3248e5c1a01387fa55`
(`deterministic_output: true`), identical to every generation of base-torch28-001; the smoke run has the same hash.
The retained `images/measured-0.png` is pixel-identical to base-torch28-001's (RGB difference 0), so the reference
visual check (sign "HELLLO WORLD") applies unchanged.

### Smoke gate

[`__attempt__20261015-160007`](../results/runs/a100-flux-klein-base-torch28-002__attempt__20261015-160007/):
complete; hits 2, misses 2. First 26665.1 ms; one measured 26148.5 ms (text 0.38 / denoise 25934.1 / VAE 170.1 ms).
Load 9.20 s. Memory 17.33 / 19.61 / 20.62 GiB. Not a baseline (n = 1).

### Delta vs reference (base-torch28-001 baseline)

| Metric | base-torch28-001 median | base-torch28-002 median | Δ |
|---|---:|---:|---:|
| wall_ms | 26476.6 | 26290.4 | −186.2 (−0.70%) |
| text_encode_ms | 93.0 | 0.38 (retrieval) | −92.6 |
| denoise_ms | 26171.4 | 26069.8 | −101.6 (−0.39%) |
| Steady per-step | 523.6 | 521.5 | −2.1 (−0.4%) |
| vae_decode_ms | 170.0 | 169.7 | −0.3 |
| postprocess_ms | 28.8 | 37.8 | +9.0 |
| other_ms | 12.9 | 12.5 | −0.4 |
| First generation wall_ms | 26518.4 | 26562.5 | +44.1 (+0.2%) |
| peak_alloc / peak_reserved / device_used_peak (GiB) | 17.33 / 19.61 / 20.62 | 17.33 / 19.61 / 20.62 | 0 |

## Observations

- Both text encodes (93.0 ms together, 0.35% of the reference wall) are replaced by a 0.38 ms retrieval on every hit;
  the cache receipt shows exactly one miss per prompt (2) and 26 hits.
- Wall falls by 186.2 ms (−0.70%): 92.6 ms from text encode and 101.6 ms from denoise, partly offset by a 9.0 ms
  higher postprocess median.
- The measured denoise ranges overlap only between 26076.2 and 26089.9 ms (26043.8–26089.9 here, 26076.2–26283.7 in
  the reference); the reference has the wider spread (stdev 78.0 vs 14.1 ms).
- Memory peaks are unchanged; VAE decode sets the allocator peak in every generation.
- Output is bit-identical to base-torch28-001 in all 14 generations.

## Interpretation

- With all weights resident, conditioning reuse removes only the text-encoder compute, which is a negligible share of a
  50-step CFG generation; transformer time is 99.2% of wall. The saving holds only while the prompt, the negative prompt
  and every encoding argument stay identical.
- Reuse does not lower memory here because the encoder weights stay resident; a memory effect would need releasing the
  encoder after the misses (a separate stage-residency change).
- Hypothesis: the 101.6 ms lower denoise median is run-to-run variation in the same job (the cache does not touch the
  transformer calls), not an effect of reuse; repeated paired runs would test it.

## Next experiment

- Copy of `a100-flux-klein-base-torch28-bf16-condreuse.json` adding text-encoder release after the misses (stage residency, separately labelled), to measure the memory side.
