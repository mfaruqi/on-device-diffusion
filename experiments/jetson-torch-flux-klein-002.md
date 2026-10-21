---
type: experiment-record
id: jetson-torch-flux-klein-002
status: complete
device: jetson-orin-nano
engine: pytorch-diffusers
runs: [jetson-torch-flux-klein-002__baseline__20261014-174108, jetson-torch-flux-klein-002__attempt__20261014-173551]
updated: 2026-10-14
---

# jetson-torch-flux-klein-002: SDPA restricted to the math backend, PyTorch/diffusers klein Q4_0 + NF4, sequential residency, 512×512

Status: **complete** (2026-10-14, run `__baseline__20261014-174108`; no Slurm job, run directly on the
device). Smoke gate `__attempt__20261014-173551` complete.

## Question

What do latency and memory become when every scaled-dot-product-attention call is forced onto
PyTorch's math (non-fused) SDPA backend, with everything else as in
[jetson-torch-flux-klein-001](jetson-torch-flux-klein-001.md)? Reference: the same-session 001 baseline
[`__baseline__20261014-171410`](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/).
It isolates the attention kernel's share of per-step time in this compressed-checkpoint workload (RQ1).

## Setup

- [Configuration](../configs/jetson-torch-flux-klein-q4-nf4-512-sdpa-math.json) and
  [smoke configuration](../configs/jetson-torch-flux-klein-q4-nf4-512-sdpa-math-smoke.json); the
  run-directory `config.json` copies match these files. Reference config:
  [jetson-torch-flux-klein-q4-nf4-512.json](../configs/jetson-torch-flux-klein-q4-nf4-512.json).
- **One change vs the reference:** `optimizations.attention_backend: "math"`. The config otherwise
  differs only in `id`, `description` and `reference_config`.
- Workload, compressed checkpoints and execution policy unchanged from 001: distilled klein
  `e7b7dc27f91deacad38e78976d1f2b499d76a294`, batch 1, 512×512, 4 steps, guidance 1.0,
  `max_sequence_length` 512, fixed prompt, seed 0; GGUF Q4_0 transformer (BF16 compute), pre-quantized
  bitsandbytes NF4 text encoder, BF16 VAE; sequential stage residency (`text_encoder:
  release_after_encode`, `transformer: release_during_encode`), so encoder load and transformer reload
  are inside `wall_ms` and `other_ms`. No offload, VAE tiling, compile or compiler.
- How the change is applied ([scripts/run_flux_portable.py](../scripts/run_flux_portable.py),
  `restrict_attention`): each generation runs inside `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])`,
  and `torch.nn.functional.scaled_dot_product_attention` is wrapped to count calls per generation. The
  runner fails the run if any generation makes zero SDPA calls; the counts are written to
  [attention.json](../results/runs/jetson-torch-flux-klein-002__baseline__20261014-174108/attention.json).
  The restriction covers the whole generation (text encoder, transformer and VAE). The counting
  wrapper is present only in this variant; the 001 reference calls SDPA unwrapped and unrestricted.
- Protocol: 1 first + 3 warm-up (discarded) + 10 measured; smoke: 1 first + 1 measured.
  Tegrastats every 200 ms.
- Runner sha256 `7d8b7d39…` (`source-provenance.json`), same as the 001 reference run. Launched under
  [scripts/jetson_memguard.sh](../scripts/jetson_memguard.sh) with `MIN_AVAIL_MB=400` and
  `MAX_SWAP_GROWTH_MB=1536` (memguard.json); the guard is a launcher safety setting outside the measured config.
- Device and software identical to the 001 reference run (`environment.json` differs only in its
  timestamp): Jetson Orin Nano 8 GB, R36.4.4, 25W, CUDA 12.6, torch 2.8.0, diffusers 0.40.0,
  transformers 5.16.1, bitsandbytes 0.48.0.dev0+ff389db. NVML unavailable; whole-system RAM from tegrastats.
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md).

## Results

[Run](../results/runs/jetson-torch-flux-klein-002__baseline__20261014-174108/),
[summary.json](../results/runs/jetson-torch-flux-klein-002__baseline__20261014-174108/summary.json),
[runs.csv](../results/runs/jetson-torch-flux-klein-002__baseline__20261014-174108/runs.csv),
[stages.csv](../results/runs/jetson-torch-flux-klein-002__baseline__20261014-174108/stages.csv),
[stage-residency.json](../results/runs/jetson-torch-flux-klein-002__baseline__20261014-174108/stage-residency.json),
[memguard.json](../results/runs/jetson-torch-flux-klein-002__baseline__20261014-174108/memguard.json).
Started 2026-10-14T21:41:08Z, finished 22:02:48Z; `status.json` `complete`; guard not triggered (exit 0).

### Latency

| Metric | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall_ms | 84221.5 | 82457.6–87453.3 | – |
| gpu_span_ms | 84220.6 | 82456.4–87452.1 | – |
| text_encode_ms | 1720.4 | 1544.0–2187.1 | 2.0% |
| denoise_ms | 19297.6 | 19213.3–19704.5 | 22.9% |
| denoise_step_0_ms | 5210.9 | 5169.8–5636.4 | – |
| denoise_step_1_ms | 4696.9 | 4678.8–4714.2 | – |
| denoise_step_2_ms | 4695.9 | 4682.4–4713.4 | – |
| denoise_step_3_ms | 4693.4 | 4680.2–4712.3 | – |
| Steady per-step (mean of steps 1–3 per generation) | 4694.6 | 4681.2–4713.3 | – |
| vae_decode_ms | 511.2 | 509.5–1409.4 | 0.6% |
| postprocess_ms | 50.2 | 42.5–96.7 | 0.1% |
| other_ms (includes encoder load and transformer reload) | 62535.7 | 61010.6–64578.7 | 74.3% |
| First generation: wall_ms | 98363.1 | – | – |
| First generation: text_encode / denoise (step 0) / vae_decode / postprocess / other | 5641.0 / 20491.5 (6370.4) / 3780.3 / 120.8 / 68305.5 | – | – |
| Warm-ups (3, discarded): wall_ms | – | 83491.7–87965.2 | – |
| Cold load: load_total_s | 75.37 s (`from_pretrained` 74.85 s, `.to` 0.51 s); cached, not a cold read | – | – |

The steady per-step row is computed from `runs.csv`, not a `summary.json` field. Per-run
`other_ms / wall_ms` ranged 0.732–0.748.

Residency costs, measured generations ([stage-residency.json](../results/runs/jetson-torch-flux-klein-002__baseline__20261014-174108/stage-residency.json)):
transformer free 0.498 s (0.490–0.535), text encoder load 33.264 s (32.541–34.405), text encoder free
0.454 s (0.450–0.459), transformer reload 28.367 s (27.212–30.068); median sum 62.503 s.
`other_ms` minus the residency sum is 30–36 ms in eight measured generations and 121 ms and 206 ms in
indices 9 and 10.

### Attention evidence

[attention.json](../results/runs/jetson-torch-flux-klein-002__baseline__20261014-174108/attention.json):
`attention_backend: "math"`, **137 SDPA calls in each of the 14 generations** (first, 3 warm-ups,
10 measured); the smoke run also recorded 137 in each of its 2 generations
([attention.json](../results/runs/jetson-torch-flux-klein-002__attempt__20261014-173551/attention.json)).
The file does not split calls by module. The transformer has 5 double-stream and 20 single-stream
blocks (block names in load.json), which would give 100 calls over 4 steps if each block makes one
call; the remaining 37 would then come from the text encoder and the VAE (inference, not recorded).

### Memory

| Quantity | Value | Source |
|---|---:|---|
| Weights: transformer / VAE | 2.291 / 0.157 GiB | load.json |
| Allocated / reserved after load | 2.475 / 2.523 GiB | load.json |
| peak_alloc_gib, measured | 3.152 GiB (3.151–3.152) | summary.json |
| peak_reserved_gib, measured (all 10 equal) | 3.512 GiB | summary.json |
| First generation peak alloc / reserved | 3.151 / 3.512 GiB | summary.json |
| Stage setting the allocator peak | denoise steps, 3.151–3.152 GiB (0.665–0.666 GiB above the 2.486 GiB at step start); `vae_decode` 3.081–3.083 GiB; `text_encode` 3.065–3.066 GiB | stages.csv |
| host_rss_gib, measured | median 1.594 (1.275–1.671) GiB | summary.json |
| host_peak_rss_gib (includes load) | 5.543 GiB | summary.json |
| System RAM peak, whole capture (tegrastats, 6127 samples) | 6.113 GiB of 7.441 GiB | summary.json `system_memory` |
| Swap occupancy, whole capture (tegrastats) | 0.301 → 1.218 GiB | summary.json `system_memory` |
| Guard: swap at start → maximum | 308 → 1247 MB (growth 939 MB) | memguard.json |
| Guard: MemAvailable at start → minimum | 5997 → 1102 MB | memguard.json |

Allocated and reserved are not added; host RSS and GPU allocations share one physical pool and are not
added. Maximum sampled GPU temperature 56.3 °C.

### Repeatability

All 14 generations have the same `image_sha256`
(`4109992326cba4d812e27cd5fc8c94ac59c9db2d85b5f976c80c61bbbff89dc8`; `deterministic_output: true`), as do
both smoke generations. This hash differs from the 001 reference output (`fe94904f…`): the math backend
changes the numerical result. The images were not visually inspected for this record and no quality
metric was computed. Measured generation index 9 is the slowest in wall time (87453.3 ms) and has the
maximum text encode (2187.1 ms), step 0 (5636.4 ms) and VAE decode (1409.4 ms); the other nine measured
VAE decodes are 509.5–518.2 ms and step 0 values 5169.8–5235.9 ms.

### Smoke gate

[`__attempt__20261014-173551`](../results/runs/jetson-torch-flux-klein-002__attempt__20261014-173551/):
first / measured wall 94750.6 / 87815.2 ms; denoise 20073.6 / 19274.7 ms (step 0 5944.4 / 5185.7 ms);
VAE decode 3586.3 / 512.8 ms; peak alloc 3.152 GiB, reserved 3.531 GiB; load 71.79 s; guard minimum
MemAvailable 1061 MB, swap 306 → 611 MB. Not a baseline.

### Delta vs reference (001, `__baseline__20261014-171410`, medians of 10 measured)

| Metric | 001 | 002 (math) | Δ | Δ % |
|---|---:|---:|---:|---:|
| wall_ms | 80314.9 | 84221.5 | +3906.6 | +4.9% |
| text_encode_ms | 1241.3 | 1720.4 | +479.1 | +38.6% |
| denoise_ms | 14795.5 | 19297.6 | +4502.1 | +30.4% |
| denoise_step_0_ms | 4015.3 | 5210.9 | +1195.6 | +29.8% |
| Steady per-step (mean of steps 1–3) | 3590.8 | 4694.6 | +1103.9 | +30.7% |
| vae_decode_ms | 491.4 | 511.2 | +19.8 | +4.0% |
| postprocess_ms | 47.8 | 50.2 | +2.4 | +5.0% |
| other_ms | 63343.9 | 62535.7 | −808.3 | −1.3% |
| peak_alloc_gib | 3.081 | 3.152 | +0.071 | +2.3% |
| peak_reserved_gib | 3.422 | 3.512 | +0.090 | +2.6% |
| host_rss_gib | 1.277 | 1.594 | +0.317 | +24.8% |

Residency medians in the reference: text encoder load 33.411 s, transformer reload 29.526 s.

## Observations

- Forcing the math SDPA backend lengthens each steady denoise step by 1.10 s (3.59 → 4.69 s, +30.7%)
  and step 0 by 1.20 s; denoise total rises 4.50 s.
- Text encoding rises 0.48 s (+38.6%); VAE decode rises 20 ms (+4.0%).
- Wall time rises 3.91 s (+4.9%), less than the 5.00 s sum of the compute-stage increases, because
  `other_ms` is 0.81 s lower (transformer reload median 28.37 s vs 29.53 s in the reference).
- The allocator peak moves from VAE decode to the denoise steps: each step allocates 0.665 GiB above
  its starting level, versus 0.422 GiB in the reference; the overall peak rises 0.071 GiB.
- Every generation made 137 SDPA calls under the restriction.
- Output is bit-identical within the run but differs from the reference output.

## Interpretation

- In this configuration, attention running through the math path costs about 1.1 s per denoise step,
  so the fused SDPA kernel that PyTorch selects by default accounts for at least that much per-step
  saving relative to math. Which fused kernel the reference used is not recorded.
- Hypothesis: the larger denoise transient (+0.243 GiB per step) is the materialized attention-score
  matrix of the math path; the data do not show its shape.
- Because per-generation weight reloading dominates wall time (≈ 62.5 s of ≈ 84 s), a 30% per-step
  slowdown appears as only +4.9% end to end under this residency policy.
- The changed image hash means this variant is not a bit-exact execution change; any quality
  statement needs an image comparison.

## Next experiment

- New config: `attention_backend: "efficient"` (and `"flash"` if supported on this device) on `jetson-torch-flux-klein-q4-nf4-512`, to identify which fused kernel the default selects.
- Saved-image diagnostics (PSNR/LPIPS) of the 002 output against the 001 output.
