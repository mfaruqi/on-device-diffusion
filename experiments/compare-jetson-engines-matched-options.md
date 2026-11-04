---
type: experiment-record
id: compare-jetson-engines-matched-options
status: complete
device: jetson-orin-nano
engine: [stable-diffusion-cpp, edge-dit-cpp, pytorch-diffusers]
runs: [jetson-flux-klein-003__repeat__20261014-163411, jetson-flux-klein-010__baseline__20261014-183501, jetson-flux-klein-011__baseline__20261014-185639, jetson-edgedit-flux-klein-006__baseline__20261014-170149, jetson-edgedit-flux-klein-007__attempt__20261014-170729, jetson-edgedit-flux-klein-007__baseline__20261014-171051, jetson-torch-flux-klein-001__baseline__20261014-171410, jetson-torch-flux-klein-002__baseline__20261014-174108, jetson-torch-flux-klein-003__baseline__20261014-180843]
updated: 2026-10-14
---

# Compare: Jetson distilled klein 512² across sd.cpp, edge-dit and PyTorch, with matched option toggles

## Question

On the Jetson Orin Nano, how do stable-diffusion.cpp, edge-dit.cpp and PyTorch/diffusers differ for
the distilled klein 512² workload, stage by stage, and do two execution options (fused/flash
attention off; VAE tiling with 128 px tiles and 0.25 overlap) have the same effect in every engine?
Week 2 engine coverage (rolled over), RQ1. All runs were measured on 2026-10-14 in one session.

## Runs

| Run directory | Engine | Config | Record | What differs |
|---|---|---|---|---|
| [003 repeat](../results/runs/jetson-flux-klein-003__repeat__20261014-163411/) | sd.cpp 19bbbca | [disk-baseline](../configs/jetson-flux-klein-q4-512-disk-baseline.json) | [003](jetson-flux-klein-003.md) | sd.cpp reference: diffusion flash attention on, untiled VAE |
| [010](../results/runs/jetson-flux-klein-010__baseline__20261014-183501/) | sd.cpp | [disk-noflash](../configs/jetson-flux-klein-q4-512-disk-noflash.json) | [010](jetson-flux-klein-010.md) | diffusion flash attention off |
| [011](../results/runs/jetson-flux-klein-011__baseline__20261014-185639/) | sd.cpp | [disk-vaetile](../configs/jetson-flux-klein-q4-512-disk-vaetile.json) | [011](jetson-flux-klein-011.md) | VAE tiling 128 px / 0.25 |
| [006](../results/runs/jetson-edgedit-flux-klein-006__baseline__20261014-170149/) | edge-dit 97cc585 | [embq8-vaetile](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json) | [006](jetson-edgedit-flux-klein-006.md) | edge-dit reference: flash attention on, VAE tiling on (required to fit, [004](jetson-edgedit-flux-klein-004.md)) |
| [007 smoke](../results/runs/jetson-edgedit-flux-klein-007__attempt__20261014-170729/), [007 full](../results/runs/jetson-edgedit-flux-klein-007__baseline__20261014-171051/) | edge-dit | [embq8-vaetile-noflash](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-noflash.json) | [007](jetson-edgedit-flux-klein-007.md) | flash attention off; full run stopped by the memory guard after 2 generations |
| [001](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/) | PyTorch 2.8.0 / diffusers 0.40.0 | [q4-nf4-512](../configs/jetson-torch-flux-klein-q4-nf4-512.json) | [001](jetson-torch-flux-klein-001.md) | PyTorch reference: SDPA backend chosen by PyTorch, untiled VAE |
| [002](../results/runs/jetson-torch-flux-klein-002__baseline__20261014-174108/) | PyTorch | [sdpa-math](../configs/jetson-torch-flux-klein-q4-nf4-512-sdpa-math.json) | [002](jetson-torch-flux-klein-002.md) | SDPA restricted to the math backend |
| [003](../results/runs/jetson-torch-flux-klein-003__baseline__20261014-180843/) | PyTorch | [vaetile](../configs/jetson-torch-flux-klein-q4-nf4-512-vaetile.json) | [003](jetson-torch-flux-klein-003.md) | VAE tiling 128 px / 0.25 |

Within each engine, every variant differs from that engine's reference by exactly one option.
Across engines, the references differ in more than one way (below).

## Matching

| Setting | sd.cpp | edge-dit | PyTorch |
|---|---|---|---|
| Transformer | GGUF Q4_0 `d1023499…` | same file | same file (diffusers GGUF loader, BF16 compute) |
| Text encoder (Qwen3-4B) | GGUF Q4_K_M `f6f85177…` | Q4_K_M with token embedding Q6_K→Q8_0 `d83c059b…` | bitsandbytes NF4 `eaa2048d…` |
| VAE | BF16 | BF16 | BF16 |
| Resolution / steps / guidance / CFG / batch | 512² / 4 / 1 / 1 / 1 | same | same |
| Schedule | Flux2, image_seq_len 1024, mu 2.031 | Flux2, mu 2.031 | FlowMatchEuler, dynamic shift ([scheduler.json](../results/runs/jetson-torch-flux-klein-001__baseline__20261014-171410/scheduler.json)) |
| Prompt / seed | development prompt / 0 | same | same |
| Initial latents | engine RNG | engine RNG | torch RNG — not shared |
| Weight residency | disk-backed parameters, reloaded every generation (inside text_encode and denoise step 0) | all resident, one load | sequential: encoder and transformer reloaded every generation (inside other_ms) |
| VAE tiling when on | 128 px, overlap 0.25: 5×5 = 25 tiles (engine log) | 8×8 latent = 128 px, overlap 0.25: 25 tiles (engine log) | 128 px, overlap 0.25: 6×6 = 36 tiles (derived from the diffusers `tiled_decode` loop; not logged) |
| Attention "off" | `--diffusion-fa 0` (log confirms) | `--no-flash-attention` (flag only; not logged) | `sdpa_kernel([MATH])`, 137 SDPA calls/generation counted |
| Protocol | 1 + 3 + 10 | 1 + 3 + 10 (one process, 14 passes) | 1 + 3 + 10 |
| Stage timing | engine callbacks (`results.jsonl`) | engine breakdown line, 10 ms rounding | CUDA events |
| System memory | tegrastats 1000 ms | tegrastats 200 ms | tegrastats 200 ms |

**Confounds across engines:** text-encoder checkpoint, weight-residency policy, initial latents,
stage-boundary definitions, VAE tile count in PyTorch, and the memory sampling interval for sd.cpp.
Cross-engine statements below are therefore **qualitative** except where a stage is unaffected by
the confound (noted per row). Within-engine option effects are matched comparisons.

## Results

All numbers are medians of the 10 measured generations (seconds) from each run's `summary.json`,
except the derived per-step values (below the table) and the 007 values (single observations).

### Engine references

| Stage | sd.cpp (003 repeat) | edge-dit (006) | PyTorch (001) |
|---|---:|---:|---:|
| Wall per generation | 60.64 (55.10–67.04) | 11.96 (11.94–12.03) | 80.31 (78.17–82.60) |
| Text encode | 22.56 (includes encoder reload) | 0.93 | 1.24 (reload excluded) |
| Denoise, 4 steps | 36.01 (step 0 includes transformer reload) | 9.19 | 14.80 |
| VAE decode | 4.64 (untiled) | 1.74 (tiled, 25 tiles) | 0.49 (untiled) |
| Other | 0.28 | 0.12 | 63.34 (encoder load + transformer reload) |
| Steady denoise step (derived) | 2.713 (steps 2–4) | ≤ 2.296 (mean of all 4; no per-step timing) | 3.591 (steps 1–3) |
| First generation | 26.74 | 17.39 | 94.59 |
| One-time load (s) | 66.3 | 64.7 | 82.3 |
| Whole-system RAM peak (GiB, whole capture) | 6.00 | 6.72 | 6.11 |
| Swap max (GiB) | 0.29 | 0.29 | 1.23 |

Steady-step derivation: sd.cpp from `engine/results.jsonl` step-end timestamps,
(t_step_end[3] − t_step_end[0]) / 3; PyTorch from `runs.csv`, mean of `denoise_step_1..3_ms`;
edge-dit has no per-step timestamps, so `denoise_ms / 4` is an upper bound that includes step 1.
The [003 profile](jetson-flux-klein-003.md) attributes sd.cpp's reload time to text encoding and
step 0 and shows later steps as almost continuously GPU-busy.

### Attention off

| Engine | Steady step on → off | Change | Wall on → off |
|---|---:|---:|---:|
| sd.cpp (010) | 2.713 → 3.535 | ×1.30 | 60.64 → 69.74 |
| PyTorch (002) | 3.591 → 4.695 | ×1.31 | 80.31 → 84.22 |
| edge-dit (007) | sampling 9.27 → 12.68 (second generation, single observations) | ×1.37 | not measured: full run stopped by the guard (MemAvailable 399 MB < 400 MB) during generation 3. Minimum MemAvailable: 007 smoke 615 MB, 007 full 399 MB; 006 full 584 MB, 006 smoke 697 MB |

### VAE tiling 128 px / 0.25

| Engine | VAE decode untiled → tiled | Change | Steady step |
|---|---:|---:|---|
| sd.cpp (011) | 4.64 → 5.86 | +1.22 s | unchanged (2.713 → 2.713) |
| PyTorch (003) | 0.49 → 1.85 | +1.36 s (×3.8) | unchanged (3.591 → 3.581) |
| edge-dit (006) | untiled not feasible ([004](jetson-edgedit-flux-klein-004.md)) → 1.74 | – | – |

PyTorch whole-system RAM peak falls from 6.11 to 5.66 GiB with tiling; sd.cpp's stays at
5.97–6.00 GiB (1 s sampling).

### Session drift in sd.cpp wall time

Engine-reported loading per generation rose from 43.62 s (003 repeat) to 49.57 s (010) and 53.68 s (011);
their text-encode and step-0 intervals are 1.8–4.9 s longer than the repeat's, although neither option
touches those stages, and 011's steady step is identical to the repeat's. The 010/011 baselines ran
about two hours after the repeat, but their failed smokes, run right after it, also showed slow text
encoding (n = 1 each), so elapsed time alone does not explain the drift; the cause (page-cache / I/O
state) is a hypothesis. The 003 repeat itself reproduces the 2026-10-05 baseline
(wall 60.64 vs 60.94 s; steady step 2.713 vs 2.716 s) ([003](jetson-flux-klein-003.md)).

## Findings

- **Weight residency, not compute, separates the engines' wall times on this device.** edge-dit
  keeps all weights resident and generates in 12.0 s; sd.cpp (disk-backed) and PyTorch
  (sequential residency) reload weights every generation: sd.cpp's engine-reported loading is 43.62 s per generation in
  the 003 repeat (inside text encode, step 0 and VAE decode; [003](jetson-flux-klein-003.md)), and
  PyTorch's reloads take 63.3 s (`other_ms`). Qualitative: the residency
  policies differ by design and each is the configuration that fits for that engine.
- **Steady transformer compute ranks edge-dit ≤ 2.30 < sd.cpp 2.71 < PyTorch 3.59 s per step** on the
  same Q4_0 transformer file. Qualitative ordering (different stage boundaries; edge-dit value is an
  upper bound including step 1). Hypothesis: PyTorch's GGUF path dequantizes to BF16 per matmul.
- **Turning fused attention off costs about 30% per step in both sd.cpp and PyTorch** (×1.30, ×1.31;
  matched within each engine) and about 37% in edge-dit from two single observations. On edge-dit it
  also removes the memory headroom: the run hit the 400 MB guard floor. This option effect
  transfers across the two engines with full measurements.
- **VAE tiling at 128 px / 0.25 adds about 1.2–1.4 s to decode** in sd.cpp and PyTorch and leaves
  denoising unchanged; it is the option that makes edge-dit feasible, and lowers PyTorch's RAM peak by
  0.45 GiB. Matched tiled decode: edge-dit 1.74 s vs PyTorch 1.85 s (36 vs 25 tiles) vs sd.cpp 5.86 s
  (includes VAE reload in disk mode; qualitative).
- **Text encoding differs by checkpoint and residency, not comparably**: 0.93 s (edge-dit, resident),
  1.24 s (PyTorch NF4, reload excluded), 22.56 s (sd.cpp, reload included).

Output hashes: both sd.cpp options change the output pixels (reference `2d487c94…`, flash off
`275c50ae…`, tiling `252b6cb0…`), and so does edge-dit's flash-off run (`b7589892…` vs `7c7a84f3…`).
PyTorch outputs are bit-identical within each run; no paired image-difference or quality metric
was computed.

## Limits

- Different text-encoder checkpoints and initial latents: no image-quality or output-equality claim.
  edge-dit retains only its final image, so its run-to-run determinism is unknown.
- edge-dit's attention-off result is two single observations, not a protocol; its flash attention
  state is evidenced only by the command-line flag.
- sd.cpp memory is sampled at 1 s; its RAM peak may be underestimated relative to the 200 ms samples.
- The residency gap is not fully decomposed into policy vs implementation. sd.cpp all-resident failed untiled in
  [001](jetson-flux-klein-001.md); with VAE tiling ([012](jetson-flux-klein-012.md), 2026-10-15) it generated four
  images at about 16.3 s each before a weight-preparation failure, so no 10-image protocol exists. PyTorch
  all-resident failed in [torch 001](jetson-torch-flux-klein-001.md).
- Base (50 steps) is not covered; cross-device transfer is not addressed here.
