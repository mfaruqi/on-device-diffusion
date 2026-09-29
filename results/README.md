# Results index

One row per run directory under `results/runs/`. Numbers are medians of measured runs. stable-diffusion.cpp runs have no allocator metric; their device-wide peak is shown instead. See
`wiki/methods/baseline-metrics.md` for definitions. Images and profiler traces in each run
directory are git-ignored (kept locally, not in git).

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `a100-flux-klein-001__baseline__20260923-221530` | [a100-flux-klein-001](../experiments/a100-flux-klein-001.md) | gilbreth-a100-40gb (PCIe, g008) | a100-flux-klein-bf16 | 1240.6 | 47.0 / 980.8 / 165.3 | 17.32 | complete (job 11807390) |
| `a100-flux-klein-001__profile__20260923-221717` | [a100-flux-klein-001](../experiments/a100-flux-klein-001.md) | gilbreth-a100-40gb (PCIe, g008) | a100-flux-klein-bf16 + `--profile` | 1236.8 | 47 / 981 / 165 | 17.32 | complete (job 11807396) |
| `a100-flux-klein-001__repeat__20260925-123908` | [a100-flux-klein-001](../experiments/a100-flux-klein-001.md) | gilbreth-a100-40gb (PCIe, g006) | a100-flux-klein-bf16 (repeat after runner refactor) | 1229.6 | 47.5 / 979.5 / 165.9 | 17.32 | complete (job 11817919) |
| `a100-sdcpp-flux-klein-001__baseline__20260925-114138` | [a100-sdcpp-flux-klein-001](../experiments/a100-sdcpp-flux-klein-001.md) | gilbreth-a100-40gb (**SXM4**, n001) | a100-sdcpp-flux-klein-bf16 | 2294.8 | 60.7 / 1625.6 / 585.4 | – (device 22.88) | complete (job 11817893); SXM4 node, not comparable with PCIe runs |
| `a100-sdcpp-flux-klein-001__baseline__20260925-114328` | [a100-sdcpp-flux-klein-001](../experiments/a100-sdcpp-flux-klein-001.md) | gilbreth-a100-40gb (PCIe, g004) | a100-sdcpp-flux-klein-bf16 | 2423.3 | 64.7 / 1725.4 / 603.5 | – (device 22.88) | complete (job 11817906) |
| `a100-sdcpp-flux-klein-001__profile__20260925-124021` | [a100-sdcpp-flux-klein-001](../experiments/a100-sdcpp-flux-klein-001.md) | gilbreth-a100-40gb (PCIe, g006) | a100-sdcpp-flux-klein-bf16 + `--nsys` | – | – | – | failed (job 11817920): nsys could not attach to the harness |
| `a100-sdcpp-flux-klein-001__profile__20260925-124703` | [a100-sdcpp-flux-klein-001](../experiments/a100-sdcpp-flux-klein-001.md) | gilbreth-a100-40gb (PCIe, g005) | a100-sdcpp-flux-klein-bf16 + `--nsys` | 2510.3 | 66.4 / 1785.1 / 629.2 | – (device 22.88) | complete (job 11818087); profiled, not a baseline number |

Jetson feasibility attempt (partial evidence import; no benchmark medians):

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `jetson-flux-klein-001__attempt__20260928-183618` | [jetson-flux-klein-001](../experiments/jetson-flux-klein-001.md) | jetson-orin-nano | jetson-flux-klein-q4-512-all-resident | – | – | – | failed (exit 1): insufficient memory during denoising preparation; full logs pending |
| `jetson-flux-klein-002__attempt__20260928-184410` | [jetson-flux-klein-002](../experiments/jetson-flux-klein-002.md) | jetson-orin-nano | jetson-flux-klein-q4-512-segmented | – | – | – | failed (exit 1): text-encoding memory capacity; full logs pending |
| `jetson-flux-klein-003__attempt__20260928-185009` | [jetson-flux-klein-003](../experiments/jetson-flux-klein-003.md) | jetson-orin-nano | jetson-flux-klein-q4-512-segmented-disk | – | – | – | complete (exit 0): one image inspected; full CLI logs analyzed, feasibility only |
| `jetson-flux-klein-003__profile__20260928-195625` | [jetson-flux-klein-003](../experiments/jetson-flux-klein-003.md) | jetson-orin-nano | jetson-flux-klein-stage-profile | – | – | – | complete; imported timeline reviewed; single profile |
| `jetson-flux-klein-003__profile__20260928-194910` | [jetson-flux-klein-003](../experiments/jetson-flux-klein-003.md) | jetson-orin-nano | jetson-flux-klein-stage-profile | – | – | – | generation succeeded; stage validation failed |
| `jetson-flux-klein-003__profile__20260928-191418` | [jetson-flux-klein-003](../experiments/jetson-flux-klein-003.md) | jetson-orin-nano | jetson-flux-klein-q4-512-segmented-disk-profile | – | – | – | CUDA events confirmed in supplied stats; original trace import pending |

## Run directory names

`<experiment-id>__<kind>__<YYYYMMDD-HHMMSS>`, where kind is `baseline` (clean timing run), `profile`
(profiler attached; not quoted as latency), `repeat` (a baseline re-run to check reproducibility) or
`attempt` (a single feasibility try, not the full protocol). The runners create these names (`--kind`
overrides the default). W&B runs use the directory name as their id.

Renamed on 2026-09-28. Files inside a run (`environment.json`, `engine/command.json`, READMEs) still
show the old path from when they were written:

| Old directory name | New directory name |
|---|---|
| `a100-flux-klein-001-20260923-221530` | `a100-flux-klein-001__baseline__20260923-221530` |
| `a100-flux-klein-001-20260923-221717-profile` | `a100-flux-klein-001__profile__20260923-221717` |
| `a100-flux-klein-001-20260925-123908` | `a100-flux-klein-001__repeat__20260925-123908` |
| `a100-sdcpp-flux-klein-001-20260925-114138` | `a100-sdcpp-flux-klein-001__baseline__20260925-114138` |
| `a100-sdcpp-flux-klein-001-20260925-114328` | `a100-sdcpp-flux-klein-001__baseline__20260925-114328` |
| `a100-sdcpp-flux-klein-001-20260925-124021-profile` | `a100-sdcpp-flux-klein-001__profile__20260925-124021` |
| `a100-sdcpp-flux-klein-001-20260925-124703-profile` | `a100-sdcpp-flux-klein-001__profile__20260925-124703` |
| `jetson-flux-klein-q4-512-20260928-183618` | `jetson-flux-klein-001__attempt__20260928-183618` |
| `jetson-flux-klein-002-20260928-184410` | `jetson-flux-klein-002__attempt__20260928-184410` |
| `jetson-flux-klein-003-20260928-185009` | `jetson-flux-klein-003__attempt__20260928-185009` |

## Files in each run directory

| File | Purpose |
|---|---|
| `config.json` | Exact workload and checkpoint revision |
| `environment.json` | GPU, driver, packages, CUDA runtime, git state, Slurm job |
| `environment-freeze.txt` | `pip freeze` |
| `scheduler.json` | Scheduler config from the loaded pipeline |
| `load.json` | Cached load/placement time, memory after load, per-component weight sizes |
| `runs.csv` | One row per generation (first / warmup / measured), with stage times and memory |
| `stages.csv` | Long format: every stage of every run, GPU and host time, allocator memory |
| `memory_segments.json` | Allocator peaks for every stage and every gap between stages |
| `summary.json` | Median / min / max / stdev over measured runs, first run, determinism flag |
| `nvidia-smi.txt` | Snapshot after the measured runs (context only, not a peak) |
| `images/` | `first.png`, `measured-0.png` for a functional check (git-ignored) |
| `profile/` | `op_table.txt` (committed), `trace.json` (git-ignored), only with `--profile`; `denoise_kernels.md/.json` from `scripts/analyze_profile.py` |
| `engine/` | stable-diffusion.cpp runs only: harness command, sd.cpp log, raw stage timestamps (see [sdcpp-howto.md](../wiki/methods/sdcpp-howto.md)) |
| `status.json` | running / complete / failed (+ traceback) |
