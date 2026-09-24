# Results index

One row per run directory under `results/runs/`. Numbers are medians of measured runs;
see `notes/baseline-metrics.md` for definitions. Images and profiler traces in each run
directory are git-ignored and stay on scratch.

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `a100-flux-klein-001-20260923-221530` | [a100-flux-klein-001](../notes/a100-flux-klein-001.md) | gilbreth-a100-40gb | a100-flux-klein-bf16 | 1240.6 | 47.0 / 980.8 / 165.3 | 17.32 | complete (job 11807390) |
| `a100-flux-klein-001-20260923-221717-profile` | [a100-flux-klein-001](../notes/a100-flux-klein-001.md) | gilbreth-a100-40gb | a100-flux-klein-bf16 + `--profile` | 1236.8 | 47 / 981 / 165 | 17.32 | complete (job 11807396) |

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
| `profile/` | `op_table.txt` (committed), `trace.json` (git-ignored), only with `--profile` |
| `status.json` | running / complete / failed (+ traceback) |
