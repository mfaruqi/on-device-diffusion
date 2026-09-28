---
name: benchmark-code
description: Conventions for writing or modifying benchmark runners, device adapters, Slurm/device scripts and analysis scripts in this repo. Use whenever generating or editing code under scripts/ or adding support for a new device, backend, engine or optimization option.
---

# Benchmark code conventions

The runner's outputs feed later planner work (the seed database and RQ1–RQ3 comparisons). Keep
them uniform and trustworthy across devices and engines.

## Non-negotiables

- **Config-driven, no hidden knobs.** Every setting that affects latency, memory or output comes from
  the config and is echoed into the run directory (`config.json`).
- **Assert, don't adapt.** Check preconditions (device present, dtype, pinned revision, expected number
  of denoiser calls, option actually applied). Refuse unknown or unsupported `optimizations` keys.
  Never catch an OOM or error and retry with other settings.
- **Fail with a record.** Write `status.json` as `running` at the start. On exception, write `failed` with
  the traceback and keep partial CSV rows.
- **Offline during measurement.** Downloads happen only in `--prepare-only`.
- **Don't modify library source** (diffusers etc.). Instrument with wrappers and hooks, as `StageRecorder` does.
- **Instrumentation must not change execution.** No syncs inside timed regions except where the metric
  definition says so. GPU stage times use device events or an equivalent.

## Uniform output schema

Every runner, including wrappers around C++ engines and non-CUDA backends, writes the same files
(`config.json`, `environment.json`, `load.json`, `runs.csv`, `stages.csv`, `summary.json`, `status.json`)
with the same column names and units (ms, s for load, GiB). When a backend can't produce a column,
write it empty and record why in `summary.json → unavailable_metrics`. Don't invent substitutes under
the same name.

Stage names are shared across workloads: `text_encode`, `denoise_step_<i>`, `vae_decode`, `postprocess`.
Add new names only for genuinely new stages, and define them in `wiki/methods/baseline-metrics.md`.

## Device-specific code

- Put device differences (memory source, timing primitive, environment capture) behind a small
  adapter. Don't scatter `if device ==` branches through the benchmark logic.
- Each adapter documents its memory source. Its definition goes in `baseline-metrics.md` under that
  device's heading, not in the code comments of another device.
- Seeds aren't portable across frameworks. For cross-engine comparisons, pass explicit initial latents.

## Style

- Match `scripts/run_flux.py`: standard library + torch/diffusers, small functions, `write_json` helpers,
  comments only where the reason isn't obvious.
- Python 3.10 compatible (the Gilbreth env). No new dependencies without saying so. Record them in
  the environment capture.
- Slurm scripts: `#SBATCH` defaults for `you139` / `a100-40gb`, call the env's python directly, set
  `HF_HUB_OFFLINE=1`, overridable via `--export=ALL,VAR=...`. Comment each non-obvious line for a new Slurm user.

## Before handing back code

- Syntax-check it (`python -m py_compile`). If a GPU is needed to test it, say it's untested and give
  the exact `sbatch` command.
- If a metric's measurement changed, update `wiki/methods/baseline-metrics.md` and say that old runs are not
  directly comparable.
