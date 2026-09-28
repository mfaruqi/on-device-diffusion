---
name: new-experiment
description: Scaffold and run a new labelled experiment configuration (new device, new workload, or one optimization change against a reference). Use when the user wants to run a baseline, try a variant (quantization, residency, tiling, compile, reuse policy), or port the benchmark to another device.
---

# New experiment

## 1. Place it in the plan

Before writing anything, state in one or two lines:
- which milestone week (see `wiki/project/overview.md`) and which RQ it serves,
- which **reference configuration** it is compared against (same device, same workload),
- the **one** thing that changes relative to that reference.

If more than one thing changes, split it into several configs. If the reference doesn't exist yet on
this device, run the reference first.

## 2. Name it

- Experiment id: `<device>-<model>-<NNN>`, where NNN increments per device+model (`a100-flux-klein-002`).
- Config file: `configs/<device>-<model>-<variant>.json`. The variant names the change (`bf16`,
  `bf16-vae-tiled`, `int4-transformer`, `bf16-seq-residency`).
- A lower resolution, a different checkpoint or a quantized checkpoint is a **different workload**:
  say so in `description` and in the id's variant.

## 3. Write the config

Copy the reference config and change only:
- `id`, `description` (one sentence: what changes vs which reference),
- `device_label`,
- the single field in `optimizations` (or `model` / `precision` / `workload` if that *is* the change).

Keep `protocol` identical to the reference unless the protocol itself is being studied. Checkpoints
are always pinned: run `--prepare-only` on a machine with network access, then commit `*.resolved.json`.

## 4. Make the runner support it (if needed)

If the runner refuses the new option, extend it following `/benchmark-code`. Don't bypass its
assertions. The runner must apply exactly what the config says and record it, or fail loudly.

## 5. Run

- Gilbreth: `sbatch --export=ALL,CONFIG=configs/<name>.resolved.json scripts/gilbreth.slurm`
  (add `PROFILE=1` for a separate profiler job). Explain Slurm steps briefly; the user is new to HPC.
- Other devices: follow `wiki/methods/<device>-howto.md`. Write it the first time a device is used, covering
  the setup, the environment and how memory is measured on that device.
- Check `status.json` and `summary.json` before logging anything.

## 6. Log

Run `/log-experiment`, which ends with `/wiki-ingest`. A run isn't finished until the record, the
results-index row and the wiki registry row exist.
