---
type: method
summary: How to run experiments on Gilbreth — Slurm basics, accounts, storage rules, the baseline procedure.
status: active
updated: 2026-09-28
---

# Running experiments on Gilbreth

## What Slurm is

Gilbreth is a shared cluster. When you SSH in, you land on a **login (front-end)
node** (`gilbreth-fe0x`). It has no usable GPU and is only for editing, small
downloads and submitting work. GPUs live on **compute nodes**, and **Slurm** is the
scheduler that hands them out.

You describe a job in a **Slurm script**: a bash script with `#SBATCH` comment lines
that request resources (account, partition/GPU type, number of GPUs, CPUs, RAM,
time limit). `sbatch script.slurm` queues it. When resources free up, Slurm runs the
script on a compute node and writes its stdout/stderr to a log file. Because
the script records every resource request and command, anyone can re-run the
same experiment with one command. That makes the script part of the reproducibility
record, so it lives in git (`scripts/gilbreth.slurm`).

Useful commands:

| Command | What it does |
|---|---|
| `slist` | Accounts you can charge and their GPU partitions |
| `sbatch scripts/gilbreth.slurm` | Submit the batch job |
| `squeue -u $USER` | See your queued/running jobs |
| `scancel JOBID` | Cancel a job |
| `sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS` | Job accounting after it finishes |
| `sinteractive -A you139 -p a100-40gb --gpus-per-node=1 -t 1:00:00` | Interactive shell on a GPU node (for debugging) |

## This project's setup (as of 2026-09-23)

| Item | Value |
|---|---|
| Account | `you139`: 1× A100-40GB (`a100-40gb` partition), QOS `normal`/`standby`/`training` |
| Other account | `ai-forge`: A10s only (not used for the A100 baseline) |
| Python env | `~/.conda/envs/2025.06-py313/flux_env`: Python 3.10, torch 2.5.1+cu121, diffusers 0.40.0 (has `Flux2KleinPipeline`), transformers 5.16.1 |
| Model cache | `HF_HOME=/scratch/gilbreth/mfaruqi/huggingface` (~16 GB for klein 4B). **Scratch can be purged**, so re-run `--prepare-only` if the files are gone |
| Repository | `/home/mfaruqi/on-device-diffusion` (moved from scratch on 2026-09-24; home has daily snapshots, 25 GB quota) |
| Big outputs | images and `profile/trace.json` stay in `results/runs/<id>/` and are git-ignored. Watch the home quota (`myquota`) |
| Reference images (planned) | `/depot/you139/mfaruqi/on-device-diffusion/references/` (100 GB lab-shared depot) |

Keep the environment unchanged between baseline and later comparisons. Each run records
`pip freeze` in `environment-freeze.txt`.

## Storage rules

- **Scratch** (`/scratch/gilbreth/mfaruqi`): files not read or modified for 60 days are deleted, with an
  email warning a week before. There is no backup. Keep only re-creatable data here: the model cache and
  temporary job outputs.
- **Home** (`~`): 25 GB, daily snapshots. Holds the repo and the conda env.
- **Depot** (`/depot/you139`): 100 GB, shared with the lab. Holds long-lived artifacts such as reference images.
- **GitHub**: push after each session. Until then, local commits exist only on the cluster.

## Baseline procedure (FLUX.2 klein, A100)

```bash
cd ~/on-device-diffusion
PY=~/.conda/envs/2025.06-py313/flux_env/bin/python
export HF_HOME=/scratch/gilbreth/mfaruqi/huggingface

# 1. Login node: pin the revision + download (~16 GB, no GPU)
$PY scripts/run_flux.py --config configs/a100-flux-klein-bf16.json --prepare-only
#    -> writes configs/a100-flux-klein-bf16.resolved.json (commit this)

# 2. Submit the baseline (~5–10 min of GPU time expected)
mkdir -p logs
sbatch scripts/gilbreth.slurm
squeue -u $USER

# 3. Separately, the instrumented profiler run
sbatch --export=ALL,PROFILE=1 scripts/gilbreth.slurm

# 4. Read logs/flux-baseline-JOBID.out -> it prints the run directory
```

If a job fails, `results/runs/<id>-<stamp>/status.json` holds the traceback, and any
completed rows are already in `runs.csv`. The runner will not fall back to other
precision or offload settings on its own.
