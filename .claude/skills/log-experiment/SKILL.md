---
name: log-experiment
description: Document a completed (or failed) run. Writes or updates the experiment record in experiments/, adds a row to results/README.md, then hands off to /wiki-ingest for the registry row and log line. Use after any benchmark, profiling or quality run finishes, or when the user asks to "log", "document" or "write up" results.
---

# Log an experiment

## Scope rule (read first)

The note is about **this run only**: this device, this configuration, this workload.

- Don't mention other devices, other models or future experiments in Setup, Results, Observations or
  Interpretation, even if they came up in the conversation. Example: an A100 note does not say how the
  Jetson will differ. That belongs in a Jetson note, or in `/compare-runs` once both runs exist.
- Interpretation explains what *these* numbers mean for *this* configuration (where time goes, what
  sets the peak, which levers the data point to). General levers are fine; device-transfer predictions are not.
- The only forward-looking content is the **Next experiment** section, a short list of one-line items
  that name the config each would change.
- Write for someone who never saw the chat. No session narrative ("we tried", "the user wanted").

## Where the numbers come from

Read them from the run directory. Don't retype them from the conversation:
- `summary.json`: medians, min/max, stdev, first run, determinism flag
- `load.json`: load times, weight sizes
- `memory_segments.json` / `stages.csv`: which stage sets the peak
- `environment.json`: GPU, driver, package versions, git commit, Slurm job, node
- `status.json`: complete / failed + traceback
- `profile/op_table.txt`: kernel shares (report shares, not sums; profiled times are inflated)

Metric names and meanings must match `wiki/methods/baseline-metrics.md`. If a new metric was introduced,
add its definition there in the same change.

## Experiment note: `experiments/<experiment-id>.md`

Use this structure (see `experiments/a100-flux-klein-001.md` for a filled-in example). The frontmatter
is required; `wiki_lint.py` checks it against the registry.

```
---
type: experiment-record
id: <experiment-id>
status: complete | failed | partial
device: <device page slug, e.g. a100-pcie-40gb>
engine: <engine page slug, e.g. pytorch-diffusers>
runs: [<run-dir>, <run-dir>]
updated: YYYY-MM-DD
---

# <id>: <one-line description>

Status: **<complete|failed|partial>** (<date>, job <id>).

## Question
<the single question this configuration answers, and which reference it's compared against>

## Setup
- Config / resolved config / pinned revision
- Workload (precision, resolution, steps, guidance, batch, prompt set, seed)
- What changed vs the reference (exactly one thing), or "reference configuration"
- Protocol (first / warm-up / measured)
- Device, node, environment (link to run-dir files)
- Metric definitions: link to baseline-metrics.md

## Results
Run directory, job, node, versions, repo commit.
### Latency   table: stage | median | min–max | share      (+ first run, + cold load, separate rows)
### Memory    table: weights per component, peak allocated, peak reserved, device-wide peak, host/system
### Repeatability   determinism, spread, functional check of the image
### Delta vs reference   (only for variants: same table columns, absolute and % change)
### Quality   (only if measured: metric, prompt set tier, n)

## Observations
Facts visible in the data, one bullet each.

## Interpretation
What the facts imply for this configuration. Label hypotheses.

## Next experiment
One-line items, each naming the config it would change.
```

For a **failed** run, keep the same note. Fill in Setup, the error from `status.json`, any partial
measurements, and the suspected cause, labelled as a hypothesis.

## Results index: `results/README.md`

Add one row per run directory with columns in the existing order. Numbers are medians. Status includes the job id.

## Wiki: hand off to `/wiki-ingest`

The record is the full write-up. The wiki gets **one registry row** in `wiki/experiments.md` (linking the
record and every run directory) and **one log line** in `wiki/log.md`
(`## [YYYY-MM-DD] experiment | <id>: <title>`). Run `/wiki-ingest experiments/<id>.md`. It adds those,
and proposes finding/RQ/open-question updates only if the experiment changes a conclusion
([anti-flood rules](../../../wiki/SCHEMA.md#anti-flood-rules)). Don't copy results into wiki pages.

## Before finishing

- Numbers in the record and the results index agree with each other and with `summary.json`.
- `python3 scripts/wiki_lint.py` reports 0 errors.
- No cross-device or cross-conversation content outside "Next experiment".
- Offer to commit (on the branch) and remind the user to push.
