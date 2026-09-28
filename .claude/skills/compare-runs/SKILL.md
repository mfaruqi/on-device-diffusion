---
name: compare-runs
description: Compare two or more completed runs across devices, workloads, engines or configurations (RQ1 transfer, variant vs reference, engine vs engine). Writes experiments/compare-<topic>.md. Use only when every run being compared already exists in results/runs/.
---

# Compare runs

This is the **only** place where statements about more than one device, engine or workload are written.

## Preconditions

1. List the run directories being compared, with their configs.
2. Check they are **matched**: checkpoint (and whether it's compressed), precision per component,
   resolution, steps, guidance, batch, prompt/initial latents, protocol. Put this in a table.
3. Anything unmatched is either (a) the variable under study, which must be exactly one, or (b) a
   confound, stated explicitly. If there are confounds, say the comparison is qualitative only.
4. Memory must be compared with the same kind of metric on both sides (device-wide vs device-wide,
   or total physical memory on shared-memory targets). Never compare allocator memory on one device
   with system memory on another.

## Note: `experiments/compare-<topic>.md`

```
# Compare: <topic>
## Question   (and the RQ it serves)
## Runs       table: run dir | device | config | what differs
## Matching   table of held-fixed settings; list confounds
## Results    side-by-side stage latency (median, range), memory, quality if measured; absolute and ratio
## Findings   what transfers and what doesn't, with the evidence for each
## Limits     what this comparison can't conclude
```

Numbers come from each run's `summary.json`. Link the experiment records and don't re-derive their content.
The record needs `type: experiment-record` frontmatter (see `/log-experiment`). Then run
`/wiki-ingest experiments/compare-<topic>.md`. Comparisons are where findings that transfer, or don't,
come from, so expect it to propose finding updates for approval.
