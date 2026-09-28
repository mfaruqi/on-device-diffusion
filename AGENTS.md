# On-device diffusion: instructions for coding agents (Codex, Claude Code)

Master's/research project, Purdue EcoAI Lab (advisor Prof. Haoran You). Author: Mahad Faruqi.

## North star

Read `wiki/project/overview.md` (condensed proposal) before planning any work. The PDF in `proposal/`
is the source of truth. In one line: **a diffusion-aware compiler/runtime planner that jointly
chooses reuse regions, precision and memory residency to minimize latency under a memory budget
and a declared quality requirement**, evaluated across A100, Jetson Orin Nano, M1 and iPhone.

Every task should map to a week in the milestone table and to RQ1, RQ2 or RQ3. If a request doesn't
fit any of them, say so briefly and ask whether it's a deliberate scope change. Don't silently
redefine the RQs or the hypothesis. Quote the proposal rather than paraphrasing it.

## Project wiki (read first)

- At the start of a session, read `wiki/hot.md` (current state, ≤ 40 lines). Use `wiki/index.md` to find pages.
- Answer project questions from the wiki and the records it cites before anything else (`/wiki-ask`, or
  the read-only `wiki-librarian` agent). If the wiki has no supported answer, say so.
- The wiki holds conclusions, not data. Follow `wiki/SCHEMA.md` for any wiki edit: cite every claim,
  update before create, one registry row per experiment, never auto-resolve contradictions.
- After an experiment or comparison is written up, run `/wiki-ingest`. Run `/wiki-lint` weekly.
  `python3 scripts/wiki_lint.py` must stay at 0 errors.
- Rewrite `wiki/hot.md` at the end of a session that changed the project's state.

## Sources of truth, in priority order

1. `proposal/OnDeviceDiffusionProposal.pdf` → `wiki/project/overview.md`
2. `wiki/methods/baseline-metrics.md`: the definition of every reported metric
3. Experiment notes `experiments/<experiment-id>.md` and raw files in `results/runs/`
4. The code (`scripts/`, `configs/`): what the runner actually does

## Documentation scope (important)

- **Each document is about its own subject only.** An A100 experiment note describes the A100 run:
  setup, numbers, observations and interpretation *for that configuration on that device*. Don't
  add predictions about other devices, other models, or plans mentioned in the current conversation.
- **Cross-device or cross-workload statements go only in comparison records** (`experiments/compare-*.md`)
  and in wiki findings built from them. A record's "Next experiment" section may name future work
  in one line.
- **Write for a reader who never saw the chat**: the advisor, or the author six weeks from now. No
  "as discussed", no "the user asked", no session narrative. State facts, methods and results.
- **Separate measurement from interpretation.** Numbers come from `summary.json` / CSVs, never retyped
  from memory. Hypotheses are labelled as such.
- Shared definitions (metrics, protocols, device procedures) live in one file and are linked, not repeated.

## Experiment invariants

- **One change per labelled configuration.** The reference is BF16, no optimizations. Every variant is
  a new config file with its change named in the `optimizations` block.
- **Never fall back silently** (precision, offload, resolution, checkpoint). A failed run keeps its
  config, partial measurements and error in `status.json`.
- Pin checkpoint revisions to commit hashes (`--prepare-only` → `*.resolved.json`). GPU jobs run offline.
- Protocol: first run reported separately, warm-ups discarded, medians and min–max of measured runs.
  Memory in GiB. Never add allocated and reserved.
- Compressed/replacement checkpoints (quantized, distilled, Bonsai, DreamLite) are reported separately
  from execution-only changes. A lower resolution is a separately labelled workload.
- Keep calibration and held-out prompt sets separate. Don't look at the held-out set until week 10.
- If the runner changes how a metric is measured, update `wiki/methods/baseline-metrics.md` in the same commit.

## Repository map

| Path | Contents |
|---|---|
| `configs/` | one JSON per labelled configuration, plus the `*.resolved.json` pinned copy |
| `scripts/` | runners (`run_flux.py`) and Slurm scripts (`gilbreth.slurm`) |
| `wiki/` | curated project wiki: hot, index, log, registry, project, RQs, findings, systems, concepts, papers, methods (metrics, how-tos). Rules in `wiki/SCHEMA.md` |
| `experiments/` | one record per experiment and comparison (full write-up, with frontmatter) |
| `raw/` | immutable sources to ingest: `papers/`, `meetings/` |
| `results/README.md` | index: one row per run directory |
| `results/runs/<id>-<timestamp>/` | small raw outputs (CSV/JSON). Images and traces are git-ignored |
| `flux_test/` | old FLUX.1-schnell exploration. Not a baseline; don't build on it |

Naming: experiment id `<device>-<model>-<NNN>` (e.g. `a100-flux-klein-001`); config
`<device>-<model>-<variant>.json` (e.g. `a100-flux-klein-bf16.json`).

## Environment (Gilbreth A100)

- Python: `~/.conda/envs/2025.06-py313/flux_env/bin/python` (call it directly; no activation).
- `HF_HOME=/scratch/gilbreth/mfaruqi/huggingface`. Scratch is purged after 60 days of no use.
- Slurm: `--account=you139 --partition=a100-40gb`. The login node has no GPU; never run benchmarks there.
- Long-lived shared artifacts go in `/depot/you139/mfaruqi/on-device-diffusion/`.
- The user is new to Slurm/HPC: explain concepts briefly when they come up.
- Other devices get their own how-to file (`wiki/methods/<device>-howto.md`) when first used.

## Git

- Work on a branch, never `main`. No global identity is set on Gilbreth; commit with
  `git -c user.name="Mahad Faruqi" -c user.email="mfaruqi@purdue.edu" commit ...`.
- Never commit weights, tokens, images, traces or logs (see `.gitignore`).

## Skills in this repo

Skills live in `.agents/skills/` (Codex) and `.claude/skills/` (Claude Code); these are the same files.
Invoke a skill as `$name` in Codex or `/name` in Claude Code. Where a skill or page says `/wiki-ingest`,
it means that skill in whichever tool you are. The read-only `wiki-librarian` agent is defined in
`.codex/agents/wiki-librarian.toml` (Codex) and `.claude/agents/wiki-librarian.md` (Claude Code).
A git pre-commit hook (`scripts/githooks/`, enabled with `git config core.hooksPath scripts/githooks`)
blocks commits while `python3 scripts/wiki_lint.py` reports errors.


- `/new-experiment`: scaffold a labelled config and run it
- `/log-experiment`: write the experiment record and results-index row, then `/wiki-ingest`
- `/benchmark-code`: conventions for writing or changing runners on any device
- `/compare-runs`: the only place where devices or workloads are compared
- `/scope-check`: status against the milestone table and RQs, with a drift check
- `/wiki-ingest`: integrate a record, paper, meeting note or decision into the wiki (minimal, approved changes)
- `/wiki-ask`: answer from the wiki with citations; `wiki-librarian` agent for the same, read-only
- `/wiki-lint`: `scripts/wiki_lint.py` plus a scoped contradiction/staleness pass
