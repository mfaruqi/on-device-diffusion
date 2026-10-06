---
type: method
summary: Rules for maintaining this wiki. Read before any wiki edit.
status: active
updated: 2026-10-06
---

# Wiki schema

This wiki holds the project's **durable knowledge**: what we know, why we decided things, and what's
open. It doesn't hold data. Per-run numbers live in run directories and experiment records, and the
wiki links to them. The pattern follows Karpathy's
[LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). The rules below
draw on what its implementers reported working (see [Why these rules](#why-these-rules)).

## Layers

| Layer | Where | Who writes | Rule |
|---|---|---|---|
| Raw evidence | `proposal/`, `results/runs/<id>/`, `raw/papers/`, `raw/meetings/` | runners, the user | immutable; never edited |
| Records | `experiments/<id>.md`, `experiments/compare-*.md` | `/log-experiment`, `/compare-runs` | one per experiment; full detail lives here |
| Wiki | `wiki/` | the LLM, with the user's approval for conclusions | short, linked, cited |
| Schema | this file | the user and the LLM together | changes are logged in `log.md` as `schema` |

## Special files

| File | Purpose | Budget |
|---|---|---|
| [hot.md](hot.md) | current state: week, in progress, next actions, blockers. **Rewritten**, not appended | ≤ 40 lines |
| [index.md](index.md) | one line per page: link and summary, grouped by section | one line per page |
| [log.md](log.md) | chronological one-liners, `## [YYYY-MM-DD] kind \| title`, newest first | entries ≤ 5 lines |
| [log-2026-09.md](log-2026-09.md) | archived log months (`log-YYYY-MM.md`), moved out of `log.md` when it exceeds its budget | ≤ 160 lines each |
| [experiments.md](experiments.md) | registry: **one row per experiment**, linking to the record and run directories | one row each |
| [open-questions.md](open-questions.md) | unresolved contradictions and open questions, each marked `Status: Unresolved` | – |

`kind` in log entries is one of: `experiment`, `ingest`, `decision`, `finding`, `lint`, `schema`.

## Page types and frontmatter

Every page starts with YAML frontmatter. Required: `type`, `summary` (one line, used by the
index), `status`, `updated`. Allowed values:

| type | Folder | status values | Extra fields | Budget |
|---|---|---|---|---|
| `project` | `project/` | active, archived | – | ≤ 160 lines |
| `rq` | `rq/` | active | – | ≤ 80 lines |
| `finding` | `findings/` | tentative, supported, contested, superseded, refuted | `confidence: high\|medium\|low`, `rq`, `sources`, optional `supersedes`, `superseded_by` | ≤ 40 lines |
| `system` | `systems/` | active, planned, retired | `kind: engine\|device\|model\|cluster`, optional `aliases` | ≤ 80 lines |
| `concept` | `concepts/` | active | optional `aliases` | ≤ 60 lines |
| `paper` | `papers/` | unread, read, ingested | `ref` (citation), `source` (link to raw copy or URL) | ≤ 60 lines |
| `method` | `methods/` | active, superseded | – | none (reference docs) |

Experiment records carry frontmatter too (`type: experiment-record`, `id`, `status`, `device`,
`engine`, `runs`) so the lint can cross-check them against the registry.

## Writing rules

1. **Cite everything.** Every factual sentence on a finding, system or concept page links to its
   evidence: an experiment record, a run directory, a raw source or a paper page. A wiki page is
   **never the only evidence** for a finding; findings must cite at least one record, run or raw source.
2. **Numbers follow the state rule.** Pages describe the *shape* of a result. At most two headline
   numbers per finding, each followed by a machine-checkable citation of the JSON value it came from:
   `**2423.3 ms** ([`measured.wall_ms.median`](../../results/runs/<run-id>/summary.json))`.
   The lint reads that key and checks the number, rounded as displayed. Anything more detailed stays
   in the record.
3. **Separate evidence from interpretation.** Use `Interpretation:` or `Hypothesis:` for anything not
   directly measured. Don't upgrade a hypothesis to a finding without new evidence.
4. **Never smooth over contradictions.** If new evidence disagrees with a page, add both claims with
   citations to `open-questions.md` as `Status: Unresolved`, mark the finding `contested`, and let the
   user resolve it.
5. **Supersede, don't delete.** A finding that no longer holds gets `status: superseded` or `refuted`
   and a `superseded_by` link. Dead ends stay visible so they aren't retried.
6. **Scope rule.** System pages describe only that system. Statements across devices, engines or
   workloads belong in findings or comparison records built from matched runs.
7. **Links** are relative markdown links (`[text](../findings/x.md)`), so they work in Obsidian, VS Code
   and on GitHub. No `[[wikilinks]]`. Link only to pages that exist; the lint checks this.
8. **Names**: lowercase-hyphenated file names. Finding titles are claims ("ggml flash attention is 4×
   slower than SDPA-flash at 4608 tokens"), not topics.

## Anti-flood rules

1. **A new experiment adds one registry row and one log line.** Nothing else, unless it changes a
   conclusion: a new finding, new support for or a contradiction of an existing one, a changed RQ
   status, or a new open question.
2. **Update before create.** Before creating a page, search titles, `aliases` and `index.md`. Create a
   page only for a distinct entity, concept, finding or paper that other pages would link to.
   Attributes and updates go into the existing page.
3. **No pages from answers alone.** An answer to a question becomes a page only if the user asks and
   it rests on cited records or raw sources.
4. **No mirrors.** Don't create a page that restates a single record. Link to the record instead. A
   page earns its place by combining several sources, or by being linked from several places.
5. **Budgets are hard limits.** A page over budget gets split or compacted, and the lint flags it.
   Papers get a page only when they are ingested; unread references stay as lines in
   [papers/reading-list.md](papers/reading-list.md).

## Operations

- **Ingest** (`/wiki-ingest <source>`): read the source, then **propose** a change plan (registry row,
  pages to update or create, open questions) and wait for approval before touching findings, RQ,
  decision or system pages. Registry rows, log lines and index lines count as bookkeeping and can be
  written directly. Afterwards, run `python3 scripts/wiki_lint.py`.
- **Query** (`/wiki-ask`): read `hot.md` and `index.md`, then the relevant pages, then their cited
  records. Answer with citations. If the wiki has no supported answer, say so. Don't guess, and don't
  file the answer back unless asked.
- **Lint** (`/wiki-lint`): run `scripts/wiki_lint.py` (deterministic, no tokens), then check pages changed
  since the last lint, and their neighbours, for contradictions and missing links. Report first; fix
  only what the user approves.
- **Hot page**: rewrite `hot.md` at the end of a session that changed the project's state. It's a
  manual trigger, not every turn.

## Why these rules

These are condensed from Karpathy's gist and the implementations people reported in its comment
thread (September 2026). Each rule was raised independently by several commenters. The page-type
budgets and the Current/Archive split come from benjimixvidz. "No mirrors" comes from blurman-ai's
measurement that per-entity pages copying small files cost more tokens than grepping the originals.
Claim-level citations with lint against originals come from badwally, ranjankumar and frankchu91.
Deterministic lint split from LLM judgment: hmbseaotter, theafh, distorx. Never auto-resolve
contradictions: theafh, bluejaeha, FBoschman. The state rule for live values comes from WadeGIMPBC.
Triage and approval before writes: frankchu91, laphilosophia, asakin. The hot cache comes from AgriciDaniel.
Dead ends kept as anti-repetition memory come from skyllwt (OmegaWiki).
