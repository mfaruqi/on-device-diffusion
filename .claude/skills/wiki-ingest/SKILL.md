---
name: wiki-ingest
description: Integrate a new source into the project wiki (wiki/) — an experiment record, a comparison, a paper, a meeting note, or a decision. Proposes a minimal change plan and applies it after approval. Use after /log-experiment or /compare-runs, when the user adds a paper or meeting notes to raw/, or says "add this to the wiki".
---

# Wiki ingest

Read `wiki/SCHEMA.md` first; it's the rulebook. This skill is the procedure.

## 1. Classify the source

| Source | Where it lives | Default outcome |
|---|---|---|
| Experiment record (`experiments/<id>.md`) | written by `/log-experiment` | **bookkeeping only**: registry row + log line |
| Comparison record (`experiments/compare-*.md`) | written by `/compare-runs` | registry row + log line; propose findings for claims that transfer or don't |
| Paper | `raw/papers/` (PDF or markdown) or a URL | paper page + reading-list line; propose updates to concepts/RQ pages it bears on |
| Meeting note | `raw/meetings/YYYY-MM-DD.md` (as written, never edited) | decisions → `project/decisions.md`; action items → `hot.md`; questions → `open-questions.md` |
| Decision made in conversation | none: write the record | new D-NNN entry in `project/decisions.md` |

## 2. Triage: what's durable?

Answer these from the source, and quote where each answer comes from:
1. Does it **create** a conclusion that no page holds yet, which other pages would link to? → a new finding or concept.
2. Does it **support or contradict** an existing finding? Search `wiki/findings/` titles, summaries and
   aliases (`grep -ril`) before deciding. Support → add the citation and maybe raise `confidence`.
   Contradiction → an `open-questions.md` entry with both sides cited, and the finding goes `contested`.
   **Never** rewrite the old claim to fit.
3. Does it change an **RQ's evidence or gaps**, a **milestone status**, or `hot.md`'s next actions?
4. Is it a **dead end** (a variant that didn't help, a failed approach)? That's a finding with status
   `refuted`, kept so it isn't retried.
If the answer to all four is no, it's bookkeeping only. That's the normal case for a routine experiment.

## 3. Propose, then write

Show the user a short plan before touching any finding, RQ, decision, system or concept page:

```
Source: <path>
Bookkeeping (will write): registry row …, log line …, index regen
Proposed (needs OK):
  update findings/<x>.md  — add citation to <run>; confidence medium→high
  new    findings/<y>.md  — "<claim-shaped title>" (sources: …)
  open-questions.md       — <question>, both sides cited
Not added (and why): <things deliberately left in the record>
```

Write bookkeeping directly, and the rest only after approval. Follow SCHEMA's writing rules:
- cite every claim;
- at most two numbers per finding, each with a `` [`key`](…/summary.json) `` citation;
- keep to the page budgets;
- **update before create**.

## 4. Finish

1. `python3 scripts/wiki_lint.py --write-index`, and fix every error it reports.
2. Add one `log.md` entry: `## [YYYY-MM-DD] <kind> | <title>`, with a link to the source.
3. If the project's state changed (a milestone moved, a blocker cleared), rewrite `hot.md`, keeping it ≤ 40 lines.
4. Tell the user which files changed, one line each.

## Don'ts
- Don't copy tables or per-run numbers into the wiki. Link to the record.
- Don't create a page for a single record ("no mirrors"), or for an unread reference (it stays a
  reading-list line).
- Don't invent links. Link only to files that exist; the lint will fail otherwise.
- Don't put conversation narrative in pages ("as discussed"). Write for a reader who never saw the chat.
