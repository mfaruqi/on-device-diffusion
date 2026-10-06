---
name: wiki-lint
description: Health-check the project wiki — deterministic checks via scripts/wiki_lint.py (links, anchors, frontmatter, numbers vs JSON, registry, budgets, orphans), then an LLM pass for contradictions, staleness and gaps on recently changed pages. Reports first, fixes only what the user approves. Use weekly, after a batch of ingests, or when the user asks to check or clean up the wiki.
---

# Wiki lint

## 1. Deterministic pass (no judgment needed)

```bash
python3 scripts/wiki_lint.py --write-index
```

Errors: broken links or anchors, bad frontmatter, a quoted number that no longer matches its JSON, an
experiment record or run directory missing from the registry, bad log headings. Each has a mechanical
fix, so fix them directly and re-run until the output shows `0 error(s)`.

Warnings (over budget, orphans): propose the fix instead of applying it. Over budget → a split, or
compaction into an Archive section. Orphan → which page should link to it, or whether it should go.

## 2. Judgment pass (scoped)

Check only the pages changed since the last `lint` entry in `wiki/log.md`, plus the pages they link to
(`git diff --name-only <last-lint-commit> -- wiki/` plus uncommitted and untracked pages from
`git status --porcelain wiki/`, or file mtimes if nothing is committed).
For each page, look for:
1. **Contradictions**: two pages, or a page and its cited record, saying different things. Never pick
   a winner. Propose an `open-questions.md` entry with both sides cited.
2. **Stale claims**: a finding whose cited record or run has been superseded by a newer experiment
   (check `wiki/experiments.md`), or whose RQ page no longer matches its status.
   On any page, including methods/how-tos, a status statement ("not yet", "pending", "remains",
   "has not been run") must still be true according to the registry and records.
3. **Unsupported claims**: a sentence on a finding, system or concept page with no link to evidence,
   or evidence that is only another wiki page.
4. **Missing links**: a page names an entity or concept that has its own page but doesn't link to it.
5. **Scope violations**: a system page talking about another device or engine; a cross-device claim
   not backed by a comparison record.
6. **Flooding**: per-run tables or numbers beyond the two-per-finding rule; a page that mirrors a single record.

## 3. Report, then fix

List the findings grouped by severity, each with a proposed fix. Apply only what the user approves.
Then re-run step 1 and add a log entry: `## [YYYY-MM-DD] lint | <n> fixes, <m> open`.
