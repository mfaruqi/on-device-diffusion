---
name: wiki-ask
description: Answer a question about the research project from the wiki (wiki/) and the records and runs it cites, with citations. Use when the user asks what we know, what was decided, why something is the way it is, where a result came from, or what's next — e.g. "what do we know about VAE memory?", "why do we pin PCIe nodes?".
---

# Wiki ask

1. **Read `wiki/hot.md` and `wiki/index.md`** (both short). Pick the few pages whose summaries match.
   For broad questions, use `grep -ril "<term>" wiki/` instead of reading everything.
2. **Read those pages, then the records and run files they cite** for anything the answer depends on.
   Pages summarize, records hold the detail. If the two disagree, the record wins; flag the page for `/wiki-lint`.
3. **Answer** in chat, short, with a link for every claim:
   - wiki pages, e.g. `[finding](wiki/findings/…md)`;
   - records, e.g. `[record](experiments/…md)`;
   - run files, e.g. `results/runs/<id>/summary.json`.
   Separate what is measured from what is interpretation.
4. **If the wiki doesn't answer it**, say "the wiki has no supported answer". Then check the records,
   runs and proposal directly, and say which source answered, if any. Don't guess.
5. **Don't file the answer back** unless the user asks. If they do, it goes through `/wiki-ingest` with
   the records as sources, never the chat.
6. If the question reveals a gap (a missing concept, a finding with no evidence, a stale page), end with
   one line suggesting the `/wiki-ingest` or `/wiki-lint` action that would fix it.

For questions where reading many pages would crowd this conversation, delegate to the `wiki-librarian`
agent and relay its cited answer.
