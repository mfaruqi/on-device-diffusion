---
name: scope-check
description: Check project status against the proposal's milestone table and research questions, and flag drift. Use at the start of a week, before planning a batch of work, when the user asks "where are we" or "what next", or when a request seems outside the proposal.
---

# Scope check

1. Read `wiki/hot.md`, `wiki/project/milestones.md`, `wiki/project/overview.md`, the RQ pages in
   `wiki/rq/` and `wiki/experiments.md`.
   Work out the current week from today's date and the milestone table.
2. Report a table: this week's and last week's milestone goals | status (done / partial / not started)
   | evidence (note or run link).
3. For each RQ, one line: what evidence exists so far and what's missing.
4. **Drift check**: flag any place where notes or current plans
   - restate the hypothesis or RQs differently from the proposal,
   - drop a proposal item (e.g. a device, workload or review task) without a recorded decision,
   - spend significant effort on work that maps to no milestone or RQ,
   - mix compressed-checkpoint results with execution-only results.
5. Recommend the next 1–3 concrete experiments, each named by the config it would create
   (see `/new-experiment`), in priority order for the current week.

Keep it short. The output is for the user in chat. If a milestone's status or the next actions
changed, propose the edit to `wiki/project/milestones.md` and `wiki/hot.md`; don't write status snapshots.
