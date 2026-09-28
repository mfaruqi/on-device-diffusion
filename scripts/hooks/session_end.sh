#!/usr/bin/env bash
# Agent session-end hook (Claude Code SessionEnd; also usable by other agents).
# Bookkeeping only: regenerate the wiki index, then print reminders for anything left undone.
# Never edits wiki content beyond the generated index.
set -u
cd "$(dirname "$0")/../.." || exit 0

out="$(python3 scripts/wiki_lint.py --write-index 2>&1)"
msgs=()
if grep -q "^ERROR  registry:" <<<"$out"; then
  msgs+=("new run directories or records are not in wiki/experiments.md: run /wiki-ingest")
fi
n_err="$(grep -c '^ERROR' <<<"$out")"
if [[ "$n_err" -gt 0 ]]; then
  msgs+=("wiki lint has $n_err error(s): run python3 scripts/wiki_lint.py")
fi
# hot.md should be at least as new as the newest log entry
hot_date="$(grep -m1 '^updated:' wiki/hot.md 2>/dev/null | awk '{print $2}')"
log_date="$(grep -m1 -o '^## \[[0-9-]*\]' wiki/log.md 2>/dev/null | tr -d '#[] ')"
if [[ -n "$hot_date" && -n "$log_date" && "$hot_date" < "$log_date" ]]; then
  msgs+=("wiki/hot.md (updated $hot_date) is older than the latest log entry ($log_date): rewrite it")
fi

if [[ ${#msgs[@]} -gt 0 ]]; then
  printf 'Wiki reminders:\n' >&2
  printf '  - %s\n' "${msgs[@]}" >&2
fi
exit 0
