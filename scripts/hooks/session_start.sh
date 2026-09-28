#!/usr/bin/env bash
# Agent session-start hook (Claude Code SessionStart; also usable by other agents).
# Prints the wiki's current-state page and a one-line lint summary as context for the new session.
# With --json, wraps it in Claude Code's hook output format (additionalContext).
set -u
cd "$(dirname "$0")/../.." || exit 0

ctx="$(printf '## wiki/hot.md (current project state, loaded automatically)\n\n'; sed '1,/^---$/{/^---$/!d};1,/^---$/d' wiki/hot.md 2>/dev/null)"
lint="$(python3 scripts/wiki_lint.py 2>/dev/null | tail -2 | tr '\n' ' ')"
ctx="$ctx"$'\n\n'"Wiki lint: ${lint:-not run}. Rules: wiki/SCHEMA.md. Index: wiki/index.md."

if [[ "${1:-}" == "--json" ]]; then
  jq -n --arg c "$ctx" '{hookSpecificOutput: {hookEventName: "SessionStart", additionalContext: $c}}'
else
  printf '%s\n' "$ctx"
fi
