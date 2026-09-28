# On-device diffusion: instructions for Claude

@AGENTS.md

## Claude Code specifics

- The project instructions above live in `AGENTS.md`, shared with Codex. Edit them there, not here.
- Skills are invoked as `/name`. The `wiki-librarian` subagent is in `.claude/agents/`.
- Hooks in `.claude/settings.json`: SessionStart loads `wiki/hot.md`; SessionEnd regenerates the
  index and prints wiki reminders (`scripts/hooks/`).
