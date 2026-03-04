# CLAUDE.md

> Code agent instructions for this repository.

## Identity

Scry is the agent identity. The canonical source of truth for identity, voice, and operational rules lives in the [grimoire](https://github.com/dunamismax/grimoire) repo:

- `SOUL.md` — identity, worldview, voice
- `AGENTS.md` — operational rules, stack contract, verification

Read those files first. Then read this repo's README and task-relevant code.

## Repo Rules

- Runtime/tooling: Python 3.12+, uv, Ruff, mypy.
- Run `uv run ruff check src/ && uv run mypy src/` before committing.
- No AI attribution in commits. Commit as `dunamismax`.
- Push directly to main. Force-push when needed.
- Dual remotes: GitHub + Codeberg.
- Never commit API keys, account numbers, or IBKR credentials.
