# scry-trader

AI-assisted, human-directed trading system built on Claude Opus 4.6, Python, and Interactive Brokers.

## Architecture

- **broker.py** — IBKR connection via `ib_async`. Handles portfolio data, market data, order submission.
- **analyst.py** — Claude analysis engine. Uses Anthropic SDK with tool_use for structured output.
- **risk.py** — Hard risk rules (position limits, stop-loss requirements, daily loss circuit breakers). These gate order submission.
- **journal.py** — SQLite-backed trade journal. Logs trades, portfolio snapshots, analysis history.
- **models.py** — Pydantic models for all data types (positions, orders, analysis, journal entries).
- **config.py** — Loads `config.toml`. All risk parameters and IBKR connection settings live there.
- **cli.py** — Click CLI. Primary interface. Commands: `portfolio`, `watch`, `ask`, `analyze`, `buy`, `sell`, `risk`, `journal`.
- **prompts/** — System prompts and tool definitions for Claude.

## Stack

- Python 3.12+ with **uv** (no pip, no conda)
- `ib_async` for IBKR (maintained fork of ib_insync)
- `anthropic` SDK direct (no LangChain)
- `click` + `rich` for CLI
- `pydantic` for data models
- `sqlite3` for journal/logging
- `httpx` for HTTP
- `ruff` for lint+format, `mypy` for types, `pytest` for tests

## Commands

```bash
uv run scry-trader portfolio          # positions, P&L, allocation
uv run scry-trader watch [SYMBOLS]    # live quotes
uv run scry-trader ask "question"     # free-form Claude query with portfolio context
uv run scry-trader analyze TICKER     # structured trade analysis
uv run scry-trader buy TICKER         # interactive buy flow
uv run scry-trader sell TICKER        # interactive sell flow
uv run scry-trader risk               # portfolio risk assessment
uv run scry-trader journal            # trade journal
uv run scry-trader journal --stats    # trade statistics
```

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
uv run pytest
```

## Design Constraints

- **Human-in-the-loop always.** No trade fires without explicit `click.confirm()`.
- **Risk rules are circuit breakers, not suggestions.** They block orders that violate limits.
- **Claude as analyst, not oracle.** It processes information and presents options. It does not predict.
- **No unnecessary abstractions.** Direct SDK calls, no framework wrappers.

## IBKR Ports

- 7497 = Paper TWS
- 4001 = Live Gateway
- 4002 = Paper Gateway

## Config

All settings in `config.toml`. Risk parameters, IBKR connection, watchlist, Claude model.
