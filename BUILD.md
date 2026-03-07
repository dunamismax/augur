# augur — Build Tracker

**Status:** Functional CLI, still paper-first and not production-ready
**Last Updated:** 2026-03-07
**Branch:** `main`

---

## What This Repo Is

Narrow Python CLI for Interactive Brokers portfolio access, LLM trade review, risk checks, and journaling. Human-in-the-loop only. No web UI, no automation-first scope creep.

## Architecture Snapshot

```
augur/
├── src/augur/
│   ├── cli.py              # Click-based CLI entry point
│   ├── config.py           # TOML config loader (Pydantic validated)
│   ├── models.py           # Core domain models
│   ├── broker.py           # IBKR connection via ib-async
│   ├── analyst.py          # Claude-powered market analysis
│   ├── risk.py             # Position sizing and risk management
│   ├── journal.py          # Trade journal / decision log
│   └── prompts/
│       ├── system.py       # System prompt for Claude analyst
│       └── tools.py        # Tool definitions for Claude
├── tests/                  # pytest + pytest-asyncio
├── config.toml             # Runtime configuration
├── data/                   # Local data storage (market data, journals)
└── pyproject.toml          # uv-managed, Python 3.12+
```

**Stack:** Python 3.12+, uv, ib-async, anthropic SDK, Click, Rich, Pydantic, Ruff, mypy.

---

## Current Reality

### Working Commands

- `augur portfolio` shows account summary plus positions.
- `augur watch [SYMBOL ...]` returns quote snapshots. It is not a streaming terminal view.
- `augur analyze`, `augur ask`, `augur buy`, `augur sell`, `augur risk`, and `augur journal` all exist.
- `buy` and `sell` now capture an operator thesis / exit rationale, run a deterministic challenge review, and only then allow submission.
- Attached bracket exits only go out with net-new exposure; reduction / flip orders submit as a single parent order so the CLI does not quietly stage opposite-side child orders against an existing position.
- `journal` now surfaces the saved thesis so later review reflects the human decision, not just the model-generated order reason.
- `augur alerts` is still a placeholder.

### Portfolio Data Honesty

- `portfolio` now prefers IBKR's portfolio feed for per-position marked-to-market fields: market price, market value, unrealized P&L, realized P&L.
- If that feed is unavailable, the CLI falls back to raw position data from `IB.positions()`. That fallback preserves symbol, quantity, and average cost, but per-position marked-to-market fields may be zero.
- Account-level totals still come from IBKR account summary tags such as `NetLiquidation` and `BuyingPower`.

### Highest-Value Remaining Work

- Validate the challenge review flow against a real IBKR paper session and confirm the live quote used for pre-trade loss math matches operator expectations.
- Verify the portfolio-feed path against a real IBKR paper session and confirm multi-account scoping behaves as expected.
- Decide whether missing marked-to-market portfolio data should warn loudly or hard-fail instead of silently falling back.
- Improve connection resilience and state reporting; transient IBKR disconnects still fail the command.
- Keep the scope narrow: IBKR + LLM review + journal CLI. No web app, no autonomous execution.

---

## Verification Snapshot

```
UV_CACHE_DIR=/tmp/uv-cache-copy uv run --offline ruff check src tests
UV_CACHE_DIR=/tmp/uv-cache-copy uv run --offline --extra dev mypy src
UV_CACHE_DIR=/tmp/uv-cache-copy uv run --offline --extra dev pytest
```

Last verified: 2026-03-07

---

## Agent Instructions

- This is a **Python** project — use `uv` for all package management, never pip directly.
- Run the strongest available verification from the local environment. Prefer `uv` when dependencies are present.
- IBKR connection requires TWS or IB Gateway running — tests should mock `ib-async` connections.
- **Never place real trades** without explicit human confirmation. Paper account only during development.
- Rich library for terminal output — use `rich.console` and `rich.table` for the current CLI.
- `config.toml` holds runtime config — never commit secrets (API keys, account numbers) to it.
- Update this BUILD.md in the same commit as meaningful changes.
