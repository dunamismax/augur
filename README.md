# Augur

AI-assisted, human-directed trading system built on Claude and Interactive Brokers.

CLI-first. Every trade requires human confirmation. Claude analyzes, sizes, and recommends — you decide.

## Stack

- **Python 3.12+** with strict mypy
- **Interactive Brokers** via ib-async
- **Claude** (Anthropic) for market analysis, trade construction, and risk assessment
- **Click + Rich** for the CLI
- **Pydantic** for typed models
- **SQLite** for trade journal persistence

## Commands

```
augur portfolio     # Positions, P&L, allocation
augur watch [SYM]   # Live quotes for watchlist or specific symbols
augur analyze TICKER # Deep Claude analysis with entry/exit/risk levels
augur ask "question" # Free-form question with portfolio context
augur buy TICKER     # Interactive buy flow with Claude sizing
augur sell TICKER    # Interactive sell flow
augur risk           # Portfolio risk assessment (rules + Claude)
augur journal        # Trade journal with stats and filtering
```

## Architecture

```
src/augur/
├── cli.py        # Click CLI — primary interface
├── broker.py     # IBKR connection, orders, quotes
├── analyst.py    # Claude integration — analysis, sizing, risk
├── risk.py       # Rule-based risk management (pre-trade + portfolio)
├── journal.py    # SQLite trade journal
├── models.py     # Pydantic models (orders, positions, analysis)
├── config.py     # TOML config loader
└── prompts/      # System prompts and tool definitions for Claude
```

## Setup

```bash
# Clone
git clone git@github.com:dunamismax/augur.git
cd augur

# Install (requires uv)
uv sync --all-extras

# NOTE: If upgrading from the old scry-trader repo name, delete and recreate
# your venv — shebangs will point to the old path and break all tooling.
#   rm -rf .venv && uv sync --all-extras

# Configure
cp config.toml.example config.toml
# Edit config.toml with your IBKR and Anthropic API settings

# Run
uv run augur portfolio
```

## Development

```bash
uv sync --all-extras
uv run pytest
uv run mypy src/
uv run ruff check src/ tests/
```

## Design Principles

- **Human-in-the-loop.** Claude recommends, you confirm. No autonomous trading.
- **Risk-first.** Rule-based guardrails run before every order. Portfolio health checks are always available.
- **CLI-native.** No web UI. Terminal is the interface. Rich tables, colored P&L, structured output.
- **Journaled.** Every trade and analysis is logged for review.

## License

MIT
