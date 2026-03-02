"""Trade journal — SQLite-backed trade logging and analysis history."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from scry_trader.models import (
    AnalysisLogEntry,
    Direction,
    PortfolioSnapshot,
    TradeJournalEntry,
    TradeOutcome,
)

SCHEMA = """\
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL,
    exit_price REAL,
    shares REAL,
    entry_date TEXT,
    exit_date TEXT,
    thesis TEXT,
    claude_analysis TEXT,
    outcome TEXT DEFAULT 'open',
    pnl REAL DEFAULT 0.0,
    notes TEXT,
    tags TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    total_value REAL,
    cash REAL,
    positions_json TEXT,
    daily_pnl REAL
);

CREATE TABLE IF NOT EXISTS analysis_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    query TEXT,
    context_json TEXT,
    response TEXT,
    tokens_used INTEGER DEFAULT 0
);
"""


class Journal:
    """SQLite-backed trade journal."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # --- Trade Operations ---

    def add_trade(self, entry: TradeJournalEntry) -> int:
        """Record a new trade. Returns the trade ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO trades
                (ticker, direction, entry_price, exit_price, shares,
                 entry_date, exit_date, thesis, claude_analysis, outcome, pnl, notes, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.ticker,
                    entry.direction.value,
                    entry.entry_price,
                    entry.exit_price,
                    entry.shares,
                    entry.entry_date.isoformat() if entry.entry_date else None,
                    entry.exit_date.isoformat() if entry.exit_date else None,
                    entry.thesis,
                    entry.claude_analysis,
                    entry.outcome.value,
                    entry.pnl,
                    entry.notes,
                    ",".join(entry.tags) if entry.tags else "",
                ),
            )
            return cursor.lastrowid or 0

    def close_trade(
        self, trade_id: int, exit_price: float, notes: str = ""
    ) -> TradeJournalEntry | None:
        """Close an open trade with exit price and calculate P&L."""
        trade = self.get_trade(trade_id)
        if trade is None:
            return None

        entry_price = trade.entry_price or 0.0
        if trade.direction == Direction.LONG:
            pnl = (exit_price - entry_price) * trade.shares
        else:
            pnl = (entry_price - exit_price) * trade.shares

        outcome = TradeOutcome.WIN if pnl > 0 else TradeOutcome.LOSS

        with self._connect() as conn:
            conn.execute(
                """UPDATE trades
                SET exit_price = ?, exit_date = ?, outcome = ?, pnl = ?,
                    notes = COALESCE(notes || '\n' || ?, notes)
                WHERE id = ?""",
                (exit_price, datetime.now().isoformat(), outcome.value, pnl, notes, trade_id),
            )

        return self.get_trade(trade_id)

    def get_trade(self, trade_id: int) -> TradeJournalEntry | None:
        """Get a single trade by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
            if row is None:
                return None
            return _row_to_trade(row)

    def get_open_trades(self) -> list[TradeJournalEntry]:
        """Get all open trades."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE outcome = 'open' ORDER BY entry_date DESC"
            ).fetchall()
            return [_row_to_trade(r) for r in rows]

    def get_recent_trades(self, limit: int = 20) -> list[TradeJournalEntry]:
        """Get recent trades, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [_row_to_trade(r) for r in rows]

    def get_trades_by_ticker(self, ticker: str) -> list[TradeJournalEntry]:
        """Get all trades for a specific ticker."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE ticker = ? ORDER BY entry_date DESC", (ticker,)
            ).fetchall()
            return [_row_to_trade(r) for r in rows]

    def get_trade_stats(self) -> dict[str, float | int]:
        """Get aggregate trade statistics."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            wins = conn.execute("SELECT COUNT(*) FROM trades WHERE outcome = 'win'").fetchone()[0]
            losses = conn.execute("SELECT COUNT(*) FROM trades WHERE outcome = 'loss'").fetchone()[
                0
            ]
            open_count = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE outcome = 'open'"
            ).fetchone()[0]
            total_pnl = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE outcome != 'open'"
            ).fetchone()[0]
            avg_win = conn.execute(
                "SELECT COALESCE(AVG(pnl), 0) FROM trades WHERE outcome = 'win'"
            ).fetchone()[0]
            avg_loss = conn.execute(
                "SELECT COALESCE(AVG(pnl), 0) FROM trades WHERE outcome = 'loss'"
            ).fetchone()[0]

        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "open": open_count,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }

    # --- Portfolio Snapshots ---

    def save_snapshot(self, snapshot: PortfolioSnapshot) -> int:
        """Save a portfolio snapshot."""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO portfolio_snapshots
                (timestamp, total_value, cash, positions_json, daily_pnl)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    snapshot.timestamp.isoformat(),
                    snapshot.total_value,
                    snapshot.cash,
                    snapshot.positions_json,
                    snapshot.daily_pnl,
                ),
            )
            return cursor.lastrowid or 0

    def get_recent_snapshots(self, limit: int = 30) -> list[PortfolioSnapshot]:
        """Get recent portfolio snapshots."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM portfolio_snapshots ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            results: list[PortfolioSnapshot] = []
            for r in rows:
                results.append(
                    PortfolioSnapshot(
                        id=r[0],
                        timestamp=datetime.fromisoformat(r[1]),
                        total_value=r[2] or 0.0,
                        cash=r[3] or 0.0,
                        positions_json=r[4] or "",
                        daily_pnl=r[5] or 0.0,
                    )
                )
            return results

    # --- Analysis Log ---

    def log_analysis(self, entry: AnalysisLogEntry) -> int:
        """Log a Claude analysis query and response."""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO analysis_log
                (timestamp, query, context_json, response, tokens_used)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    entry.timestamp.isoformat(),
                    entry.query,
                    entry.context_json,
                    entry.response,
                    entry.tokens_used,
                ),
            )
            return cursor.lastrowid or 0


def _row_to_trade(row: tuple) -> TradeJournalEntry:  # type: ignore[type-arg]
    """Convert a database row to a TradeJournalEntry."""
    return TradeJournalEntry(
        id=row[0],
        ticker=row[1],
        direction=Direction(row[2]),
        entry_price=row[3],
        exit_price=row[4],
        shares=row[5] or 0.0,
        entry_date=datetime.fromisoformat(row[6]) if row[6] else None,
        exit_date=datetime.fromisoformat(row[7]) if row[7] else None,
        thesis=row[8] or "",
        claude_analysis=row[9] or "",
        outcome=TradeOutcome(row[10]) if row[10] else TradeOutcome.OPEN,
        pnl=row[11] or 0.0,
        notes=row[12] or "",
        tags=row[13].split(",") if row[13] else [],
    )
