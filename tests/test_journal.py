"""Tests for trade journal module."""

from datetime import datetime
from pathlib import Path

import pytest

from scry_trader.journal import Journal
from scry_trader.models import (
    AnalysisLogEntry,
    Direction,
    PortfolioSnapshot,
    TradeJournalEntry,
    TradeOutcome,
)


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal(tmp_path / "test_trades.db")


@pytest.fixture
def sample_trade() -> TradeJournalEntry:
    return TradeJournalEntry(
        ticker="AAPL",
        direction=Direction.LONG,
        entry_price=150.0,
        shares=100,
        entry_date=datetime(2026, 3, 1),
        thesis="Strong earnings momentum, iPhone cycle peak",
        tags=["tech", "swing"],
    )


class TestTradeOperations:
    def test_add_and_get_trade(self, journal: Journal, sample_trade: TradeJournalEntry) -> None:
        trade_id = journal.add_trade(sample_trade)
        assert trade_id > 0

        trade = journal.get_trade(trade_id)
        assert trade is not None
        assert trade.ticker == "AAPL"
        assert trade.direction == Direction.LONG
        assert trade.entry_price == 150.0
        assert trade.shares == 100
        assert trade.outcome == TradeOutcome.OPEN

    def test_close_trade_win(self, journal: Journal, sample_trade: TradeJournalEntry) -> None:
        trade_id = journal.add_trade(sample_trade)
        closed = journal.close_trade(trade_id, exit_price=170.0, notes="Hit target")

        assert closed is not None
        assert closed.outcome == TradeOutcome.WIN
        assert closed.pnl == 2000.0  # (170 - 150) * 100
        assert closed.exit_price == 170.0

    def test_close_trade_loss(self, journal: Journal, sample_trade: TradeJournalEntry) -> None:
        trade_id = journal.add_trade(sample_trade)
        closed = journal.close_trade(trade_id, exit_price=140.0)

        assert closed is not None
        assert closed.outcome == TradeOutcome.LOSS
        assert closed.pnl == -1000.0  # (140 - 150) * 100

    def test_close_short_trade(self, journal: Journal) -> None:
        short_trade = TradeJournalEntry(
            ticker="TSLA",
            direction=Direction.SHORT,
            entry_price=200.0,
            shares=50,
            entry_date=datetime(2026, 3, 1),
            thesis="Overvalued",
        )
        trade_id = journal.add_trade(short_trade)
        closed = journal.close_trade(trade_id, exit_price=180.0)

        assert closed is not None
        assert closed.outcome == TradeOutcome.WIN
        assert closed.pnl == 1000.0  # (200 - 180) * 50

    def test_get_nonexistent_trade(self, journal: Journal) -> None:
        assert journal.get_trade(999) is None

    def test_close_nonexistent_trade(self, journal: Journal) -> None:
        assert journal.close_trade(999, exit_price=100.0) is None

    def test_get_open_trades(self, journal: Journal, sample_trade: TradeJournalEntry) -> None:
        journal.add_trade(sample_trade)
        # Add and close another trade
        t2 = TradeJournalEntry(
            ticker="MSFT", direction=Direction.LONG, entry_price=300.0, shares=10, thesis="test"
        )
        trade_id = journal.add_trade(t2)
        journal.close_trade(trade_id, exit_price=310.0)

        open_trades = journal.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0].ticker == "AAPL"

    def test_get_recent_trades(self, journal: Journal) -> None:
        for i in range(5):
            t = TradeJournalEntry(
                ticker=f"SYM{i}",
                direction=Direction.LONG,
                entry_price=float(i * 10),
                shares=10,
                thesis=f"Trade {i}",
            )
            journal.add_trade(t)

        recent = journal.get_recent_trades(limit=3)
        assert len(recent) == 3
        # Should be newest first
        assert recent[0].ticker == "SYM4"

    def test_get_trades_by_ticker(self, journal: Journal) -> None:
        for ticker in ["AAPL", "AAPL", "MSFT"]:
            t = TradeJournalEntry(
                ticker=ticker,
                direction=Direction.LONG,
                entry_price=100.0,
                shares=10,
                thesis="test",
            )
            journal.add_trade(t)

        aapl_trades = journal.get_trades_by_ticker("AAPL")
        assert len(aapl_trades) == 2

    def test_trade_tags(self, journal: Journal, sample_trade: TradeJournalEntry) -> None:
        trade_id = journal.add_trade(sample_trade)
        trade = journal.get_trade(trade_id)
        assert trade is not None
        assert "tech" in trade.tags
        assert "swing" in trade.tags


class TestTradeStats:
    def test_empty_stats(self, journal: Journal) -> None:
        stats = journal.get_trade_stats()
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0

    def test_stats_calculation(self, journal: Journal) -> None:
        # Add a winning trade
        t1 = TradeJournalEntry(
            ticker="AAPL", direction=Direction.LONG, entry_price=100.0, shares=100, thesis="test"
        )
        id1 = journal.add_trade(t1)
        journal.close_trade(id1, exit_price=120.0)  # +$2000

        # Add a losing trade
        t2 = TradeJournalEntry(
            ticker="TSLA", direction=Direction.LONG, entry_price=200.0, shares=50, thesis="test"
        )
        id2 = journal.add_trade(t2)
        journal.close_trade(id2, exit_price=190.0)  # -$500

        # Add an open trade
        t3 = TradeJournalEntry(
            ticker="MSFT", direction=Direction.LONG, entry_price=300.0, shares=10, thesis="test"
        )
        journal.add_trade(t3)

        stats = journal.get_trade_stats()
        assert stats["total_trades"] == 3
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["open"] == 1
        assert stats["win_rate"] == 50.0
        assert stats["total_pnl"] == 1500.0  # 2000 - 500
        assert stats["avg_win"] == 2000.0
        assert stats["avg_loss"] == -500.0


class TestPortfolioSnapshots:
    def test_save_and_get_snapshot(self, journal: Journal) -> None:
        snap = PortfolioSnapshot(
            total_value=100_000,
            cash=30_000,
            positions_json='[{"symbol": "AAPL", "qty": 100}]',
            daily_pnl=500.0,
        )
        snap_id = journal.save_snapshot(snap)
        assert snap_id > 0

        snapshots = journal.get_recent_snapshots(limit=1)
        assert len(snapshots) == 1
        assert snapshots[0].total_value == 100_000


class TestAnalysisLog:
    def test_log_analysis(self, journal: Journal) -> None:
        entry = AnalysisLogEntry(
            query="What's my risk?",
            response="Your portfolio is well-diversified.",
            tokens_used=500,
        )
        log_id = journal.log_analysis(entry)
        assert log_id > 0
