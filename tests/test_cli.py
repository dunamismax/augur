"""Integration-style tests for the trade flow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from augur import cli
from augur.config import AppConfig, DatabaseConfig, IBKRConfig, RiskConfig
from augur.journal import Journal
from augur.models import (
    AccountSummary,
    Direction,
    OrderAction,
    OrderResult,
    OrderSpec,
    OrderType,
    Position,
    TradeOutcome,
    WatchlistItem,
)

if TYPE_CHECKING:
    from pathlib import Path


class FakeAnalyst:
    order_spec: OrderSpec

    def __init__(self, claude_config: object, risk_config: object) -> None:
        self.claude_config = claude_config
        self.risk_config = risk_config

    def construct_order(
        self, symbol: str, direction: str, portfolio: AccountSummary | None = None
    ) -> OrderSpec:
        return self.order_spec


class FakeBroker:
    def __init__(self, config: object) -> None:
        self.config = config


def _app_config(db_path: Path) -> AppConfig:
    return AppConfig(
        ibkr=IBKRConfig(account="DU123"),
        risk=RiskConfig(paper_trading=False),
        database=DatabaseConfig(path=str(db_path)),
    )


async def _async_return[T](value: T) -> T:
    return value


@pytest.fixture(autouse=True)
def confirm_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.click, "confirm", lambda prompt: True)
    monkeypatch.setattr(
        cli.click,
        "prompt",
        lambda text, **kwargs: (
            "REVIEWED"
            if "Type REVIEWED" in text
            else "Fresh catalyst with defined invalidation and exit plan."
        ),
    )


def test_buy_flow_overrides_llm_symbol_and_side_and_logs_fill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "trades.db"
    monkeypatch.setattr(cli, "_load", lambda: _app_config(db_path))
    FakeAnalyst.order_spec = OrderSpec.model_construct(
        symbol="MSFT",
        action=OrderAction.SELL,
        quantity=10,
        order_type=OrderType.MARKET,
        reference_price=101.0,
        stop_loss_price=95.0,
        reason="buy test",
    )
    monkeypatch.setattr(cli, "Analyst", FakeAnalyst)
    monkeypatch.setattr(cli, "Broker", FakeBroker)
    monkeypatch.setattr(
        cli,
        "_get_portfolio_and_quote",
        lambda broker, symbol: _async_return(
            (
                AccountSummary(total_value=100_000, buying_power=100_000),
                WatchlistItem(symbol=symbol, last_price=101.0),
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "_submit_order",
        lambda broker, spec: _async_return(
            OrderResult(
                order_id=1,
                symbol=spec.symbol,
                action=spec.action,
                quantity=spec.quantity,
                status="Filled",
                filled_price=101.25,
                filled_quantity=10,
            )
        ),
    )

    cli._trade_flow("AAPL", OrderAction.BUY, shares=None, limit_price=None)

    journal = Journal(db_path)
    open_trades = journal.get_open_trades()
    assert len(open_trades) == 1
    assert open_trades[0].ticker == "AAPL"
    assert open_trades[0].direction == Direction.LONG
    assert open_trades[0].shares == 10
    assert open_trades[0].thesis == "Fresh catalyst with defined invalidation and exit plan."
    assert open_trades[0].claude_analysis == "buy test"


def test_sell_flow_partial_fill_closes_only_filled_lot_quantity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "trades.db"
    journal = Journal(db_path)
    journal.add_trade(
        cli.TradeJournalEntry(
            ticker="AAPL",
            direction=Direction.LONG,
            entry_price=90.0,
            shares=100,
            thesis="existing lot",
        )
    )

    monkeypatch.setattr(cli, "_load", lambda: _app_config(db_path))
    FakeAnalyst.order_spec = OrderSpec(
        symbol="AAPL",
        action=OrderAction.SELL,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=105.0,
        stop_loss_price=110.0,
        reason="partial exit",
    )
    monkeypatch.setattr(cli, "Analyst", FakeAnalyst)
    monkeypatch.setattr(cli, "Broker", FakeBroker)
    monkeypatch.setattr(
        cli,
        "_get_portfolio_and_quote",
        lambda broker, symbol: _async_return(
            (
                AccountSummary(
                    total_value=100_000,
                    buying_power=100_000,
                    positions=[Position(symbol=symbol, quantity=100, avg_cost=90.0)],
                ),
                WatchlistItem(symbol=symbol, last_price=105.0),
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "_submit_order",
        lambda broker, spec: _async_return(
            OrderResult(
                order_id=2,
                symbol=spec.symbol,
                action=spec.action,
                quantity=spec.quantity,
                status="PartiallyFilled",
                filled_price=105.0,
                filled_quantity=40,
            )
        ),
    )

    cli._trade_flow("AAPL", OrderAction.SELL, shares=None, limit_price=None)

    open_lots = Journal(db_path).get_open_lots(ticker="AAPL", direction=Direction.LONG)
    assert len(open_lots) == 1
    assert open_lots[0].shares == 60
    closed = [
        trade
        for trade in Journal(db_path).get_trades_by_ticker("AAPL")
        if trade.outcome != TradeOutcome.OPEN
    ]
    assert len(closed) == 1
    assert closed[0].shares == 40
    assert "Operator rationale:" in closed[0].notes
    assert "Claude rationale: partial exit" in closed[0].notes


def test_sell_flow_can_flip_from_long_to_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "trades.db"
    journal = Journal(db_path)
    journal.add_trade(
        cli.TradeJournalEntry(
            ticker="AAPL",
            direction=Direction.LONG,
            entry_price=95.0,
            shares=100,
            thesis="swing",
        )
    )

    monkeypatch.setattr(cli, "_load", lambda: _app_config(db_path))
    FakeAnalyst.order_spec = OrderSpec(
        symbol="AAPL",
        action=OrderAction.SELL,
        quantity=150,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
        stop_loss_price=104.0,
        reason="flip short",
    )
    monkeypatch.setattr(cli, "Analyst", FakeAnalyst)
    monkeypatch.setattr(cli, "Broker", FakeBroker)
    monkeypatch.setattr(
        cli,
        "_get_portfolio_and_quote",
        lambda broker, symbol: _async_return(
            (
                AccountSummary(
                    total_value=100_000,
                    buying_power=100_000,
                    positions=[Position(symbol=symbol, quantity=100, avg_cost=95.0)],
                ),
                WatchlistItem(symbol=symbol, last_price=100.0),
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "_submit_order",
        lambda broker, spec: _async_return(
            OrderResult(
                order_id=3,
                symbol=spec.symbol,
                action=spec.action,
                quantity=spec.quantity,
                status="Filled",
                filled_price=100.0,
                filled_quantity=150,
            )
        ),
    )

    cli._trade_flow("AAPL", OrderAction.SELL, shares=None, limit_price=None)

    journal = Journal(db_path)
    short_lots = journal.get_open_lots(ticker="AAPL", direction=Direction.SHORT)
    assert len(short_lots) == 1
    assert short_lots[0].shares == 50


def test_bracket_flow_uses_parent_fill_for_journal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "trades.db"
    monkeypatch.setattr(cli, "_load", lambda: _app_config(db_path))
    FakeAnalyst.order_spec = OrderSpec(
        symbol="AAPL",
        action=OrderAction.BUY,
        quantity=20,
        order_type=OrderType.MARKET,
        reference_price=101.0,
        take_profit_price=110.0,
        stop_loss_price=96.0,
        reason="bracket entry",
    )
    monkeypatch.setattr(cli, "Analyst", FakeAnalyst)
    monkeypatch.setattr(cli, "Broker", FakeBroker)
    monkeypatch.setattr(
        cli,
        "_get_portfolio_and_quote",
        lambda broker, symbol: _async_return(
            (
                AccountSummary(total_value=100_000, buying_power=100_000),
                WatchlistItem(symbol=symbol, last_price=101.0),
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "_submit_bracket",
        lambda broker, spec: _async_return(
            [
                OrderResult(
                    order_id=10,
                    symbol=spec.symbol,
                    action=spec.action,
                    quantity=spec.quantity,
                    status="Filled",
                    filled_price=101.5,
                    filled_quantity=20,
                ),
                OrderResult(
                    order_id=11,
                    symbol=spec.symbol,
                    action=OrderAction.SELL,
                    quantity=spec.quantity,
                    status="Submitted",
                ),
                OrderResult(
                    order_id=12,
                    symbol=spec.symbol,
                    action=OrderAction.SELL,
                    quantity=spec.quantity,
                    status="Submitted",
                ),
            ]
        ),
    )

    cli._trade_flow("AAPL", OrderAction.BUY, shares=None, limit_price=None)

    open_trades = Journal(db_path).get_open_trades()
    assert len(open_trades) == 1
    assert open_trades[0].entry_price == 101.5
    assert open_trades[0].shares == 20


def test_reducing_sell_with_bracket_levels_submits_single_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "trades.db"
    journal = Journal(db_path)
    journal.add_trade(
        cli.TradeJournalEntry(
            ticker="AAPL",
            direction=Direction.LONG,
            entry_price=95.0,
            shares=100,
            thesis="core position",
        )
    )

    monkeypatch.setattr(cli, "_load", lambda: _app_config(db_path))
    FakeAnalyst.order_spec = OrderSpec(
        symbol="AAPL",
        action=OrderAction.SELL,
        quantity=150,
        order_type=OrderType.MARKET,
        reference_price=100.0,
        take_profit_price=94.0,
        stop_loss_price=105.0,
        reason="flip with attached exits",
    )
    monkeypatch.setattr(cli, "Analyst", FakeAnalyst)
    monkeypatch.setattr(cli, "Broker", FakeBroker)
    monkeypatch.setattr(
        cli,
        "_get_portfolio_and_quote",
        lambda broker, symbol: _async_return(
            (
                AccountSummary(
                    total_value=100_000,
                    buying_power=100_000,
                    positions=[Position(symbol=symbol, quantity=100, avg_cost=95.0)],
                ),
                WatchlistItem(symbol=symbol, last_price=100.0),
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "_submit_bracket",
        lambda broker, spec: pytest.fail("reducing or flip orders must not submit brackets"),
    )
    monkeypatch.setattr(
        cli,
        "_submit_order",
        lambda broker, spec: _async_return(
            OrderResult(
                order_id=13,
                symbol=spec.symbol,
                action=spec.action,
                quantity=spec.quantity,
                status="Filled",
                filled_price=100.0,
                filled_quantity=150,
            )
        ),
    )

    cli._trade_flow("AAPL", OrderAction.SELL, shares=None, limit_price=None)

    updated_journal = Journal(db_path)
    short_lots = updated_journal.get_open_lots(ticker="AAPL", direction=Direction.SHORT)
    assert len(short_lots) == 1
    assert short_lots[0].shares == 50


def test_rejected_order_does_not_touch_journal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "trades.db"
    monkeypatch.setattr(cli, "_load", lambda: _app_config(db_path))
    FakeAnalyst.order_spec = OrderSpec(
        symbol="AAPL",
        action=OrderAction.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        reference_price=101.0,
        stop_loss_price=95.0,
        reason="reject",
    )
    monkeypatch.setattr(cli, "Analyst", FakeAnalyst)
    monkeypatch.setattr(cli, "Broker", FakeBroker)
    monkeypatch.setattr(
        cli,
        "_get_portfolio_and_quote",
        lambda broker, symbol: _async_return(
            (
                AccountSummary(total_value=100_000, buying_power=100_000),
                WatchlistItem(symbol=symbol, last_price=101.0),
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "_submit_order",
        lambda broker, spec: _async_return(
            OrderResult(
                order_id=4,
                symbol=spec.symbol,
                action=spec.action,
                quantity=spec.quantity,
                status="Inactive",
                filled_quantity=0,
            )
        ),
    )

    cli._trade_flow("AAPL", OrderAction.BUY, shares=None, limit_price=None)

    assert Journal(db_path).get_recent_trades() == []


def test_delayed_fill_does_not_journal_until_fill_arrives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "trades.db"
    monkeypatch.setattr(cli, "_load", lambda: _app_config(db_path))
    FakeAnalyst.order_spec = OrderSpec(
        symbol="AAPL",
        action=OrderAction.SELL,
        quantity=10,
        order_type=OrderType.MARKET,
        reference_price=101.0,
        stop_loss_price=105.0,
        reason="delayed",
    )
    monkeypatch.setattr(cli, "Analyst", FakeAnalyst)
    monkeypatch.setattr(cli, "Broker", FakeBroker)
    monkeypatch.setattr(
        cli,
        "_get_portfolio_and_quote",
        lambda broker, symbol: _async_return(
            (
                AccountSummary(total_value=100_000, buying_power=100_000),
                WatchlistItem(symbol=symbol, last_price=101.0),
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "_submit_order",
        lambda broker, spec: _async_return(
            OrderResult(
                order_id=5,
                symbol=spec.symbol,
                action=spec.action,
                quantity=spec.quantity,
                status="Submitted",
                filled_quantity=0,
            )
        ),
    )

    cli._trade_flow("AAPL", OrderAction.SELL, shares=None, limit_price=None)

    assert Journal(db_path).get_recent_trades() == []
