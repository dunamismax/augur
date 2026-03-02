"""Tests for Pydantic models."""

from scry_trader.models import (
    AccountSummary,
    ConvictionLevel,
    Direction,
    OrderAction,
    OrderSpec,
    OrderType,
    Position,
    RiskLevel,
    TradeAnalysis,
)


class TestPosition:
    def test_pnl_percent_positive(self) -> None:
        pos = Position(
            symbol="AAPL",
            quantity=100,
            avg_cost=100.0,
            unrealized_pnl=500.0,
        )
        assert pos.pnl_percent == 5.0

    def test_pnl_percent_negative(self) -> None:
        pos = Position(
            symbol="AAPL",
            quantity=100,
            avg_cost=100.0,
            unrealized_pnl=-300.0,
        )
        assert pos.pnl_percent == -3.0

    def test_pnl_percent_zero_cost(self) -> None:
        pos = Position(symbol="AAPL", quantity=0, avg_cost=0.0)
        assert pos.pnl_percent == 0.0


class TestOrderSpec:
    def test_basic_order(self) -> None:
        order = OrderSpec(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
            stop_loss_price=140.0,
        )
        assert order.symbol == "AAPL"
        assert order.action == OrderAction.BUY
        assert order.limit_price == 150.0

    def test_bracket_order(self) -> None:
        order = OrderSpec(
            symbol="MSFT",
            action=OrderAction.BUY,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=300.0,
            take_profit_price=330.0,
            stop_loss_price=285.0,
        )
        assert order.take_profit_price == 330.0
        assert order.stop_loss_price == 285.0


class TestTradeAnalysis:
    def test_from_dict(self) -> None:
        analysis = TradeAnalysis(
            symbol="XLE",
            direction="long",  # type: ignore[arg-type]
            conviction="high",  # type: ignore[arg-type]
            risk_level="moderate",  # type: ignore[arg-type]
            bull_case="Oil supply disruption",
            bear_case="Ceasefire negotiations",
            reasoning="Geopolitical risk premium in energy",
            entry_price=85.0,
            target_price=95.0,
            stop_loss_price=80.0,
        )
        assert analysis.symbol == "XLE"
        assert analysis.direction == Direction.LONG
        assert analysis.conviction == ConvictionLevel.HIGH
        assert analysis.risk_level == RiskLevel.MODERATE


class TestAccountSummary:
    def test_defaults(self) -> None:
        summary = AccountSummary()
        assert summary.total_value == 0.0
        assert summary.positions == []
