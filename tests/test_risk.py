"""Tests for risk management module."""

from datetime import datetime

from augur.config import RiskConfig
from augur.models import (
    AccountSummary,
    Direction,
    OrderAction,
    OrderSpec,
    OrderType,
    Position,
    TradeJournalEntry,
    TradeOutcome,
)
from augur.risk import RiskManager, classify_order_exposure


def _default_config(**overrides: float | bool) -> RiskConfig:
    defaults: dict[str, float | bool] = {
        "max_position_pct": 40.0,
        "max_daily_loss_pct": 5.0,
        "max_leverage": 2.0,
        "require_stop_loss": True,
        "paper_trading": True,
    }
    defaults.update(overrides)
    return RiskConfig(**defaults)  # type: ignore[arg-type]


def _make_portfolio(
    total_value: float = 100_000,
    cash: float = 30_000,
    buying_power: float = 60_000,
    positions: list[Position] | None = None,
    unrealized_pnl: float = 0.0,
    margin_used: float = 0.0,
) -> AccountSummary:
    return AccountSummary(
        total_value=total_value,
        cash=cash,
        buying_power=buying_power,
        positions=positions or [],
        unrealized_pnl=unrealized_pnl,
        margin_used=margin_used,
    )


def _make_order(
    symbol: str = "AAPL",
    action: OrderAction = OrderAction.BUY,
    quantity: float = 100,
    limit_price: float = 150.0,
    stop_loss_price: float | None = 140.0,
    order_type: OrderType = OrderType.LIMIT,
) -> OrderSpec:
    return OrderSpec(
        symbol=symbol,
        action=action,
        quantity=quantity,
        limit_price=limit_price,
        stop_loss_price=stop_loss_price,
        order_type=order_type,
    )


class TestRiskCheckOrder:
    def test_valid_order_passes(self) -> None:
        rm = RiskManager(_default_config())
        order = _make_order(quantity=100, limit_price=150.0, stop_loss_price=140.0)
        portfolio = _make_portfolio(total_value=100_000)
        result = rm.check_order(order, portfolio)
        assert result.ok

    def test_missing_stop_loss_fails(self) -> None:
        rm = RiskManager(_default_config())
        order = _make_order(stop_loss_price=None)
        order.stop_price = None
        portfolio = _make_portfolio()
        result = rm.check_order(order, portfolio)
        assert not result.ok
        assert any("stop-loss" in v.lower() or "stop_loss" in v.lower() for v in result.violations)

    def test_stop_loss_not_required_when_disabled(self) -> None:
        rm = RiskManager(_default_config(require_stop_loss=False))
        order = _make_order(stop_loss_price=None)
        order.stop_price = None
        portfolio = _make_portfolio()
        result = rm.check_order(order, portfolio)
        assert result.ok

    def test_position_too_large_fails(self) -> None:
        rm = RiskManager(_default_config(max_position_pct=10.0))
        # Order for $15,000 in a $100k portfolio = 15% > 10% limit
        order = _make_order(quantity=100, limit_price=150.0)
        portfolio = _make_portfolio(total_value=100_000)
        result = rm.check_order(order, portfolio)
        assert not result.ok
        assert any("position size" in v.lower() for v in result.violations)

    def test_position_within_limit_passes(self) -> None:
        rm = RiskManager(_default_config(max_position_pct=40.0))
        # $15,000 order in $100k portfolio = 15% < 40%
        order = _make_order(quantity=100, limit_price=150.0)
        portfolio = _make_portfolio(total_value=100_000)
        result = rm.check_order(order, portfolio)
        assert result.ok

    def test_exceeds_buying_power_fails(self) -> None:
        rm = RiskManager(_default_config())
        order = _make_order(quantity=1000, limit_price=150.0)  # $150k order
        portfolio = _make_portfolio(buying_power=50_000, total_value=200_000)
        result = rm.check_order(order, portfolio)
        assert not result.ok
        assert any("buying power" in v.lower() for v in result.violations)

    def test_excessive_leverage_fails(self) -> None:
        rm = RiskManager(_default_config(max_leverage=2.0))
        order = _make_order(quantity=100, limit_price=150.0)  # $15k
        portfolio = _make_portfolio(
            total_value=50_000,
            margin_used=90_000,  # already at 1.8x, adding $15k = 2.1x
            buying_power=100_000,
        )
        result = rm.check_order(order, portfolio)
        assert not result.ok
        assert any("leverage" in v.lower() for v in result.violations)

    def test_daily_loss_circuit_breaker(self) -> None:
        rm = RiskManager(_default_config(max_daily_loss_pct=5.0))
        order = _make_order()
        portfolio = _make_portfolio(
            total_value=100_000,
            unrealized_pnl=-6_000,  # 6% loss > 5% limit
        )
        result = rm.check_order(order, portfolio)
        assert not result.ok
        assert any("daily loss" in v.lower() for v in result.violations)

    def test_paper_trading_warning(self) -> None:
        rm = RiskManager(_default_config(paper_trading=True))
        order = _make_order()
        portfolio = _make_portfolio()
        result = rm.check_order(order, portfolio)
        assert any("paper trading" in w.lower() for w in result.warnings)

    def test_market_order_without_reference_price_fails(self) -> None:
        rm = RiskManager(_default_config())
        order = OrderSpec(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            stop_loss_price=140.0,
        )
        portfolio = _make_portfolio(total_value=100_000)
        result = rm.check_order(order, portfolio)
        assert not result.ok
        assert any("reference_price" in v.lower() for v in result.violations)

    def test_market_order_with_reference_price_passes(self) -> None:
        rm = RiskManager(_default_config())
        order = OrderSpec(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            reference_price=150.0,
            stop_loss_price=140.0,
        )
        portfolio = _make_portfolio(total_value=100_000)
        result = rm.check_order(order, portfolio)
        assert result.ok

    def test_existing_position_concentration(self) -> None:
        rm = RiskManager(_default_config(max_position_pct=20.0))
        # Already holding $15k of AAPL, buying another $15k = $30k = 30% > 20%
        existing = Position(symbol="AAPL", quantity=100, avg_cost=150.0, market_value=15_000)
        order = _make_order(symbol="AAPL", quantity=100, limit_price=150.0)
        portfolio = _make_portfolio(total_value=100_000, positions=[existing])
        result = rm.check_order(order, portfolio)
        assert not result.ok
        assert any("total position" in v.lower() for v in result.violations)

    def test_sell_that_reduces_long_bypasses_entry_checks(self) -> None:
        rm = RiskManager(_default_config(max_position_pct=5.0, max_daily_loss_pct=2.0))
        order = _make_order(
            action=OrderAction.SELL,
            quantity=50,
            limit_price=150.0,
            stop_loss_price=None,
        )
        portfolio = _make_portfolio(
            positions=[Position(symbol="AAPL", quantity=100, avg_cost=150.0)],
            unrealized_pnl=-3_000,
        )
        result = rm.check_order(order, portfolio)
        assert result.ok
        assert any("reduces exposure" in warning.lower() for warning in result.warnings)

    def test_sell_flip_only_checks_excess_short(self) -> None:
        rm = RiskManager(_default_config(max_position_pct=5.0, paper_trading=False))
        order = _make_order(
            action=OrderAction.SELL,
            quantity=150,
            limit_price=150.0,
            stop_loss_price=160.0,
        )
        portfolio = _make_portfolio(
            total_value=100_000,
            positions=[Position(symbol="AAPL", quantity=100, avg_cost=150.0)],
        )
        result = rm.check_order(order, portfolio)
        assert not result.ok
        assert any("position size" in violation.lower() for violation in result.violations)

    def test_buy_to_cover_short_is_treated_as_reducing(self) -> None:
        rm = RiskManager(_default_config(max_daily_loss_pct=2.0))
        order = _make_order(
            action=OrderAction.BUY,
            quantity=30,
            limit_price=150.0,
            stop_loss_price=None,
        )
        portfolio = _make_portfolio(
            positions=[Position(symbol="AAPL", quantity=-40, avg_cost=150.0)],
            unrealized_pnl=-3_000,
        )
        result = rm.check_order(order, portfolio)
        assert result.ok
        assert any("reduces exposure" in warning.lower() for warning in result.warnings)


class TestPortfolioHealth:
    def test_healthy_portfolio(self) -> None:
        rm = RiskManager(_default_config())
        portfolio = _make_portfolio(
            total_value=100_000,
            cash=30_000,
            positions=[
                Position(symbol="AAPL", quantity=50, avg_cost=150.0, market_value=7_500),
                Position(symbol="MSFT", quantity=30, avg_cost=300.0, market_value=9_000),
            ],
        )
        result = rm.check_portfolio_health(portfolio)
        assert result.ok

    def test_low_cash_warning(self) -> None:
        rm = RiskManager(_default_config())
        portfolio = _make_portfolio(total_value=100_000, cash=3_000)
        result = rm.check_portfolio_health(portfolio)
        assert any("low cash" in w.lower() for w in result.warnings)

    def test_high_cash_warning(self) -> None:
        rm = RiskManager(_default_config())
        portfolio = _make_portfolio(total_value=100_000, cash=85_000)
        result = rm.check_portfolio_health(portfolio)
        assert any("high cash" in w.lower() for w in result.warnings)

    def test_daily_loss_violation(self) -> None:
        rm = RiskManager(_default_config(max_daily_loss_pct=5.0))
        portfolio = _make_portfolio(total_value=100_000, unrealized_pnl=-6_000)
        result = rm.check_portfolio_health(portfolio)
        assert not result.ok
        assert any("circuit breaker" in v.lower() for v in result.violations)

    def test_approaching_daily_loss_warning(self) -> None:
        rm = RiskManager(_default_config(max_daily_loss_pct=5.0))
        # 4% loss > 3.5% threshold (70% of 5%)
        portfolio = _make_portfolio(total_value=100_000, unrealized_pnl=-4_000)
        result = rm.check_portfolio_health(portfolio)
        assert any("approaching" in w.lower() for w in result.warnings)

    def test_empty_portfolio(self) -> None:
        rm = RiskManager(_default_config())
        portfolio = _make_portfolio(total_value=0)
        result = rm.check_portfolio_health(portfolio)
        assert result.ok


class TestTradeChallenge:
    def test_challenge_trade_computes_loss_and_reward(self) -> None:
        rm = RiskManager(_default_config())
        order = _make_order(
            quantity=100,
            limit_price=150.0,
            stop_loss_price=145.0,
        )
        order.take_profit_price = 165.0

        challenge = rm.challenge_trade(
            order,
            _make_portfolio(total_value=100_000),
            "Breakout with catalyst and clear invalidation below support.",
            [],
            now=datetime(2026, 3, 7, 9, 30),
        )

        assert challenge.blockers == []
        assert challenge.max_loss == 500.0
        assert challenge.max_loss_pct == 0.5
        assert challenge.reward_risk_ratio == 3.0
        assert challenge.post_trade_position_pct == 15.0

    def test_challenge_trade_requires_operator_thesis(self) -> None:
        rm = RiskManager(_default_config())
        challenge = rm.challenge_trade(
            _make_order(),
            _make_portfolio(),
            "   ",
            [],
            now=datetime(2026, 3, 7, 9, 30),
        )

        assert challenge.blockers
        assert "thesis" in challenge.blockers[0].lower()

    def test_challenge_trade_flags_recent_loss_and_adding_to_loser(self) -> None:
        rm = RiskManager(_default_config())
        order = _make_order(quantity=50, limit_price=140.0, stop_loss_price=132.0)
        history = [
            TradeJournalEntry(
                ticker="AAPL",
                direction=Direction.LONG,
                entry_price=150.0,
                exit_price=142.0,
                shares=25,
                entry_date=datetime(2026, 3, 1),
                exit_date=datetime(2026, 3, 5),
                outcome=TradeOutcome.LOSS,
            )
        ]
        portfolio = _make_portfolio(
            positions=[
                Position(
                    symbol="AAPL",
                    quantity=100,
                    avg_cost=150.0,
                    market_value=14_000.0,
                    unrealized_pnl=-1_000.0,
                )
            ]
        )

        challenge = rm.challenge_trade(
            order,
            portfolio,
            "Adding on planned retest with defined invalidation and catalyst.",
            history,
            now=datetime(2026, 3, 7, 9, 30),
        )

        warning_text = " ".join(challenge.warnings).lower()
        prompt_text = " ".join(challenge.prompts).lower()
        assert "adding to a losing" in warning_text
        assert "losing journal entry" in warning_text
        assert "waiting for strength" in prompt_text
        assert "different from that losing setup" in prompt_text

    def test_challenge_trade_does_not_flag_revenge_trade_on_pure_exit(self) -> None:
        rm = RiskManager(_default_config())
        order = _make_order(
            action=OrderAction.SELL,
            quantity=50,
            limit_price=140.0,
            stop_loss_price=145.0,
        )
        history = [
            TradeJournalEntry(
                ticker="AAPL",
                direction=Direction.LONG,
                entry_price=150.0,
                exit_price=142.0,
                shares=25,
                entry_date=datetime(2026, 3, 1),
                exit_date=datetime(2026, 3, 5),
                outcome=TradeOutcome.LOSS,
            )
        ]
        portfolio = _make_portfolio(
            positions=[Position(symbol="AAPL", quantity=100, avg_cost=150.0)]
        )

        challenge = rm.challenge_trade(
            order,
            portfolio,
            "Planned trim into resistance after the thesis weakened.",
            history,
            now=datetime(2026, 3, 7, 9, 30),
        )

        warning_text = " ".join(challenge.warnings).lower()
        prompt_text = " ".join(challenge.prompts).lower()
        assert "losing journal entry" not in warning_text
        assert "revenge trade" not in warning_text
        assert "different from that losing setup" not in prompt_text


class TestExposureClassification:
    def test_sell_flip_is_split_between_reduce_and_open(self) -> None:
        portfolio = _make_portfolio(
            positions=[Position(symbol="AAPL", quantity=100, avg_cost=150.0)]
        )
        order = _make_order(
            action=OrderAction.SELL,
            quantity=150,
            limit_price=150.0,
            stop_loss_price=160.0,
        )
        exposure = classify_order_exposure(order, portfolio)
        assert exposure.reducing_quantity == 100
        assert exposure.opening_quantity == 50
        assert exposure.reducing_direction == Direction.LONG
        assert exposure.opening_direction == Direction.SHORT
