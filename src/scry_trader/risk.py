"""Risk management rules and checks — circuit breakers, not suggestions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scry_trader.config import RiskConfig
    from scry_trader.models import AccountSummary, OrderSpec


@dataclass
class RiskCheckResult:
    """Result of a risk check."""

    passed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.passed


class RiskManager:
    """Enforces hard risk rules before order submission."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def check_order(self, order: OrderSpec, portfolio: AccountSummary) -> RiskCheckResult:
        """Run all risk checks against a proposed order. Returns pass/fail with details."""
        violations: list[str] = []
        warnings: list[str] = []

        # Paper trading gate
        if self.config.paper_trading:
            warnings.append("Paper trading mode is ON. Orders go to paper account.")

        # Stop-loss requirement
        if (
            self.config.require_stop_loss
            and order.stop_loss_price is None
            and order.stop_price is None
        ):
            violations.append(
                "Stop-loss required. Set stop_loss_price or stop_price on the order."
            )

        # Position size check
        if portfolio.total_value > 0:
            order_value = _estimate_order_value(order)
            position_pct = (order_value / portfolio.total_value) * 100

            if position_pct > self.config.max_position_pct:
                violations.append(
                    f"Position size {position_pct:.1f}% exceeds maximum "
                    f"{self.config.max_position_pct}% of portfolio "
                    f"(${order_value:,.0f} of ${portfolio.total_value:,.0f})"
                )

            # Check if adding this position creates excessive concentration
            existing = _get_existing_position_value(order.symbol, portfolio)
            total_in_symbol = existing + order_value
            total_pct = (total_in_symbol / portfolio.total_value) * 100
            if total_pct > self.config.max_position_pct:
                violations.append(
                    f"Total position in {order.symbol} would be {total_pct:.1f}% "
                    f"of portfolio (exceeds {self.config.max_position_pct}% limit)"
                )

        # Buying power check
        if portfolio.buying_power > 0:
            order_value = _estimate_order_value(order)
            if order_value > portfolio.buying_power:
                violations.append(
                    f"Order value ${order_value:,.0f} exceeds buying power "
                    f"${portfolio.buying_power:,.0f}"
                )

        # Leverage check
        if portfolio.total_value > 0 and portfolio.margin_used > 0:
            invested = portfolio.margin_used + _estimate_order_value(order)
            leverage = invested / portfolio.total_value
            if leverage > self.config.max_leverage:
                violations.append(
                    f"Leverage would be {leverage:.1f}x "
                    f"(exceeds {self.config.max_leverage}x limit)"
                )

        # Daily loss check
        if portfolio.total_value > 0 and portfolio.unrealized_pnl < 0:
            daily_loss_pct = abs(portfolio.unrealized_pnl) / portfolio.total_value * 100
            if daily_loss_pct > self.config.max_daily_loss_pct:
                violations.append(
                    f"Daily loss is {daily_loss_pct:.1f}% "
                    f"(exceeds {self.config.max_daily_loss_pct}% circuit breaker). "
                    "Consider closing positions before opening new ones."
                )

        # Naked options check
        if not self.config.allow_naked_options:
            # This would need contract type info — placeholder for Phase 3
            pass

        passed = len(violations) == 0
        return RiskCheckResult(passed=passed, violations=violations, warnings=warnings)

    def check_portfolio_health(self, portfolio: AccountSummary) -> RiskCheckResult:
        """Check overall portfolio health without a specific order."""
        violations: list[str] = []
        warnings: list[str] = []

        if portfolio.total_value <= 0:
            return RiskCheckResult(passed=True, warnings=["No portfolio data available."])

        # Check largest position
        largest_pct = 0.0
        largest_sym = ""
        for pos in portfolio.positions:
            pos_value = (
                abs(pos.market_value) if pos.market_value else abs(pos.quantity * pos.avg_cost)
            )
            pct = (pos_value / portfolio.total_value) * 100 if portfolio.total_value > 0 else 0.0
            if pct > largest_pct:
                largest_pct = pct
                largest_sym = pos.symbol

        if largest_pct > self.config.max_position_pct:
            warnings.append(
                f"Largest position: {largest_sym} at {largest_pct:.1f}% "
                f"(exceeds {self.config.max_position_pct}% limit)"
            )

        # Cash level check
        if portfolio.total_value > 0:
            cash_pct = (portfolio.cash / portfolio.total_value) * 100
            if cash_pct < 5:
                warnings.append(f"Low cash: {cash_pct:.1f}% of portfolio. Limited flexibility.")
            elif cash_pct > 80:
                warnings.append(f"High cash: {cash_pct:.1f}%. Capital is sitting idle.")

        # Daily P&L check
        if portfolio.total_value > 0 and portfolio.unrealized_pnl < 0:
            loss_pct = abs(portfolio.unrealized_pnl) / portfolio.total_value * 100
            if loss_pct > self.config.max_daily_loss_pct:
                violations.append(
                    f"Daily loss circuit breaker: down {loss_pct:.1f}% "
                    f"(limit: {self.config.max_daily_loss_pct}%). "
                    "Consider reducing exposure."
                )
            elif loss_pct > self.config.max_daily_loss_pct * 0.7:
                warnings.append(
                    f"Approaching daily loss limit: down {loss_pct:.1f}% "
                    f"(limit: {self.config.max_daily_loss_pct}%)"
                )

        passed = len(violations) == 0
        return RiskCheckResult(passed=passed, violations=violations, warnings=warnings)


def _estimate_order_value(order: OrderSpec) -> float:
    """Estimate the dollar value of an order."""
    price = order.limit_price or order.stop_price or 0.0
    return abs(order.quantity * price)


def _get_existing_position_value(symbol: str, portfolio: AccountSummary) -> float:
    """Get the current market value of an existing position in a symbol."""
    for pos in portfolio.positions:
        if pos.symbol == symbol:
            if pos.market_value:
                return abs(pos.market_value)
            return abs(pos.quantity * pos.avg_cost)
    return 0.0
