"""Pydantic models for positions, orders, signals, and analysis."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# --- Enums ---


class Direction(StrEnum):
    LONG = "long"
    SHORT = "short"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class TradeOutcome(StrEnum):
    WIN = "win"
    LOSS = "loss"
    OPEN = "open"


class ConvictionLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class RiskLevel(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


# --- Portfolio & Market Models ---


class Position(BaseModel):
    symbol: str
    quantity: float
    avg_cost: float
    market_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    account: str = ""

    @property
    def pnl_percent(self) -> float:
        cost_basis = self.avg_cost * abs(self.quantity)
        if cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / cost_basis) * 100


class AccountSummary(BaseModel):
    total_value: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0
    positions: list[Position] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class WatchlistItem(BaseModel):
    symbol: str
    last_price: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0
    volume: int = 0
    bid: float = 0.0
    ask: float = 0.0


# --- Order Models ---


class OrderSpec(BaseModel):
    """A complete order specification ready for submission."""

    symbol: str
    action: OrderAction
    quantity: float
    order_type: OrderType = OrderType.LIMIT
    limit_price: float | None = None
    stop_price: float | None = None
    trailing_percent: float | None = None

    # Bracket order components
    take_profit_price: float | None = None
    stop_loss_price: float | None = None

    # Metadata
    reason: str = ""
    time_in_force: str = "DAY"


class OrderResult(BaseModel):
    """Result of an order submission."""

    order_id: int
    symbol: str
    action: OrderAction
    quantity: float
    status: str
    filled_price: float | None = None
    filled_quantity: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


# --- Analysis Models (Claude structured output) ---


class TradeAnalysis(BaseModel):
    """Structured analysis of a potential trade."""

    symbol: str
    direction: Direction
    conviction: ConvictionLevel
    risk_level: RiskLevel

    bull_case: str
    bear_case: str
    catalyst: str = ""
    risk_factors: list[str] = Field(default_factory=list)

    entry_price: float | None = None
    target_price: float | None = None
    stop_loss_price: float | None = None
    reward_risk_ratio: float | None = None

    recommended_position_size: float | None = None
    recommended_portfolio_pct: float | None = None
    reasoning: str = ""


class PositionSizeRecommendation(BaseModel):
    """Position sizing recommendation from Claude."""

    symbol: str
    shares: float
    dollar_amount: float
    portfolio_percent: float
    risk_per_share: float = 0.0
    total_risk: float = 0.0
    reasoning: str = ""


class PortfolioRiskAssessment(BaseModel):
    """Portfolio-level risk assessment."""

    overall_risk: RiskLevel
    total_exposure: float
    cash_percent: float
    largest_position_pct: float
    largest_position_symbol: str = ""
    sector_concentrations: dict[str, float] = Field(default_factory=dict)
    correlation_warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    reasoning: str = ""


# --- Journal Models ---


class TradeJournalEntry(BaseModel):
    """A trade journal entry."""

    id: int | None = None
    ticker: str
    direction: Direction
    entry_price: float | None = None
    exit_price: float | None = None
    shares: float = 0.0
    entry_date: datetime | None = None
    exit_date: datetime | None = None
    thesis: str = ""
    claude_analysis: str = ""
    outcome: TradeOutcome = TradeOutcome.OPEN
    pnl: float = 0.0
    notes: str = ""
    tags: list[str] = Field(default_factory=list)


class PortfolioSnapshot(BaseModel):
    """Point-in-time portfolio snapshot."""

    id: int | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    total_value: float = 0.0
    cash: float = 0.0
    positions_json: str = ""
    daily_pnl: float = 0.0


class AnalysisLogEntry(BaseModel):
    """Log of a Claude analysis query."""

    id: int | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    query: str = ""
    context_json: str = ""
    response: str = ""
    tokens_used: int = 0
