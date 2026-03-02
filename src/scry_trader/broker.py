"""IBKR connection manager using ib_async."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from ib_async import (
    IB,
    LimitOrder,
    MarketOrder,
    Order,
    Stock,
    StopLimitOrder,
    StopOrder,
    Ticker,
    Trade,
)

from scry_trader.models import (
    AccountSummary,
    OrderResult,
    OrderSpec,
    OrderType,
    Position,
    WatchlistItem,
)

if TYPE_CHECKING:
    from scry_trader.config import IBKRConfig

logger = logging.getLogger(__name__)


class BrokerError(Exception):
    """Raised when a broker operation fails."""


class Broker:
    """Manages connection to Interactive Brokers via ib_async."""

    def __init__(self, config: IBKRConfig) -> None:
        self.config = config
        self.ib = IB()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected and self.ib.isConnected()

    async def connect(self) -> None:
        """Connect to IB Gateway or TWS."""
        if self.connected:
            return
        try:
            await self.ib.connectAsync(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout,
            )
            self._connected = True
            logger.info(
                "Connected to IBKR at %s:%d (client %d)",
                self.config.host,
                self.config.port,
                self.config.client_id,
            )
        except Exception as e:
            self._connected = False
            raise BrokerError(f"Failed to connect to IBKR: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self.connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    def _require_connection(self) -> None:
        if not self.connected:
            raise BrokerError("Not connected to IBKR. Call connect() first.")

    # --- Portfolio ---

    async def get_positions(self) -> list[Position]:
        """Fetch all current positions."""
        self._require_connection()
        ib_positions = self.ib.positions()
        positions: list[Position] = []
        for p in ib_positions:
            positions.append(
                Position(
                    symbol=p.contract.symbol,
                    quantity=p.position,
                    avg_cost=p.avgCost,
                    account=p.account,
                )
            )
        return positions

    async def get_account_summary(self) -> AccountSummary:
        """Fetch account summary with positions and values."""
        self._require_connection()
        account_values = self.ib.accountSummary()

        summary = AccountSummary()
        for av in account_values:
            tag = av.tag
            val = av.value
            if tag == "NetLiquidation":
                summary.total_value = float(val)
            elif tag == "TotalCashValue":
                summary.cash = float(val)
            elif tag == "BuyingPower":
                summary.buying_power = float(val)
            elif tag == "RealizedPnL":
                summary.realized_pnl = float(val)
            elif tag == "UnrealizedPnL":
                summary.unrealized_pnl = float(val)
            elif tag == "GrossPositionValue":
                summary.margin_used = float(val)

        summary.positions = await self.get_positions()
        return summary

    # --- Market Data ---

    async def get_quote(self, symbol: str) -> WatchlistItem:
        """Get a snapshot quote for a single symbol."""
        self._require_connection()
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        ticker = self.ib.reqMktData(contract, snapshot=True)
        await asyncio.sleep(2)  # allow data to arrive
        self.ib.cancelMktData(contract)

        last = _safe_float(ticker.last)
        close = _safe_float(ticker.close)
        return WatchlistItem(
            symbol=symbol,
            last_price=last,
            change=last - close if close > 0 else 0.0,
            change_percent=(last - close) / close * 100 if close > 0 else 0.0,
            volume=int(_safe_float(ticker.volume)),
            bid=_safe_float(ticker.bid),
            ask=_safe_float(ticker.ask),
        )

    async def get_quotes(self, symbols: list[str]) -> list[WatchlistItem]:
        """Get snapshot quotes for multiple symbols."""
        self._require_connection()
        contracts = [Stock(s, "SMART", "USD") for s in symbols]
        self.ib.qualifyContracts(*contracts)

        tickers: list[Ticker] = []
        for c in contracts:
            tickers.append(self.ib.reqMktData(c, snapshot=True))

        await asyncio.sleep(3)  # allow data for all tickers

        items: list[WatchlistItem] = []
        for symbol, ticker in zip(symbols, tickers, strict=False):
            if ticker.contract:
                self.ib.cancelMktData(ticker.contract)
            last = _safe_float(ticker.last)
            close = _safe_float(ticker.close)
            items.append(
                WatchlistItem(
                    symbol=symbol,
                    last_price=last,
                    change=last - close if close > 0 else 0.0,
                    change_percent=((last - close) / close * 100) if close > 0 else 0.0,
                    volume=int(_safe_float(ticker.volume)),
                    bid=_safe_float(ticker.bid),
                    ask=_safe_float(ticker.ask),
                )
            )
        return items

    async def get_options_chain(
        self, symbol: str, right: str = "", strike: float = 0.0
    ) -> list[dict[str, Any]]:
        """Fetch options chain for a symbol."""
        self._require_connection()
        stock = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(stock)
        chains = self.ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
        await asyncio.sleep(1)

        results: list[dict[str, Any]] = []
        for chain in chains:
            results.append(
                {
                    "exchange": chain.exchange,
                    "expirations": list(chain.expirations),
                    "strikes": list(chain.strikes),
                }
            )
        return results

    # --- Order Submission ---

    async def submit_order(self, spec: OrderSpec) -> OrderResult:
        """Submit an order to IBKR. Returns the order result."""
        self._require_connection()
        contract = Stock(spec.symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        order = _build_order(spec)
        trade: Trade = self.ib.placeOrder(contract, order)

        # Wait briefly for fill or acknowledgment
        await asyncio.sleep(1)

        return OrderResult(
            order_id=trade.order.orderId,
            symbol=spec.symbol,
            action=spec.action,
            quantity=spec.quantity,
            status=trade.orderStatus.status,
            filled_price=(trade.orderStatus.avgFillPrice or None),
            filled_quantity=trade.orderStatus.filled,
        )

    async def submit_bracket_order(self, spec: OrderSpec) -> list[OrderResult]:
        """Submit a bracket order (entry + take-profit + stop-loss)."""
        self._require_connection()
        if not spec.take_profit_price or not spec.stop_loss_price:
            raise BrokerError("Bracket order requires take_profit_price and stop_loss_price")

        contract = Stock(spec.symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        bracket = self.ib.bracketOrder(
            action=spec.action.value,
            quantity=spec.quantity,
            limitPrice=spec.limit_price or 0,
            takeProfitPrice=spec.take_profit_price,
            stopLossPrice=spec.stop_loss_price,
        )

        results: list[OrderResult] = []
        for order in bracket:
            trade = self.ib.placeOrder(contract, order)
            await asyncio.sleep(0.5)
            results.append(
                OrderResult(
                    order_id=trade.order.orderId,
                    symbol=spec.symbol,
                    action=spec.action,
                    quantity=spec.quantity,
                    status=trade.orderStatus.status,
                )
            )
        return results

    async def cancel_order(self, order_id: int) -> None:
        """Cancel an open order."""
        self._require_connection()
        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id:
                self.ib.cancelOrder(trade.order)
                return
        raise BrokerError(f"Order {order_id} not found in open orders")


def _build_order(spec: OrderSpec) -> Order:
    """Convert an OrderSpec to an ib_async Order object."""
    action = spec.action.value
    qty = spec.quantity

    if spec.order_type == OrderType.MARKET:
        return MarketOrder(action, qty)
    elif spec.order_type == OrderType.LIMIT:
        if spec.limit_price is None:
            raise BrokerError("Limit order requires limit_price")
        return LimitOrder(action, qty, spec.limit_price)
    elif spec.order_type == OrderType.STOP:
        if spec.stop_price is None:
            raise BrokerError("Stop order requires stop_price")
        return StopOrder(action, qty, spec.stop_price)
    elif spec.order_type == OrderType.STOP_LIMIT:
        if spec.limit_price is None or spec.stop_price is None:
            raise BrokerError("Stop-limit order requires both limit_price and stop_price")
        return StopLimitOrder(action, qty, spec.limit_price, spec.stop_price)
    else:
        raise BrokerError(f"Unsupported order type: {spec.order_type}")


def _safe_float(value: Any) -> float:
    """Convert ib_async nan/None values to 0.0."""
    if value is None:
        return 0.0
    try:
        f = float(value)
        if f != f:  # NaN check
            return 0.0
        return f
    except (ValueError, TypeError):
        return 0.0
