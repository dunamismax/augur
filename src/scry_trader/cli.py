"""Click CLI — the primary interface for scry-trader."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scry_trader.analyst import Analyst, AnalystError
from scry_trader.broker import Broker, BrokerError
from scry_trader.config import AppConfig, load_config
from scry_trader.journal import Journal
from scry_trader.models import (
    AccountSummary,
    AnalysisLogEntry,
    Direction,
    OrderAction,
    OrderResult,
    OrderSpec,
    OrderType,
    TradeJournalEntry,
    TradeOutcome,
    WatchlistItem,
)
from scry_trader.risk import RiskManager

if TYPE_CHECKING:
    from collections.abc import Coroutine

console = Console()


def _load() -> AppConfig:
    return load_config()


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from sync Click context."""
    return asyncio.run(coro)


# --- Main Group ---


@click.group()
@click.version_option(package_name="scry-trader")
def main() -> None:
    """scry-trader — AI-assisted, human-directed trading system."""
    pass


# --- Portfolio ---


@main.command()
def portfolio() -> None:
    """Show current positions, P&L, and allocation."""
    config = _load()
    broker = Broker(config.ibkr)

    try:
        summary = _run(_get_portfolio(broker))
    except BrokerError as e:
        console.print(f"[red]Broker error:[/red] {e}")
        sys.exit(1)

    _display_portfolio(summary)


async def _get_portfolio(broker: Broker) -> AccountSummary:
    await broker.connect()
    try:
        return await broker.get_account_summary()
    finally:
        await broker.disconnect()


def _display_portfolio(summary: AccountSummary) -> None:
    # Account overview
    account_table = Table(title="Account Summary", show_header=False, border_style="blue")
    account_table.add_column("Metric", style="bold")
    account_table.add_column("Value", justify="right")
    account_table.add_row("Net Liquidation", f"${summary.total_value:,.2f}")
    account_table.add_row("Cash", f"${summary.cash:,.2f}")
    account_table.add_row("Buying Power", f"${summary.buying_power:,.2f}")
    account_table.add_row(
        "Unrealized P&L",
        _colored_pnl(summary.unrealized_pnl),
    )
    account_table.add_row(
        "Realized P&L",
        _colored_pnl(summary.realized_pnl),
    )
    console.print(account_table)
    console.print()

    # Positions
    if not summary.positions:
        console.print("[dim]No open positions.[/dim]")
        return

    pos_table = Table(title="Positions", border_style="blue")
    pos_table.add_column("Symbol", style="bold")
    pos_table.add_column("Qty", justify="right")
    pos_table.add_column("Avg Cost", justify="right")
    pos_table.add_column("Mkt Price", justify="right")
    pos_table.add_column("Mkt Value", justify="right")
    pos_table.add_column("P&L", justify="right")
    pos_table.add_column("P&L %", justify="right")

    for pos in summary.positions:
        pos_table.add_row(
            pos.symbol,
            f"{pos.quantity:,.0f}",
            f"${pos.avg_cost:,.2f}",
            f"${pos.market_price:,.2f}" if pos.market_price else "—",
            f"${pos.market_value:,.2f}" if pos.market_value else "—",
            _colored_pnl(pos.unrealized_pnl),
            _colored_pct(pos.pnl_percent),
        )

    console.print(pos_table)


# --- Watch ---


@main.command()
@click.argument("symbols", nargs=-1)
def watch(symbols: tuple[str, ...]) -> None:
    """Show watchlist with live prices."""
    config = _load()
    watch_symbols = list(symbols) if symbols else config.watchlist.symbols
    broker = Broker(config.ibkr)

    try:
        items = _run(_get_quotes(broker, watch_symbols))
    except BrokerError as e:
        console.print(f"[red]Broker error:[/red] {e}")
        sys.exit(1)

    table = Table(title="Watchlist", border_style="green")
    table.add_column("Symbol", style="bold")
    table.add_column("Last", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Change %", justify="right")
    table.add_column("Volume", justify="right")
    table.add_column("Bid", justify="right")
    table.add_column("Ask", justify="right")

    for item in items:
        table.add_row(
            item.symbol,
            f"${item.last_price:,.2f}" if item.last_price else "—",
            _colored_pnl(item.change),
            _colored_pct(item.change_percent),
            f"{item.volume:,}" if item.volume else "—",
            f"${item.bid:,.2f}" if item.bid else "—",
            f"${item.ask:,.2f}" if item.ask else "—",
        )

    console.print(table)


async def _get_quotes(broker: Broker, symbols: list[str]) -> list[WatchlistItem]:
    await broker.connect()
    try:
        return await broker.get_quotes(symbols)
    finally:
        await broker.disconnect()


# --- Ask ---


@main.command()
@click.argument("question")
@click.option("--no-portfolio", is_flag=True, help="Don't include portfolio context")
def ask(question: str, no_portfolio: bool) -> None:
    """Ask Claude a free-form question with portfolio context."""
    config = _load()
    analyst = Analyst(config.claude, config.risk)
    journal = Journal(config.database.path)

    portfolio: AccountSummary | None = None
    if not no_portfolio:
        try:
            broker = Broker(config.ibkr)
            portfolio = _run(_get_portfolio(broker))
        except BrokerError:
            console.print("[dim]Could not connect to IBKR. Proceeding without portfolio.[/dim]")

    console.print("[dim]Thinking...[/dim]")
    try:
        response = analyst.ask(question, portfolio=portfolio)
    except AnalystError as e:
        console.print(f"[red]Analysis error:[/red] {e}")
        sys.exit(1)

    console.print()
    console.print(Panel(response, title="Claude", border_style="cyan"))

    # Log the analysis
    journal.log_analysis(
        AnalysisLogEntry(
            query=question,
            response=response,
        )
    )


# --- Analyze ---


@main.command()
@click.argument("ticker")
@click.option("--question", "-q", default="", help="Specific question about this ticker")
def analyze(ticker: str, question: str) -> None:
    """Deep analysis of a specific ticker."""
    config = _load()
    analyst = Analyst(config.claude, config.risk)
    ticker = ticker.upper()

    portfolio: AccountSummary | None = None
    try:
        broker = Broker(config.ibkr)
        portfolio = _run(_get_portfolio(broker))
    except BrokerError:
        console.print("[dim]No IBKR connection. Analyzing without portfolio context.[/dim]")

    console.print(f"[dim]Analyzing {ticker}...[/dim]")
    try:
        analysis = analyst.analyze_trade(ticker, question=question, portfolio=portfolio)
    except AnalystError as e:
        console.print(f"[red]Analysis error:[/red] {e}")
        sys.exit(1)

    # Display structured analysis
    conviction_color = {
        "high": "green",
        "medium": "yellow",
        "low": "red",
        "none": "dim",
    }.get(analysis.conviction.value, "white")

    risk_color = {
        "low": "green",
        "moderate": "yellow",
        "high": "red",
        "extreme": "bold red",
    }.get(analysis.risk_level.value, "white")

    console.print()
    console.print(
        Panel(
            f"[bold]{ticker}[/bold] — {analysis.direction.value.upper()}\n"
            f"Conviction: [{conviction_color}]"
            f"{analysis.conviction.value.upper()}[/{conviction_color}] "
            f"| Risk: [{risk_color}]"
            f"{analysis.risk_level.value.upper()}[/{risk_color}]",
            title="Trade Analysis",
            border_style="cyan",
        )
    )

    table = Table(show_header=False, border_style="blue", padding=(0, 2))
    table.add_column("Label", style="bold")
    table.add_column("Value")

    if analysis.entry_price:
        table.add_row("Entry", f"${analysis.entry_price:,.2f}")
    if analysis.target_price:
        table.add_row("Target", f"${analysis.target_price:,.2f}")
    if analysis.stop_loss_price:
        table.add_row("Stop Loss", f"${analysis.stop_loss_price:,.2f}")
    if analysis.reward_risk_ratio:
        table.add_row("R:R Ratio", f"{analysis.reward_risk_ratio:.1f}:1")
    if analysis.recommended_portfolio_pct:
        table.add_row("Position Size", f"{analysis.recommended_portfolio_pct:.1f}% of portfolio")

    console.print(table)

    console.print(Panel(analysis.bull_case, title="Bull Case", border_style="green"))
    console.print(Panel(analysis.bear_case, title="Bear Case", border_style="red"))

    if analysis.risk_factors:
        risk_text = "\n".join(f"  - {r}" for r in analysis.risk_factors)
        console.print(Panel(risk_text, title="Risk Factors", border_style="yellow"))

    if analysis.reasoning:
        console.print(Panel(analysis.reasoning, title="Reasoning", border_style="dim"))


# --- Buy / Sell ---


@main.command()
@click.argument("ticker")
@click.option("--shares", type=float, help="Number of shares (overrides Claude recommendation)")
@click.option("--limit", "limit_price", type=float, help="Limit price")
def buy(ticker: str, shares: float | None, limit_price: float | None) -> None:
    """Interactive buy flow with Claude sizing recommendation."""
    _trade_flow(ticker.upper(), OrderAction.BUY, shares, limit_price)


@main.command()
@click.argument("ticker")
@click.option("--shares", type=float, help="Number of shares to sell")
@click.option("--limit", "limit_price", type=float, help="Limit price")
def sell(ticker: str, shares: float | None, limit_price: float | None) -> None:
    """Interactive sell flow."""
    _trade_flow(ticker.upper(), OrderAction.SELL, shares, limit_price)


def _trade_flow(
    ticker: str,
    action: OrderAction,
    shares: float | None,
    limit_price: float | None,
) -> None:
    config = _load()
    analyst = Analyst(config.claude, config.risk)
    risk_mgr = RiskManager(config.risk)
    journal = Journal(config.database.path)

    # Get portfolio context
    broker = Broker(config.ibkr)
    try:
        portfolio = _run(_get_portfolio(broker))
    except BrokerError as e:
        console.print(f"[red]Broker error:[/red] {e}")
        sys.exit(1)

    # Get Claude's order recommendation
    direction = "buy" if action == OrderAction.BUY else "sell"
    console.print(f"[dim]Building {direction} order for {ticker}...[/dim]")

    try:
        order_spec = analyst.construct_order(ticker, direction, portfolio=portfolio)
    except AnalystError as e:
        console.print(f"[red]Analysis error:[/red] {e}")
        sys.exit(1)

    # Override with user-specified values
    if shares is not None:
        order_spec.quantity = shares
    if limit_price is not None:
        order_spec.limit_price = limit_price
        order_spec.order_type = OrderType.LIMIT

    # Display the order
    console.print()
    _display_order(order_spec)

    # Risk check
    risk_result = risk_mgr.check_order(order_spec, portfolio)
    if risk_result.warnings:
        for w in risk_result.warnings:
            console.print(f"[yellow]Warning:[/yellow] {w}")
    if not risk_result.ok:
        console.print()
        for v in risk_result.violations:
            console.print(f"[red]RISK VIOLATION:[/red] {v}")
        console.print("\n[red]Order blocked by risk management.[/red]")
        return

    # Confirm
    console.print()
    if not click.confirm("Submit this order?"):
        console.print("[dim]Order cancelled.[/dim]")
        return

    # Submit
    try:
        if order_spec.take_profit_price and order_spec.stop_loss_price:
            results = _run(_submit_bracket(broker, order_spec))
            for r in results:
                console.print(
                    f"[green]Order {r.order_id}:[/green] {r.status} "
                    f"({r.symbol} {r.action.value} {r.quantity})"
                )
        else:
            result = _run(_submit_order(broker, order_spec))
            console.print(
                f"[green]Order {result.order_id}:[/green] {result.status} "
                f"({result.symbol} {result.action.value} {result.quantity})"
            )
    except BrokerError as e:
        console.print(f"[red]Order submission failed:[/red] {e}")
        sys.exit(1)

    # Journal entry
    journal.add_trade(
        TradeJournalEntry(
            ticker=ticker,
            direction=Direction.LONG if action == OrderAction.BUY else Direction.SHORT,
            entry_price=order_spec.limit_price,
            shares=order_spec.quantity,
            entry_date=datetime.now(),
            thesis=order_spec.reason,
        )
    )
    console.print("[dim]Trade logged to journal.[/dim]")


async def _submit_order(broker: Broker, spec: OrderSpec) -> OrderResult:
    await broker.connect()
    try:
        return await broker.submit_order(spec)
    finally:
        await broker.disconnect()


async def _submit_bracket(broker: Broker, spec: OrderSpec) -> list[OrderResult]:
    await broker.connect()
    try:
        return await broker.submit_bracket_order(spec)
    finally:
        await broker.disconnect()


def _display_order(spec: OrderSpec) -> None:
    table = Table(title="Order Preview", show_header=False, border_style="cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Symbol", spec.symbol)
    table.add_row("Action", spec.action.value)
    table.add_row("Quantity", f"{spec.quantity:,.0f}")
    table.add_row("Type", spec.order_type.value)
    if spec.limit_price:
        table.add_row("Limit Price", f"${spec.limit_price:,.2f}")
    if spec.stop_price:
        table.add_row("Stop Price", f"${spec.stop_price:,.2f}")
    if spec.take_profit_price:
        table.add_row("Take Profit", f"${spec.take_profit_price:,.2f}")
    if spec.stop_loss_price:
        table.add_row("Stop Loss", f"${spec.stop_loss_price:,.2f}")
    if spec.reason:
        table.add_row("Reason", spec.reason)

    console.print(table)


# --- Risk ---


@main.command()
def risk() -> None:
    """Portfolio risk assessment."""
    config = _load()

    broker = Broker(config.ibkr)
    try:
        portfolio = _run(_get_portfolio(broker))
    except BrokerError as e:
        console.print(f"[red]Broker error:[/red] {e}")
        sys.exit(1)

    # Quick rule-based check
    risk_mgr = RiskManager(config.risk)
    health = risk_mgr.check_portfolio_health(portfolio)

    if health.warnings:
        for w in health.warnings:
            console.print(f"[yellow]Warning:[/yellow] {w}")
    if health.violations:
        for v in health.violations:
            console.print(f"[red]VIOLATION:[/red] {v}")

    # Claude deep analysis
    console.print("[dim]Running Claude risk analysis...[/dim]")
    analyst = Analyst(config.claude, config.risk)

    try:
        assessment = analyst.assess_portfolio_risk(portfolio)
    except AnalystError as e:
        console.print(f"[red]Analysis error:[/red] {e}")
        sys.exit(1)

    risk_color = {
        "low": "green",
        "moderate": "yellow",
        "high": "red",
        "extreme": "bold red",
    }.get(assessment.overall_risk.value, "white")

    console.print()
    console.print(
        Panel(
            f"Overall Risk: [{risk_color}]"
            f"{assessment.overall_risk.value.upper()}[/{risk_color}]\n"
            f"Exposure: {assessment.total_exposure:.1f}% "
            f"| Cash: {assessment.cash_percent:.1f}%\n"
            f"Largest: {assessment.largest_position_symbol} "
            f"({assessment.largest_position_pct:.1f}%)",
            title="Portfolio Risk Assessment",
            border_style="cyan",
        )
    )

    if assessment.recommendations:
        rec_text = "\n".join(f"  - {r}" for r in assessment.recommendations)
        console.print(Panel(rec_text, title="Recommendations", border_style="green"))

    if assessment.correlation_warnings:
        corr_text = "\n".join(f"  - {w}" for w in assessment.correlation_warnings)
        console.print(Panel(corr_text, title="Correlation Warnings", border_style="yellow"))

    if assessment.reasoning:
        console.print(Panel(assessment.reasoning, title="Analysis", border_style="dim"))


# --- Journal ---


@main.command()
@click.option("--ticker", "-t", help="Filter by ticker")
@click.option("--open-only", is_flag=True, help="Show only open trades")
@click.option("--stats", is_flag=True, help="Show trade statistics")
@click.option("--limit", "count", type=int, default=20, help="Number of trades to show")
def journal(ticker: str | None, open_only: bool, stats: bool, count: int) -> None:
    """View trade journal entries and statistics."""
    config = _load()
    j = Journal(config.database.path)

    if stats:
        _display_stats(j.get_trade_stats())
        return

    if ticker:
        trades = j.get_trades_by_ticker(ticker.upper())
    elif open_only:
        trades = j.get_open_trades()
    else:
        trades = j.get_recent_trades(limit=count)

    if not trades:
        console.print("[dim]No trades found.[/dim]")
        return

    table = Table(title="Trade Journal", border_style="blue")
    table.add_column("ID", style="dim")
    table.add_column("Ticker", style="bold")
    table.add_column("Dir")
    table.add_column("Entry", justify="right")
    table.add_column("Exit", justify="right")
    table.add_column("Shares", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Outcome")
    table.add_column("Date")

    for t in trades:
        outcome_style = {
            TradeOutcome.WIN: "green",
            TradeOutcome.LOSS: "red",
            TradeOutcome.OPEN: "yellow",
        }.get(t.outcome, "white")

        table.add_row(
            str(t.id) if t.id else "—",
            t.ticker,
            t.direction.value,
            f"${t.entry_price:,.2f}" if t.entry_price else "—",
            f"${t.exit_price:,.2f}" if t.exit_price else "—",
            f"{t.shares:,.0f}" if t.shares else "—",
            _colored_pnl(t.pnl) if t.outcome != TradeOutcome.OPEN else "—",
            f"[{outcome_style}]{t.outcome.value.upper()}[/{outcome_style}]",
            t.entry_date.strftime("%Y-%m-%d") if t.entry_date else "—",
        )

    console.print(table)


def _display_stats(stats: dict[str, float | int]) -> None:
    table = Table(title="Trade Statistics", show_header=False, border_style="blue")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Trades", str(stats["total_trades"]))
    table.add_row("Wins", f"[green]{stats['wins']}[/green]")
    table.add_row("Losses", f"[red]{stats['losses']}[/red]")
    table.add_row("Open", f"[yellow]{stats['open']}[/yellow]")
    table.add_row("Win Rate", f"{stats['win_rate']:.1f}%")
    table.add_row("Total P&L", _colored_pnl(float(stats["total_pnl"])))
    table.add_row("Avg Win", _colored_pnl(float(stats["avg_win"])))
    table.add_row("Avg Loss", _colored_pnl(float(stats["avg_loss"])))

    console.print(table)


# --- Alerts (placeholder for Phase 3) ---


@main.command()
def alerts() -> None:
    """Check for risk alerts or opportunities."""
    console.print("[dim]Alerts are not yet implemented (Phase 3).[/dim]")


# --- Helpers ---


def _colored_pnl(value: float) -> str:
    if value > 0:
        return f"[green]+${value:,.2f}[/green]"
    elif value < 0:
        return f"[red]-${abs(value):,.2f}[/red]"
    return "$0.00"


def _colored_pct(value: float) -> str:
    if value > 0:
        return f"[green]+{value:.2f}%[/green]"
    elif value < 0:
        return f"[red]{value:.2f}%[/red]"
    return "0.00%"
