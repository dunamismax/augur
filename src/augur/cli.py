"""Click CLI — the primary interface for Augur."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

import click
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from augur.analyst import Analyst, AnalystError
from augur.broker import Broker, BrokerError
from augur.config import AppConfig, load_config
from augur.journal import Journal
from augur.models import (
    AccountSummary,
    AnalysisLogEntry,
    OrderAction,
    OrderResult,
    OrderSpec,
    OrderType,
    TradeJournalEntry,
    TradeOutcome,
    WatchlistItem,
)
from augur.risk import RiskManager, TradeChallengeResult, classify_order_exposure

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
@click.version_option(package_name="augur")
def main() -> None:
    """Augur — AI-assisted, human-directed trading system."""
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
@click.option(
    "--thesis",
    help="Operator thesis / exit rationale captured before submission and stored in journal",
)
def buy(
    ticker: str,
    shares: float | None,
    limit_price: float | None,
    thesis: str | None,
) -> None:
    """Interactive buy flow with Claude sizing recommendation."""
    _trade_flow(ticker.upper(), OrderAction.BUY, shares, limit_price, thesis)


@main.command()
@click.argument("ticker")
@click.option("--shares", type=float, help="Number of shares to sell")
@click.option("--limit", "limit_price", type=float, help="Limit price")
@click.option(
    "--thesis",
    help="Operator thesis / exit rationale captured before submission and stored in journal",
)
def sell(
    ticker: str,
    shares: float | None,
    limit_price: float | None,
    thesis: str | None,
) -> None:
    """Interactive sell flow."""
    _trade_flow(ticker.upper(), OrderAction.SELL, shares, limit_price, thesis)


def _trade_flow(
    ticker: str,
    action: OrderAction,
    shares: float | None,
    limit_price: float | None,
    thesis: str | None = None,
) -> None:
    config = _load()
    analyst = Analyst(config.claude, config.risk)
    risk_mgr = RiskManager(config.risk)
    j = Journal(config.database.path)

    # Get portfolio context and a reference quote for market order risk estimation
    broker = Broker(config.ibkr)
    try:
        portfolio, quote = _run(_get_portfolio_and_quote(broker, ticker))
    except BrokerError as e:
        console.print(f"[red]Broker error:[/red] {e}")
        sys.exit(1)

    # Get Claude's order recommendation
    direction_str = "buy" if action == OrderAction.BUY else "sell"
    console.print(f"[dim]Building {direction_str} order for {ticker}...[/dim]")

    try:
        candidate = analyst.construct_order(
            ticker, direction_str, portfolio=portfolio
        ).model_dump()
    except AnalystError as e:
        console.print(f"[red]Analysis error:[/red] {e}")
        sys.exit(1)

    # User intent is authoritative; validate the final reviewed order after overrides.
    candidate["symbol"] = ticker
    candidate["action"] = action
    if shares is not None:
        candidate["quantity"] = shares
    if limit_price is not None:
        candidate["limit_price"] = limit_price
        candidate["order_type"] = OrderType.LIMIT

    if quote.last_price > 0:
        candidate["reference_price"] = quote.last_price

    try:
        order_spec = OrderSpec.model_validate(candidate)
    except ValidationError as e:
        console.print("[red]Invalid order returned for review:[/red]")
        for error in e.errors():
            location = ".".join(str(part) for part in error["loc"])
            console.print(f"  [red]- {location}:[/red] {error['msg']}")
        sys.exit(1)
    exposure = classify_order_exposure(order_spec, portfolio)

    operator_thesis = _capture_operator_thesis(ticker, action, thesis)

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

    challenge = risk_mgr.challenge_trade(
        order_spec,
        portfolio,
        operator_thesis,
        j.get_trades_by_ticker(ticker),
    )
    console.print()
    _display_trade_challenge(challenge, config)
    if challenge.blockers:
        for blocker in challenge.blockers:
            console.print(f"[red]CHALLENGE BLOCKER:[/red] {blocker}")
        console.print("\n[red]Order cancelled before submission.[/red]")
        return
    if challenge.warnings:
        review_ack = click.prompt(
            "Type REVIEWED to confirm you've considered the challenge",
            default="",
            show_default=False,
        )
        if review_ack.strip().upper() != "REVIEWED":
            console.print("[dim]Order cancelled.[/dim]")
            return

    # Confirm
    console.print()
    if not click.confirm("Submit this order?"):
        console.print("[dim]Order cancelled.[/dim]")
        return

    # Submit
    primary_result: OrderResult | None = None
    try:
        if (
            order_spec.take_profit_price
            and order_spec.stop_loss_price
            and exposure.reducing_quantity == 0
        ):
            results = _run(_submit_bracket(broker, order_spec))
            for r in results:
                console.print(
                    f"[green]Order {r.order_id}:[/green] {r.status} "
                    f"({r.symbol} {r.action.value} {r.quantity})"
                )
            primary_result = results[0]  # parent/entry order
        else:
            result = _run(_submit_order(broker, order_spec))
            console.print(
                f"[green]Order {result.order_id}:[/green] {result.status} "
                f"({result.symbol} {result.action.value} {result.quantity})"
            )
            primary_result = result
    except BrokerError as e:
        console.print(f"[red]Order submission failed:[/red] {e}")
        sys.exit(1)

    # Only journal the trade if it was actually filled (or partially filled).
    # The broker returns status from the first update event — don't record
    # trades that were rejected, cancelled, or haven't filled yet.
    if (
        primary_result.filled_quantity is not None
        and primary_result.filled_quantity <= 0
    ):
        console.print(
            "[dim]Order submitted but not yet filled — skipping journal entry. "
            "Run 'augur portfolio' to check status.[/dim]"
        )
        return

    # Use the actual fill price when available, otherwise fall back to spec
    entry_price = (
        primary_result.filled_price
        or order_spec.limit_price
        or order_spec.stop_price
        or order_spec.reference_price
    )
    if entry_price is None:
        console.print(
            "[yellow]Warning:[/yellow] Filled order has no usable fill price "
            "for journal."
        )
        return

    realized = _journal_fills(
        journal=j,
        ticker=ticker,
        action=action,
        filled_quantity=primary_result.filled_quantity,
        fill_price=entry_price,
        order_spec=order_spec,
        portfolio=portfolio,
        operator_thesis=operator_thesis,
    )
    if realized:
        entry_label = "entry" if realized == 1 else "entries"
        console.print(f"[dim]{realized} journal {entry_label} updated.[/dim]")


async def _get_portfolio_and_quote(
    broker: Broker, symbol: str
) -> tuple[AccountSummary, WatchlistItem]:
    """Fetch portfolio and a quote in a single connection."""
    await broker.connect()
    try:
        summary = await broker.get_account_summary()
        quote = await broker.get_quote(symbol)
        return summary, quote
    finally:
        await broker.disconnect()


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
    if spec.trailing_percent:
        table.add_row("Trailing %", f"{spec.trailing_percent:.2f}%")
    if spec.take_profit_price:
        table.add_row("Take Profit", f"${spec.take_profit_price:,.2f}")
    if spec.stop_loss_price:
        table.add_row("Stop Loss", f"${spec.stop_loss_price:,.2f}")
    table.add_row("TIF", spec.time_in_force.value)
    if spec.reason:
        table.add_row("Reason", spec.reason)

    console.print(table)


def _journal_fills(
    journal: Journal,
    ticker: str,
    action: OrderAction,
    filled_quantity: float,
    fill_price: float,
    order_spec: OrderSpec,
    portfolio: AccountSummary,
    operator_thesis: str,
) -> int:
    filled_spec = order_spec.model_copy(update={"quantity": filled_quantity})
    exposure = classify_order_exposure(filled_spec, portfolio, quantity=filled_quantity)
    updates = 0
    rationale_note = _compose_rationale_note(operator_thesis, order_spec.reason)

    if exposure.reducing_quantity > 0 and exposure.reducing_direction is not None:
        closed_lots = journal.close_position(
            ticker=ticker,
            direction=exposure.reducing_direction,
            shares=exposure.reducing_quantity,
            exit_price=fill_price,
            notes=rationale_note,
        )
        updates += len(closed_lots)
        closed_quantity = sum(lot.shares for lot in closed_lots)
        if closed_quantity < exposure.reducing_quantity:
            console.print(
                "[yellow]Warning:[/yellow] Journal had fewer open lots than the broker "
                "position being reduced."
            )

    if exposure.opening_quantity > 0 and exposure.opening_direction is not None:
        journal.add_trade(
            TradeJournalEntry(
                ticker=ticker,
                direction=exposure.opening_direction,
                entry_price=fill_price,
                shares=exposure.opening_quantity,
                open_shares=exposure.opening_quantity,
                entry_date=datetime.now(),
                thesis=operator_thesis,
                claude_analysis=order_spec.reason,
            )
        )
        updates += 1

    return updates


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
    table.add_column("Thesis")
    table.add_column("Date")

    for t in trades:
        outcome_style = {
            TradeOutcome.WIN: "green",
            TradeOutcome.LOSS: "red",
            TradeOutcome.BREAKEVEN: "dim",
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
            _summarize_text(t.thesis or t.notes or t.claude_analysis),
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


def _capture_operator_thesis(
    ticker: str,
    action: OrderAction,
    thesis: str | None,
) -> str:
    if thesis is None:
        prompt_label = (
            "Operator thesis" if action == OrderAction.BUY else "Operator exit rationale"
        )
        thesis = click.prompt(
            f"{prompt_label} for {ticker}",
            default="",
            show_default=False,
        )
    return " ".join(thesis.split())


def _display_trade_challenge(
    challenge: TradeChallengeResult,
    config: AppConfig,
) -> None:
    account_mode = "PAPER" if config.risk.paper_trading else "LIVE"
    table = Table(title="Decision Challenge", show_header=False, border_style="magenta")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Account Mode", account_mode)
    table.add_row("Configured Account", config.ibkr.account or "—")
    table.add_row("Opening Qty", f"{challenge.opening_quantity:,.0f}")
    if challenge.reducing_quantity > 0:
        table.add_row("Reducing Qty", f"{challenge.reducing_quantity:,.0f}")
    table.add_row("Post-Trade Qty", f"{challenge.post_trade_quantity:,.0f}")
    if challenge.entry_price is not None:
        table.add_row("Planned Entry", f"${challenge.entry_price:,.2f}")
    if challenge.stop_loss_price is not None:
        table.add_row("Stop Loss", f"${challenge.stop_loss_price:,.2f}")
    if challenge.take_profit_price is not None:
        table.add_row("Take Profit", f"${challenge.take_profit_price:,.2f}")
    if challenge.max_loss is not None:
        loss_text = f"${challenge.max_loss:,.2f}"
        if challenge.max_loss_pct is not None:
            loss_text += f" ({challenge.max_loss_pct:.2f}% of portfolio)"
        table.add_row("Loss to Stop", loss_text)
    if challenge.reward_risk_ratio is not None:
        table.add_row("Reward/Risk", f"{challenge.reward_risk_ratio:.2f}:1")
    if challenge.post_trade_position_pct is not None:
        table.add_row("Post-Trade Size", f"{challenge.post_trade_position_pct:.1f}% of portfolio")

    console.print(Panel(challenge.thesis or "—", title="Operator Thesis", border_style="cyan"))
    console.print(table)

    for warning in challenge.warnings:
        console.print(f"[yellow]Challenge:[/yellow] {warning}")

    if challenge.prompts:
        challenge_text = "\n".join(f"  - {prompt}" for prompt in challenge.prompts)
        console.print(
            Panel(
                challenge_text,
                title="Adversarial Questions",
                border_style="yellow",
            )
        )


def _compose_rationale_note(operator_thesis: str, claude_reason: str) -> str:
    notes: list[str] = []
    if operator_thesis:
        notes.append(f"Operator rationale: {operator_thesis}")
    if claude_reason:
        notes.append(f"Claude rationale: {claude_reason}")
    return "\n".join(notes)


def _summarize_text(value: str, limit: int = 36) -> str:
    cleaned = " ".join(value.split())
    if not cleaned:
        return "—"
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 1]}…"
