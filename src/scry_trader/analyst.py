"""Claude analysis engine — the brain of scry-trader.

Supports two backends:
  - "cli": shells out to the ``claude`` CLI (uses Claude Max subscription)
  - "api": uses the Anthropic SDK directly (requires ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import TYPE_CHECKING, Any

from scry_trader.models import (
    AccountSummary,
    OrderSpec,
    PortfolioRiskAssessment,
    PositionSizeRecommendation,
    TradeAnalysis,
    TradeJournalEntry,
)
from scry_trader.prompts.system import build_system_prompt
from scry_trader.prompts.tools import TRADING_TOOLS

if TYPE_CHECKING:
    from scry_trader.config import ClaudeConfig, RiskConfig

logger = logging.getLogger(__name__)

_CLI_TIMEOUT = 120  # seconds


class AnalystError(Exception):
    """Raised when Claude analysis fails."""


def _get_tool_schema(tool_name: str) -> dict[str, Any]:
    """Extract the ``input_schema`` for a named tool from TRADING_TOOLS."""
    for tool in TRADING_TOOLS:
        if tool["name"] == tool_name:
            return dict(tool["input_schema"])
    msg = f"Unknown tool: {tool_name}"
    raise AnalystError(msg)


class Analyst:
    """Claude-powered trading analysis engine."""

    def __init__(self, claude_config: ClaudeConfig, risk_config: RiskConfig) -> None:
        self.model = claude_config.model
        self.max_tokens = claude_config.max_tokens
        self.backend = claude_config.backend
        self.system_prompt = build_system_prompt(
            max_position_pct=risk_config.max_position_pct,
            max_sector_pct=risk_config.max_sector_pct,
            max_daily_loss_pct=risk_config.max_daily_loss_pct,
            max_leverage=risk_config.max_leverage,
        )
        self._conversation: list[dict[str, Any]] = []
        self._api_client: Any = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_api_client(self) -> Any:
        """Lazy-init the Anthropic SDK client (API backend only)."""
        if self._api_client is None:
            import anthropic

            self._api_client = anthropic.Anthropic()
        return self._api_client

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self._conversation = []

    def _build_context(self, portfolio: AccountSummary | None = None) -> str:
        """Build context string from portfolio data."""
        if portfolio is None:
            return ""

        positions_data = []
        for p in portfolio.positions:
            positions_data.append(
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_cost": p.avg_cost,
                    "market_price": p.market_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "pnl_percent": p.pnl_percent,
                }
            )

        context = {
            "account": {
                "total_value": portfolio.total_value,
                "cash": portfolio.cash,
                "buying_power": portfolio.buying_power,
                "unrealized_pnl": portfolio.unrealized_pnl,
                "realized_pnl": portfolio.realized_pnl,
            },
            "positions": positions_data,
            "timestamp": portfolio.timestamp.isoformat(),
        }
        return json.dumps(context, indent=2)

    # ------------------------------------------------------------------
    # Public interface — signatures unchanged
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        portfolio: AccountSummary | None = None,
        use_tools: bool = False,
    ) -> str:
        """Ask Claude a free-form question with optional portfolio context."""
        context = self._build_context(portfolio)
        content = question
        if context:
            content = f"Current portfolio state:\n```json\n{context}\n```\n\n{question}"

        if self.backend == "cli":
            return self._cli_text(content)
        return self._api_text(content, use_tools)

    def analyze_trade(
        self,
        symbol: str,
        question: str = "",
        portfolio: AccountSummary | None = None,
    ) -> TradeAnalysis:
        """Get structured trade analysis for a symbol."""
        context = self._build_context(portfolio)
        prompt = f"Analyze {symbol} as a potential trade."
        if question:
            prompt += f" Specifically: {question}"
        if context:
            prompt = f"Current portfolio:\n```json\n{context}\n```\n\n{prompt}"

        result = self._call_structured("analyze_trade", prompt)
        return TradeAnalysis(**result)

    def recommend_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        portfolio: AccountSummary | None = None,
    ) -> PositionSizeRecommendation:
        """Get position sizing recommendation."""
        context = self._build_context(portfolio)
        prompt = (
            f"Recommend position size for {symbol}. "
            f"Planned entry: ${entry_price:.2f}, stop-loss: ${stop_price:.2f}."
        )
        if context:
            prompt = f"Current portfolio:\n```json\n{context}\n```\n\n{prompt}"

        result = self._call_structured("recommend_position_size", prompt)
        return PositionSizeRecommendation(**result)

    def assess_portfolio_risk(
        self,
        portfolio: AccountSummary,
    ) -> PortfolioRiskAssessment:
        """Get a portfolio-level risk assessment."""
        context = self._build_context(portfolio)
        prompt = (
            f"Current portfolio:\n```json\n{context}\n```\n\n"
            "Perform a comprehensive risk assessment of this portfolio. "
            "Identify concentration risks, correlation risks, and overall exposure."
        )

        result = self._call_structured("assess_portfolio_risk", prompt)
        return PortfolioRiskAssessment(**result)

    def construct_order(
        self,
        symbol: str,
        direction: str,
        portfolio: AccountSummary | None = None,
    ) -> OrderSpec:
        """Have Claude construct a complete order specification."""
        context = self._build_context(portfolio)
        prompt = (
            f"Construct an order to {direction} {symbol}. "
            "Include appropriate entry, stop-loss, and take-profit levels."
        )
        if context:
            prompt = f"Current portfolio:\n```json\n{context}\n```\n\n{prompt}"

        result = self._call_structured("construct_order", prompt)
        return OrderSpec(**result)

    def generate_journal_entry(
        self,
        symbol: str,
        direction: str,
        thesis: str,
        portfolio: AccountSummary | None = None,
    ) -> TradeJournalEntry:
        """Generate a structured trade journal entry."""
        context = self._build_context(portfolio)
        prompt = (
            f"Generate a trade journal entry for a {direction} position in {symbol}. "
            f"Thesis: {thesis}"
        )
        if context:
            prompt = f"Current portfolio:\n```json\n{context}\n```\n\n{prompt}"

        result = self._call_structured("generate_trade_journal_entry", prompt)
        return TradeJournalEntry(**result)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _call_structured(self, tool_name: str, prompt: str) -> dict[str, Any]:
        """Get structured output from Claude via the configured backend."""
        if self.backend == "cli":
            return self._cli_structured(tool_name, prompt)
        return self._api_structured(tool_name, prompt)

    # ------------------------------------------------------------------
    # CLI backend — shells out to ``claude`` CLI
    # ------------------------------------------------------------------

    def _cli_env(self) -> dict[str, str]:
        """Build env for CLI subprocess. Strips CLAUDECODE to avoid nested-session error."""
        return {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    def _cli_run(self, cmd: list[str], prompt: str) -> str:
        """Run a ``claude`` CLI command and return stdout."""
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=_CLI_TIMEOUT,
                env=self._cli_env(),
                check=False,
            )
        except FileNotFoundError:
            msg = "claude CLI not found. Install Claude Code or switch to backend = 'api'."
            raise AnalystError(msg) from None
        except subprocess.TimeoutExpired:
            msg = f"Claude CLI timed out after {_CLI_TIMEOUT}s"
            raise AnalystError(msg) from None

        if result.returncode != 0:
            stderr = result.stderr.strip()
            msg = f"Claude CLI error (exit {result.returncode}): {stderr}"
            raise AnalystError(msg)

        return result.stdout

    def _cli_text(self, prompt: str) -> str:
        """Free-form text response via ``claude`` CLI."""
        # Include conversation history as context for multi-turn
        full_prompt = prompt
        if self._conversation:
            history_parts = [
                f"{msg['role'].upper()}: {msg['content']}" for msg in self._conversation
            ]
            full_prompt = "\n\n".join(history_parts) + "\n\n" + prompt

        self._conversation.append({"role": "user", "content": prompt})

        cmd = [
            "claude",
            "-p",
            "--output-format",
            "text",
            "--system-prompt",
            self.system_prompt,
            "--model",
            self.model,
            "--tools",
            "",
        ]

        reply = self._cli_run(cmd, full_prompt).strip()
        self._conversation.append({"role": "assistant", "content": reply})
        return reply

    def _cli_structured(self, tool_name: str, prompt: str) -> dict[str, Any]:
        """Structured JSON response via ``claude`` CLI with ``--json-schema``."""
        schema = _get_tool_schema(tool_name)

        cmd = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema),
            "--system-prompt",
            self.system_prompt,
            "--model",
            self.model,
            "--tools",
            "",
        ]

        raw = self._cli_run(cmd, prompt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            msg = f"Failed to parse CLI JSON output: {e}"
            raise AnalystError(msg) from e

        return self._unwrap_cli_json(data)

    @staticmethod
    def _unwrap_cli_json(data: Any) -> dict[str, Any]:
        """Unwrap the CLI JSON envelope to extract the actual result.

        ``claude -p --output-format json`` wraps responses in an envelope::

            {"type": "result", "result": "...", ...}

        The ``result`` field may be a JSON string (when ``--json-schema`` is used)
        or a plain dict. Handle both.
        """
        if isinstance(data, dict) and "result" in data:
            inner = data["result"]
            if isinstance(inner, str):
                try:
                    parsed = json.loads(inner)
                except json.JSONDecodeError:
                    msg = f"CLI result is not valid JSON: {inner[:200]}"
                    raise AnalystError(msg) from None
                if not isinstance(parsed, dict):
                    msg = f"Expected JSON object, got {type(parsed).__name__}"
                    raise AnalystError(msg)
                return dict(parsed)
            if isinstance(inner, dict):
                return dict(inner)
            msg = f"Unexpected result type from CLI: {type(inner).__name__}"
            raise AnalystError(msg)

        # No envelope — direct JSON object
        if isinstance(data, dict):
            return dict(data)

        msg = f"Unexpected CLI output type: {type(data).__name__}"
        raise AnalystError(msg)

    # ------------------------------------------------------------------
    # API backend — Anthropic SDK
    # ------------------------------------------------------------------

    def _api_text(self, content: str, use_tools: bool) -> str:
        """Free-form text response via Anthropic SDK."""
        import anthropic

        client = self._get_api_client()
        self._conversation.append({"role": "user", "content": content})

        try:
            if use_tools:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self.system_prompt,
                    messages=self._conversation,
                    tools=TRADING_TOOLS,
                )
            else:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self.system_prompt,
                    messages=self._conversation,
                )
        except anthropic.APIError as e:
            msg = f"Claude API error: {e}"
            raise AnalystError(msg) from e

        text_parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        reply = "\n".join(text_parts) if text_parts else ""
        self._conversation.append({"role": "assistant", "content": reply})

        return reply

    def _api_structured(self, tool_name: str, prompt: str) -> dict[str, Any]:
        """Structured output via Anthropic SDK ``tool_use``."""
        import anthropic

        client = self._get_api_client()
        prompt += f"\n\nUse the {tool_name} tool."

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
                tools=TRADING_TOOLS,
                tool_choice={"type": "tool", "name": tool_name},
            )
        except anthropic.APIError as e:
            msg = f"Claude API error: {e}"
            raise AnalystError(msg) from e

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return dict(block.input)

        msg = f"Claude did not return expected tool '{tool_name}'"
        raise AnalystError(msg)
