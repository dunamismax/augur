"""Tool definitions for Claude structured output via tool_use."""

from __future__ import annotations

from typing import Any

# Claude tool definitions for structured trading analysis output.
# These are passed to the Anthropic SDK as the `tools` parameter.

TRADING_TOOLS: list[dict[str, Any]] = [
    {
        "name": "analyze_trade",
        "description": (
            "Provide a structured analysis of a potential trade. Include bull/bear case, "
            "risk assessment, entry/exit levels, and conviction level. Use this whenever "
            "the user asks about a specific ticker or trade idea."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol"},
                "direction": {
                    "type": "string",
                    "enum": ["long", "short"],
                    "description": "Trade direction",
                },
                "conviction": {
                    "type": "string",
                    "enum": ["high", "medium", "low", "none"],
                    "description": "Conviction level for this trade",
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "moderate", "high", "extreme"],
                    "description": "Risk level assessment",
                },
                "bull_case": {
                    "type": "string",
                    "description": "The case for why this trade works",
                },
                "bear_case": {
                    "type": "string",
                    "description": "The case for why this trade fails",
                },
                "catalyst": {
                    "type": "string",
                    "description": "What triggers the move (earnings, macro event, etc.)",
                },
                "risk_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific risks to watch",
                },
                "entry_price": {
                    "type": "number",
                    "description": "Recommended entry price",
                },
                "target_price": {
                    "type": "number",
                    "description": "Price target",
                },
                "stop_loss_price": {
                    "type": "number",
                    "description": "Stop-loss level",
                },
                "reward_risk_ratio": {
                    "type": "number",
                    "description": "Reward-to-risk ratio (e.g. 2.5 means 2.5:1)",
                },
                "recommended_position_size": {
                    "type": "number",
                    "description": "Recommended number of shares/contracts",
                },
                "recommended_portfolio_pct": {
                    "type": "number",
                    "description": "Recommended position as % of portfolio",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Detailed reasoning for the recommendation",
                },
            },
            "required": [
                "symbol",
                "direction",
                "conviction",
                "risk_level",
                "bull_case",
                "bear_case",
                "reasoning",
            ],
        },
    },
    {
        "name": "recommend_position_size",
        "description": (
            "Calculate and recommend an optimal position size given portfolio context, "
            "risk tolerance, and trade parameters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "shares": {
                    "type": "number",
                    "description": "Recommended number of shares",
                },
                "dollar_amount": {
                    "type": "number",
                    "description": "Dollar value of the position",
                },
                "portfolio_percent": {
                    "type": "number",
                    "description": "Position as percentage of total portfolio",
                },
                "risk_per_share": {
                    "type": "number",
                    "description": "Dollar risk per share (entry - stop)",
                },
                "total_risk": {
                    "type": "number",
                    "description": "Total dollar risk on this position",
                },
                "reasoning": {"type": "string"},
            },
            "required": [
                "symbol",
                "shares",
                "dollar_amount",
                "portfolio_percent",
                "reasoning",
            ],
        },
    },
    {
        "name": "construct_order",
        "description": (
            "Construct a complete order specification with entry, stop-loss, and "
            "take-profit levels. Returns an order ready for review and submission."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "action": {
                    "type": "string",
                    "enum": ["BUY", "SELL"],
                },
                "quantity": {"type": "number"},
                "order_type": {
                    "type": "string",
                    "enum": ["market", "limit", "stop", "stop_limit"],
                },
                "limit_price": {"type": "number"},
                "stop_price": {"type": "number"},
                "take_profit_price": {"type": "number"},
                "stop_loss_price": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["symbol", "action", "quantity", "order_type", "reason"],
        },
    },
    {
        "name": "assess_portfolio_risk",
        "description": (
            "Perform a comprehensive risk assessment of the current portfolio. "
            "Identify concentration risks, correlation risks, and overall exposure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "overall_risk": {
                    "type": "string",
                    "enum": ["low", "moderate", "high", "extreme"],
                },
                "total_exposure": {
                    "type": "number",
                    "description": "Total invested as % of portfolio",
                },
                "cash_percent": {"type": "number"},
                "largest_position_pct": {"type": "number"},
                "largest_position_symbol": {"type": "string"},
                "sector_concentrations": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Sector name -> percentage of portfolio",
                },
                "correlation_warnings": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "reasoning": {"type": "string"},
            },
            "required": [
                "overall_risk",
                "total_exposure",
                "cash_percent",
                "largest_position_pct",
                "recommendations",
                "reasoning",
            ],
        },
    },
    {
        "name": "generate_trade_journal_entry",
        "description": (
            "Generate a structured trade journal entry capturing the thesis, "
            "analysis, and key decision factors for record keeping."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "direction": {
                    "type": "string",
                    "enum": ["long", "short"],
                },
                "thesis": {
                    "type": "string",
                    "description": "Why this trade was entered",
                },
                "entry_price": {"type": "number"},
                "stop_loss_price": {"type": "number"},
                "target_price": {"type": "number"},
                "shares": {"type": "number"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization (e.g. 'energy', 'swing', 'earnings')",
                },
                "notes": {"type": "string"},
            },
            "required": ["ticker", "direction", "thesis"],
        },
    },
]
