"""Configuration loading from config.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.toml"


class IBKRConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    timeout: int = 30
    account: str = ""


class ClaudeConfig(BaseModel):
    model: str = "claude-opus-4-6"
    max_tokens: int = 4096
    backend: str = "cli"  # "cli" (Claude Max via claude CLI) or "api" (ANTHROPIC_API_KEY)


class RiskConfig(BaseModel):
    max_position_pct: float = 40.0
    max_sector_pct: float = 60.0
    max_daily_loss_pct: float = 5.0
    max_leverage: float = 2.0
    require_stop_loss: bool = True
    allow_naked_options: bool = False
    paper_trading: bool = True


class WatchlistConfig(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "XLE", "XLF", "AAPL"])


class DatabaseConfig(BaseModel):
    path: str = "data/trades.db"


class AppConfig(BaseModel):
    ibkr: IBKRConfig = Field(default_factory=IBKRConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    watchlist: WatchlistConfig = Field(default_factory=WatchlistConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from a TOML file. Falls back to defaults if file missing."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return AppConfig()
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    return AppConfig(**data)
