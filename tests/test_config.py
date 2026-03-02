"""Tests for configuration loading."""

from pathlib import Path

from scry_trader.config import AppConfig, load_config


class TestConfig:
    def test_default_config(self) -> None:
        config = AppConfig()
        assert config.ibkr.port == 7497
        assert config.claude.model == "claude-opus-4-6"
        assert config.risk.max_position_pct == 40.0
        assert config.risk.paper_trading is True

    def test_load_missing_file(self) -> None:
        config = load_config(Path("/nonexistent/config.toml"))
        assert config.ibkr.port == 7497

    def test_load_real_config(self) -> None:
        config_path = Path(__file__).parent.parent / "config.toml"
        if config_path.exists():
            config = load_config(config_path)
            assert config.ibkr.host == "127.0.0.1"
            assert config.risk.require_stop_loss is True
