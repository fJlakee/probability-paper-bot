from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml


@dataclass
class BinanceConfig:
    base_url: str = "https://fapi.binance.com"
    interval: str = "15m"
    history_bars: int = 17280
    universe_size: int = 100
    download_workers: int = 10
    incremental_bars: int = 5
    universe_refresh_minutes: int = 60
    quote_asset: str = "USDT"
    symbol_allowlist: list[str] = field(default_factory=list)
    symbol_denylist: list[str] = field(default_factory=list)
    exclude_peg_like: bool = True
    peg_price_min: float = 0.95
    peg_price_max: float = 1.05
    peg_max_atr_pct: float = 0.003


@dataclass
class ModelConfig:
    horizon_bars: int = 12
    take_profit_pct: float = 0.01
    stop_loss_pct: float = 0.004
    min_entry_probability: float = 0.70
    validation_signal_threshold: float = 0.55
    min_training_rows: int = 400
    calibration_fraction: float = 0.20
    test_fraction: float = 0.20
    min_calibration_samples: int = 200
    min_validation_trades: int = 100
    min_validation_precision: float = 0.52
    random_state: int = 42
    retrain_minutes: int = 360
    prefilter_size: int = 25
    model_workers: int = 4
    backtest_training_days: int = 90
    backtest_retrain_days: int = 7
    backtest_universe_size: int = 25


@dataclass
class ExecutionConfig:
    initial_capital: float = 450.0
    leverage: int = 100
    notional_fraction: float = 1.0
    taker_fee_rate: float = 0.0005
    slippage_rate: float = 0.0002
    maintenance_margin_rate: float = 0.004
    funding_rate_per_8h: float = 0.0001
    scan_seconds: int = 60
    max_position_bars: int = 12
    min_expected_return_on_equity: float = 0.0001
    liquidation_penalty_rate: float = 0.002


@dataclass
class TelegramConfig:
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""


@dataclass
class StorageConfig:
    database: str = "paperbot.sqlite3"
    model_directory: str = "models"
    report_directory: str = "reports"


@dataclass
class Config:
    binance: BinanceConfig
    model: ModelConfig
    execution: ExecutionConfig
    telegram: TelegramConfig
    storage: StorageConfig


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    telegram = raw.get("telegram", {})
    telegram["bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN", telegram.get("bot_token", ""))
    telegram["chat_id"] = os.getenv("TELEGRAM_CHAT_ID", telegram.get("chat_id", ""))
    cfg = Config(
        binance=BinanceConfig(**raw.get("binance", {})),
        model=ModelConfig(**raw.get("model", {})),
        execution=ExecutionConfig(**raw.get("execution", {})),
        telegram=TelegramConfig(**telegram),
        storage=StorageConfig(**raw.get("storage", {})),
    )
    if not 0 < cfg.execution.notional_fraction <= 1:
        raise ValueError("execution.notional_fraction must be in (0, 1]")
    if cfg.execution.leverage < 1:
        raise ValueError("execution.leverage must be positive")
    if cfg.model.stop_loss_pct <= 0 or cfg.model.take_profit_pct <= 0:
        raise ValueError("TP and SL must be positive")
    if cfg.model.calibration_fraction + cfg.model.test_fraction >= 0.5:
        raise ValueError("calibration_fraction + test_fraction must leave at least 50% for training")
    return cfg
