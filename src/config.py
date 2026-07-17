"""Modern configuration system using Pydantic Settings V2 with environment variable validation."""

from pathlib import Path
from typing import List, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field, field_validator

class TradingBotSettings(BaseSettings):
    """Modern Pydantic Settings V2 configuration with runtime validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
        env_prefix="",
    )

    # --- Alpaca Credentials (Crypto) ---
    ALPACA_API_KEY: str = Field(default="", description="Alpaca API key")
    ALPACA_SECRET_KEY: str = Field(default="", description="Alpaca secret key")
    ALPACA_BASE_URL: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca API base URL (paper trading by default)"
    )

    # --- Database configuration ---
    DATABASE_URL: str = Field(
        default_factory=lambda: f"sqlite:///{Path(__file__).parent.parent / 'data' / 'trades.db'}",
        description="Database connection URL"
    )

    # --- Base44 & Telegram Dashboards ---
    DASHBOARD_API_KEY: str = Field(default="", description="Base44 dashboard API key")
    BASE44_API_URL: str = Field(
        default="https://api.base44.com/api/apps/YOUR_APP_ID/entities",
        description="Base44 API URL"
    )
    TELEGRAM_BOT_TOKEN: str = Field(default="", description="Telegram bot token")
    TELEGRAM_CHAT_ID: str = Field(default="", description="Telegram chat ID")

    # --- Trading Configurations ---
    TRADING_SYMBOLS: str = Field(
        default="BTC/USD,ETH/USD,SOL/USD",
        description="Comma-separated list of trading symbols"
    )

    @computed_field
    @property
    def SYMBOLS(self) -> List[str]:
        """Parsed list of trading symbols."""
        return [s.strip() for s in self.TRADING_SYMBOLS.split(",") if s.strip()]

    QUOTE_CURRENCY: str = Field(
        default="USD",
        description="Quote currency for equity calculations"
    )

    # --- Risk Management Limits ---
    ACCOUNT_BASE: float = Field(
        default=10000.0,
        description="Base account value for risk calculations",
        gt=0
    )
    BASE_RISK_PERCENT: float = Field(
        default=0.01,
        description="Standard risk percentage per trade (1%)",
        ge=0,
        le=1
    )
    MAX_SINGLE_TRADE_USD: float = Field(
        default=150.0,
        description="Hard cap per trade size in USD",
        gt=0
    )
    MAX_PORTFOLIO_VALUE: float = Field(
        default=500.0,
        description="Maximum exposure cap across all positions",
        gt=0
    )
    MAX_OPEN_POSITIONS: int = Field(
        default=3,
        description="Maximum concurrent assets held",
        ge=1,
        le=10
    )
    MAX_DRAWDOWN_STOP: float = Field(
        default=-10.0,
        description="Killswitch at -10% equity drawdown",
        lt=0
    )
    DAILY_LOSS_LIMIT: float = Field(
        default=-3.0,
        description="Daily stop loss limit",
        lt=0
    )

    # --- Strategy Specific Parameters ---
    PROFIT_TARGET_PCT: float = Field(
        default=0.03,
        description="Take profit percentage (3%)",
        gt=0,
        le=0.5
    )
    STOP_LOSS_PCT: float = Field(
        default=0.02,
        description="Stop loss percentage (2%)",
        gt=0,
        le=0.5
    )
    ATR_STOP_MULTIPLIER: float = Field(
        default=2.0,
        description="ATR multiple used for position sizing",
        gt=0
    )
    MAX_HOLD_HOURS: float = Field(
        default=8.0,
        description="Time limit to hold open positions in hours",
        gt=0
    )
    DUST_VALUE_USD: float = Field(
        default=1.50,
        description="Below this market value a holding is considered dust",
        gt=0
    )

    # --- Regime Classification Thresholds ---
    HURST_TREND_UP: float = Field(
        default=0.52,
        description="Hurst exponent threshold for trending regime",
        gt=0,
        lt=1
    )
    HURST_MEAN_REVERT: float = Field(
        default=0.48,
        description="Hurst exponent threshold for mean-reversion regime",
        gt=0,
        lt=1
    )
    HIGH_VOLATILITY_PCT: float = Field(
        default=5.0,
        description="ATR% above this triggers stand-aside mode",
        gt=0
    )
    RSI_OVERBOUGHT: float = Field(
        default=70.0,
        description="RSI ceiling for buys/sell trigger",
        gt=0,
        le=100
    )
    RSI_OVERSOLD: float = Field(
        default=30.0,
        description="RSI floor for sells/buy trigger",
        gt=0,
        lt=100
    )
    RSI_NEUTRAL_BUY: float = Field(
        default=35.0,
        description="RSI buy trigger in neutral regime",
        gt=0,
        lt=100
    )
    RSI_NEUTRAL_SELL: float = Field(
        default=65.0,
        description="RSI sell trigger in neutral regime",
        gt=0,
        lt=100
    )

    # --- Trailing Stop Configuration ---
    TRAILING_STOP_ENABLED: bool = Field(
        default=True,
        description="Enable trailing stop loss"
    )
    TRAILING_ACTIVATION_PCT: float = Field(
        default=0.015,
        description="Activate trailing stop once in +1.5% profit",
        gt=0,
        le=0.1
    )
    TRAILING_DISTANCE_PCT: float = Field(
        default=0.01,
        description="Exit if price falls 1.0% off its peak",
        gt=0,
        le=0.1
    )

    # --- Loop / Server Configuration ---
    LOOP_INTERVAL_SEC: int = Field(
        default=60,
        description="Seconds between full trading cycles",
        ge=10,
        le=300
    )
    STATUS_PORT: int = Field(
        default=8080,
        description="Port for HTTP status server",
        ge=1024,
        le=65535
    )
    BOT_NAME: str = Field(
        default="apex_oracle_bot",
        description="Bot name for identification"
    )

    # --- Validation ---
    @field_validator("HURST_TREND_UP", "HURST_MEAN_REVERT")
    def validate_hurst_thresholds(cls, v: float) -> float:
        """Ensure Hurst thresholds are in valid range."""
        if not (0 < v < 1):
            raise ValueError("Hurst exponent must be between 0 and 1")
        return v

    @field_validator("RSI_OVERBOUGHT", "RSI_OVERSOLD", "RSI_NEUTRAL_BUY", "RSI_NEUTRAL_SELL")
    def validate_rsi_thresholds(cls, v: float) -> float:
        """Ensure RSI thresholds are in valid range."""
        if not (0 <= v <= 100):
            raise ValueError("RSI values must be between 0 and 100")
        return v

    def log_config(self) -> str:
        """Generate configuration summary for logging."""
        config_lines = [
            "================== CONFIGURATION SUMMARY ====================",
            f"Bot Name: {self.BOT_NAME}",
            f"Exchange: Alpaca Crypto (Paper={self.ALPACA_BASE_URL.endswith('paper-api.alpaca.markets')})",
            f"Database URL: {self.DATABASE_URL}",
            f"Traded Assets: {self.SYMBOLS} (quote: {self.QUOTE_CURRENCY})",
            f"Risk Per Trade: {self.BASE_RISK_PERCENT*100:.2f}% (Max USD: ${self.MAX_SINGLE_TRADE_USD})",
            f"Max Portfolio Value Cap: ${self.MAX_PORTFOLIO_VALUE} | Max Open Positions: {self.MAX_OPEN_POSITIONS}",
            f"Drawdown Limit: {self.MAX_DRAWDOWN_STOP}% | Daily Limit: {self.DAILY_LOSS_LIMIT}%",
            f"Target: +{self.PROFIT_TARGET_PCT*100:.2f}% | Stop Loss: -{self.STOP_LOSS_PCT*100:.2f}% | Max Hold: {self.MAX_HOLD_HOURS}h",
            f"Regime: Hurst> {self.HURST_TREND_UP} trend, < {self.HURST_MEAN_REVERT} mean-rev, Vol> {self.HIGH_VOLATILITY_PCT}% stand-aside",
            f"Trailing Stop: {'ON' if self.TRAILING_STOP_ENABLED else 'OFF'} "
            f"(activate +{self.TRAILING_ACTIVATION_PCT*100:.2f}%, distance {self.TRAILING_DISTANCE_PCT*100:.2f}%)",
            f"Telegram Configured: {bool(self.TELEGRAM_BOT_TOKEN and self.TELEGRAM_CHAT_ID)}",
            f"Base44 Dashboard Configured: {bool(self.DASHBOARD_API_KEY)}",
            f"Alpaca Credentials Present: {bool(self.ALPACA_API_KEY and self.ALPACA_SECRET_KEY)}",
            "===========================================================",
        ]
        return "\n".join(config_lines)

# Global settings instance
settings = TradingBotSettings()

# Backward compatibility with old config.py imports
ALPACA_API_KEY = settings.ALPACA_API_KEY
ALPACA_SECRET_KEY = settings.ALPACA_SECRET_KEY
ALPACA_BASE_URL = settings.ALPACA_BASE_URL
DATABASE_URL = settings.DATABASE_URL
DASHBOARD_API_KEY = settings.DASHBOARD_API_KEY
BASE44_API_URL = settings.BASE44_API_URL
TELEGRAM_BOT_TOKEN = settings.TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID = settings.TELEGRAM_CHAT_ID
SYMBOLS = settings.SYMBOLS
QUOTE_CURRENCY = settings.QUOTE_CURRENCY
ACCOUNT_BASE = settings.ACCOUNT_BASE
BASE_RISK_PERCENT = settings.BASE_RISK_PERCENT
MAX_SINGLE_TRADE_USD = settings.MAX_SINGLE_TRADE_USD
MAX_PORTFOLIO_VALUE = settings.MAX_PORTFOLIO_VALUE
MAX_OPEN_POSITIONS = settings.MAX_OPEN_POSITIONS
MAX_DRAWDOWN_STOP = settings.MAX_DRAWDOWN_STOP
DAILY_LOSS_LIMIT = settings.DAILY_LOSS_LIMIT
PROFIT_TARGET_PCT = settings.PROFIT_TARGET_PCT
STOP_LOSS_PCT = settings.STOP_LOSS_PCT
ATR_STOP_MULTIPLIER = settings.ATR_STOP_MULTIPLIER
MAX_HOLD_HOURS = settings.MAX_HOLD_HOURS
DUST_VALUE_USD = settings.DUST_VALUE_USD
HURST_TREND_UP = settings.HURST_TREND_UP
HURST_MEAN_REVERT = settings.HURST_MEAN_REVERT
HIGH_VOLATILITY_PCT = settings.HIGH_VOLATILITY_PCT
RSI_OVERBOUGHT = settings.RSI_OVERBOUGHT
RSI_OVERSOLD = settings.RSI_OVERSOLD
RSI_NEUTRAL_BUY = settings.RSI_NEUTRAL_BUY
RSI_NEUTRAL_SELL = settings.RSI_NEUTRAL_SELL
TRAILING_STOP_ENABLED = settings.TRAILING_STOP_ENABLED
TRAILING_ACTIVATION_PCT = settings.TRAILING_ACTIVATION_PCT
TRAILING_DISTANCE_PCT = settings.TRAILING_DISTANCE_PCT
LOOP_INTERVAL_SEC = settings.LOOP_INTERVAL_SEC
STATUS_PORT = settings.STATUS_PORT
BOT_NAME = settings.BOT_NAME
log_config = settings.log_config