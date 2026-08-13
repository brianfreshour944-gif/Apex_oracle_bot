"""Modern configuration system using Pydantic Settings V2 with environment variable validation."""

from pathlib import Path
from typing import List, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field, field_validator, model_validator

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
        default="sqlite:///data/bot.db",
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

    # --- LLM API Keys ---
    GROQ_API_KEY: str = Field(default="", description="Groq API key for Sentiment Analysis")
    GEMINI_API_KEY: str = Field(default="", description="Gemini API key for Sentiment Analysis")



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
        default=2500.0,
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
        default=0.04,
        description="Stop loss percentage (4%)",
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
    COOLDOWN_SECONDS_BUY: int = Field(
        default=300,
        description="Seconds to block a new entry on a symbol right after closing "
                    "a position on it. Without this, a strategy that both opened and "
                    "closed a position on the same cycle's re-evaluation (or a fresh "
                    "signal on the very next cycle) can immediately re-enter, producing "
                    "rapid open/close churn that bleeds to spread/slippage on every "
                    "round trip instead of a position actually playing out.",
        ge=0
    )

    # --- Regime Classification Thresholds ---
    HURST_TREND_UP: float = Field(
        default=0.60,
        description="Hurst exponent threshold for trending regime (adjusted for R/S estimator bias)",
        gt=0,
        lt=1
    )
    HURST_MEAN_REVERT: float = Field(
        default=0.58,
        description="Hurst exponent threshold for mean-reversion regime (adjusted for R/S estimator bias)",
        gt=0,
        lt=1
    )
    HIGH_VOLATILITY_PCT: float = Field(
        default=12.0,
        description="ATR% above this triggers stand-aside mode",
        gt=0
    )
    RSI_OVERBOUGHT: float = Field(
        default=80.0,
        description="RSI ceiling for buys/sell trigger",
        gt=0,
        le=100
    )
    RSI_OVERSOLD: float = Field(
        default=25.0,
        description="RSI floor for sells/buy trigger",
        gt=0,
        lt=100
    )
    RSI_NEUTRAL_BUY: float = Field(
        default=25.0,
        description="RSI buy trigger in neutral regime",
        gt=0,
        lt=100
    )
    RSI_NEUTRAL_SELL: float = Field(
        default=55.0,
        description="RSI sell trigger in neutral regime",
        gt=0,
        lt=100
    )
    BB_ZSCORE_THRESHOLD: float = Field(
        default=1.5,
        description="Price z-score threshold for mean-reversion entries in sideways/chop (standard deviations from rolling mean)",
        gt=0,
        le=3.0
    )

    # --- Trailing Stop Configuration ---
    TRAILING_STOP_ENABLED: bool = Field(
        default=True,
        description="Enable trailing stop loss"
    )
    TRAILING_ACTIVATION_PCT: float = Field(
        default=0.04,
        description="Activate trailing stop once in +4.0% profit",
        gt=0,
        le=0.2
    )
    TRAILING_DISTANCE_PCT: float = Field(
        default=0.03,
        description="Exit if price falls 3.0% off its peak",
        gt=0,
        le=0.2
    )

    # --- Loop / Server Configuration ---
    LOOP_INTERVAL_SEC: int = Field(
        default=60,
        description="Seconds between full trading cycles",
        ge=10,
        le=300
    )
    STATUS_PORT: int = Field(
        default=8000,
        description="Port for HTTP status server",
        ge=1024,
        le=65535
    )

    BOT_NAME: str = Field(
        default="apex_oracle_bot",
        description="Bot name for identification"
    )

    # --- Position Sizing (Kelly / volatility-adjusted) ---
    KELLY_FRACTION: float = Field(
        default=0.25,
        description="Fractional Kelly multiplier applied to raw Kelly size (0=flat, 1=full Kelly)",
        ge=0,
        le=1,
    )
    VOL_LOOKBACK: int = Field(
        default=20,
        description="Bars used to estimate realized volatility for sizing",
        ge=5,
    )
    MAX_VOL_ADJUST: float = Field(
        default=2.0,
        description="Cap on volatility multiplier (higher vol -> smaller size)",
        gt=0,
    )

    # --- Circuit Breaker / Fault Tolerance ---
    CIRCUIT_FAILURE_THRESHOLD: int = Field(
        default=5,
        description="Consecutive exchange failures before opening the circuit breaker",
        ge=1,
    )
    CIRCUIT_OPEN_SECONDS: int = Field(
        default=300,
        description="Seconds the circuit stays open after tripping",
        ge=10,
    )
    MAX_ORDER_RETRIES: int = Field(
        default=3,
        description="Max attempts to place/confirm an order",
        ge=1,
        le=10,
    )
    ORDER_TIMEOUT_SEC: int = Field(
        default=30,
        description="Seconds to wait for an order fill confirmation before cancelling",
        ge=5,
    )

    # --- Additional Killswitches ---
    VOLATILITY_SPIKE_PCT: float = Field(
        default=15.0,
        description="Portfolio realized-vol spike (% over lookback) that trips killswitch",
        gt=0,
    )
    API_DOWN_MINUTES: int = Field(
        default=10,
        description="Consecutive API unavailability (minutes) that trips killswitch",
        ge=1,
    )

    # --- Observability / Alerts ---
    ENABLE_PROMETHEUS: bool = Field(
        default=True,
        description="Expose Prometheus /metrics endpoint",
    )
    ALERT_EMAIL: str = Field(
        default="",
        description="Fallback email for critical alerts (smtp not configured by default)",
    )
    ALERT_COOLDOWN_SEC: int = Field(
        default=300,
        description="Minimum seconds between repeated critical alerts",
        ge=30,
    )

    # --- Adaptive Meta-Learner (self-evolving brain weighting) ---
    # Sits on top of the committee; learns which brain to trust per regime from
    # realized outcomes. risk.py stays authoritative — this never bypasses the
    # drawdown/daily-loss killswitch, order sizing, or stop-loss logic.
    ADAPTIVE_ML_ENABLED: bool = Field(
        default=True,
        description="Let the meta-learner drive the committee decision. "
                    "Default False = paper-only shadow mode (computed, logged, but not acted on).",
    )
    ADAPTIVE_STATE_PATH: str = Field(
        default="data/adaptive_meta_state.json",
        description="Path to the atomically-persisted adaptive learner JSON state",
    )
    ADAPTIVE_LEARNING_RATE: float = Field(
        default=0.10,
        description="Exponential reward learning rate for per-brain weight updates",
        gt=0,
        le=1,
    )
    ADAPTIVE_MIN_WEIGHT: float = Field(
        default=0.02,
        description="Lower clamp for any single brain weight within a regime",
        gt=0,
        lt=1,
    )
    ADAPTIVE_MAX_WEIGHT: float = Field(
        default=0.60,
        description="Upper clamp for any single brain weight within a regime",
        gt=0,
        le=1,
    )
    ADAPTIVE_MIN_TRADES_BEFORE_LIVE: int = Field(
        default=10,
        description="Realized outcomes required before adaptive weights drive live decisions "
                    "(below this the learner runs in shadow mode even when enabled)",
        ge=0,
    )
    PPO_MIN_TRADES_BEFORE_LIVE: int = Field(
        default=1,
        description="Realized outcomes required before PPO RL meta-learner drives live decisions. "
                    "PPO is pre-trained offline via evolutionary pipeline; gate mainly ensures model file exists.",
        ge=0,
    )

    # --- Machine Learning Paths ---
    TRANSFORMER_MODEL_PATH: str = Field(
        default="models/grok_gqa_v9_best.pth",
        description="Path to PyTorch model weights"
    )
    TRANSFORMER_SCALER_PATH: str = Field(
        default="models/feature_scaler.pkl",
        description="Path to feature scaler"
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

    @field_validator("ALPACA_BASE_URL")
    @classmethod
    def clean_alpaca_base_url(cls, v: str) -> str:
        """Clean the Alpaca base URL by stripping surrounding parentheses, quotes, and whitespace."""
        if v:
            v = v.strip("()\"' ")
            if v and not v.startswith(("http://", "https://")):
                v = "https://" + v
        return v

    @model_validator(mode="after")
    def validate_credentials(self) -> "TradingBotSettings":
        """Ensure Alpaca credentials are provided."""
        # Note: In a real test environment, we might bypass this, but for the bot, it's strictly required
        if not self.ALPACA_API_KEY or not self.ALPACA_SECRET_KEY:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY are required to run the bot. "
                "Please set them in your environment variables or .env file."
            )
        return self

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
COOLDOWN_SECONDS_BUY = settings.COOLDOWN_SECONDS_BUY
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
ADAPTIVE_ML_ENABLED = settings.ADAPTIVE_ML_ENABLED
ADAPTIVE_STATE_PATH = settings.ADAPTIVE_STATE_PATH
ADAPTIVE_LEARNING_RATE = settings.ADAPTIVE_LEARNING_RATE
ADAPTIVE_MIN_WEIGHT = settings.ADAPTIVE_MIN_WEIGHT
ADAPTIVE_MAX_WEIGHT = settings.ADAPTIVE_MAX_WEIGHT
ADAPTIVE_MIN_TRADES_BEFORE_LIVE = settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE
log_config = settings.log_config