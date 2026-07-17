# config.py
import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# --- OKX Credentials ---
OKX_API_KEY = os.getenv("OKX_API_KEY", "")
OKX_SECRET_KEY = os.getenv("OKX_SECRET_KEY", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")
# Default to demo / sandbox trading mode (paper). Set OKX_USE_DEMO=false for live.
OKX_USE_DEMO = os.getenv("OKX_USE_DEMO", "true").lower() == "true"

# --- Database configuration ---
# Default to SQLite local file inside the bot folder
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "trades.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# --- Base44 & Telegram Dashboards ---
DASHBOARD_API_KEY = os.getenv("DASHBOARD_API_KEY", "")
BASE44_API_URL = os.getenv("BASE44_API_URL", "https://api.base44.com/api/apps/YOUR_APP_ID/entities")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- Trading Configurations ---
# Top high-liquidity crypto assets quoted in USDT on OKX spot.
SYMBOLS = [s.strip() for s in os.getenv("TRADING_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",") if s.strip()]

# Quote currency the bot keeps its cash in (used for equity / buying-power math).
QUOTE_CURRENCY = os.getenv("QUOTE_CURRENCY", "USDT")

# --- Risk Management Limits ---
ACCOUNT_BASE = float(os.getenv("ACCOUNT_BASE", 10000.0))
BASE_RISK_PERCENT = float(os.getenv("BASE_RISK_PERCENT", 0.01))        # 1% standard risk per trade
MAX_SINGLE_TRADE_USD = float(os.getenv("MAX_SINGLE_TRADE_USD", 150.0)) # Hard cap per trade size
MAX_PORTFOLIO_VALUE = float(os.getenv("MAX_PORTFOLIO_VALUE", 500.0))   # Max exposure cap across all positions
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", 3))          # Max concurrent assets held
MAX_DRAWDOWN_STOP = float(os.getenv("MAX_DRAWDOWN_STOP", -10.0))       # Killswitch at -10% equity drawdown
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", -3.0))         # Daily stop loss limit

# --- Strategy Specific Parameters ---
PROFIT_TARGET_PCT = float(os.getenv("PROFIT_TARGET_PCT", 0.03))       # 3% take profit for exits
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", 0.02))               # 2% stop loss for exits
ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER", 2.0))    # ATR multiple used for position sizing
MAX_HOLD_HOURS = float(os.getenv("MAX_HOLD_HOURS", 8.0))               # Time limit to hold open positions
DUST_VALUE_USD = float(os.getenv("DUST_VALUE_USD", 1.50))             # Below this market value a holding is "dust"

# --- Loop / Server ---
LOOP_INTERVAL_SEC = int(os.getenv("LOOP_INTERVAL_SEC", 60))          # Seconds between full cycles
STATUS_PORT = int(os.getenv("STATUS_PORT", 8080))                    # Port for HTTP status server
BOT_NAME = os.getenv("BOT_NAME", "apex_oracle_bot")

# --- Print configuration summary for logs ---
def log_config():
    print("================== CONFIGURATION SUMMARY ==================")
    print(f"Bot Name: {BOT_NAME}")
    print(f"Exchange: OKX Spot (Demo/Sandbox Mode={OKX_USE_DEMO})")
    print(f"Database URL: {DATABASE_URL}")
    print(f"Traded Assets: {SYMBOLS} (quote: {QUOTE_CURRENCY})")
    print(f"Risk Per Trade: {BASE_RISK_PERCENT*100:.2f}% (Max USD: ${MAX_SINGLE_TRADE_USD})")
    print(f"Max Portfolio Value Cap: ${MAX_PORTFOLIO_VALUE} | Max Open Positions: {MAX_OPEN_POSITIONS}")
    print(f"Drawdown Limit: {MAX_DRAWDOWN_STOP}% | Daily Limit: {DAILY_LOSS_LIMIT}%")
    print(f"Target: +{PROFIT_TARGET_PCT*100:.2f}% | Stop Loss: -{STOP_LOSS_PCT*100:.2f}% | Max Hold: {MAX_HOLD_HOURS}h")
    print(f"Telegram Configured: {bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)}")
    print(f"Base44 Dashboard Configured: {bool(DASHBOARD_API_KEY)}")
    print(f"OKX Credentials Present: {bool(OKX_API_KEY and OKX_SECRET_KEY and OKX_PASSPHRASE)}")
    print("===========================================================")
