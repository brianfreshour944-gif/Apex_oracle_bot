# Apex Oracle Bot (OKX)

A regime-aware crypto spot trading bot for **OKX**. It classifies the market
(Hurst exponent + Kalman smoothing + RSI/MACD/ATR), generates BUY/SELL/HOLD
signals per regime, sizes positions with a half-Kelly + volatility model, and
enforces drawdown / daily-loss killswitches. It exposes a JSON health endpoint
and sends Telegram + Base44 notifications.

> ⚠️ Trading is risky. Start with `OKX_USE_DEMO=true` (paper) and validate
> behaviour before risking real funds. This is not financial advice.

## Architecture

| File | Responsibility |
|------|----------------|
| `bot.py` | Async main loop: snapshot account, risk checks, per-symbol entry/exit, status server. |
| `exchange.py` | ccxt OKX wrapper (balances, OHLCV, market orders, fills). Demo/sandbox aware. |
| `strategies.py` | Indicators + regime classification + signal generation. |
| `risk.py` | Kelly sizing, ATR/vol position sizing, drawdown + daily-loss killswitches. |
| `db.py` | SQLAlchemy ORM (`trades`, `bot_status`), trade logging, status upsert. |
| `notifier.py` | Telegram alerts + Base44 dashboard sync. |
| `config.py` | Env-driven configuration. |

**Position model:** OKX spot has no broker-side position object, so a "position"
is a non-dust balance of a base currency. Entry price / realized PnL are derived
from the bot's own trade log (`trades` table).

## Local run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in OKX_API_KEY / SECRET / PASSPHRASE
python bot.py
```

Health check while running: `curl http://localhost:8080/status`

## Docker

```bash
cp .env.example .env          # fill credentials
docker compose up --build -d
docker compose logs -f
```

The SQLite DB persists in the `bot_db` volume at `/app/data/trades.db`.

## Deploy on Coolify (→ Oracle Cloud VM)

1. Push this repo to GitHub.
2. In Coolify, **New Resource → Application → from your GitHub repo**.
3. Build pack: **Docker Compose** (this repo ships `docker-compose.yml`) or
   **Dockerfile**.
4. Set environment variables (from `.env.example`) in Coolify's **Environment
   Variables** UI — do **not** commit a real `.env`. At minimum:
   `OKX_API_KEY`, `OKX_SECRET_KEY`, `OKX_PASSPHRASE`, `OKX_USE_DEMO`.
5. Expose port **8080** and set the **health check path** to `/status`.
6. Deploy. Coolify will auto-redeploy on new commits to the tracked branch.

On the Oracle Cloud (Ampere/Always-Free) VM that hosts Coolify, ensure the
security list / firewall allows inbound traffic on the port Coolify maps, and
that Docker has enough disk for the persistent volume.

## Configuration

All settings are environment variables — see `.env.example` for the full list
with defaults (symbols, risk caps, profit/stop targets, loop interval, etc.).
