# Apex Oracle Bot 🤖⚡

Modern, high-frequency quantitative trading bot for crypto assets utilizing **Polars** dynamic market regime classification (Hurst Exponent, ATR, RSI), automated risk management with hard drawdown limits, and high-performance async order execution via Alpaca.

---

## 📐 Architecture Diagram

```mermaid
graph TD
    A[Market Data Poller / Alpaca API] --> B[TradingStrategy Engine]
    B -->|Calculate Hurst, ATR, RSI| C{Regime Classifier}
    C -->|Trending| D[Trend Pullback Signal]
    C -->|Mean Reverting| E[Oscillator Signal]
    C -->|High Volatility| F[Stand-Aside Filter]
    
    D --> G[RiskManager]
    E --> G
    
    G -->|Max Drawdown & Daily Loss Check| H{Risk OK?}
    H -->|Yes| I[Position Sizer & Multiplier]
    H -->|No / Killswitch| J[Liquidate All & Halt]
    
    I --> K[Alpaca Trading API]
    K --> L[FastAPI Metrics & Telegram Alerts]
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python**: `>= 3.12`
- **Alpaca Trading Account** (Paper or Live API keys)
- **uv** / **pip** package manager

### 2. Environment Setup
Copy `.env.example` to `.env` and fill in your configuration:
```bash
cp .env.example .env
```

Configure your credentials in `.env`:
```env
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 3. Local Execution
Run the trading bot:
```bash
python -m src.main
```

### 4. Running with Docker
Build and run using Docker:
```bash
docker build -t apex-oracle-bot .
docker run -d --name apex-bot --env-file .env apex-oracle-bot
```

---

## ⚙️ Risk Management Parameters

The bot features layered risk protection managed dynamically by `src/risk.py`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `ACCOUNT_BASE` | `$10,000` | Account valuation baseline for per-trade risk sizing |
| `BASE_RISK_PERCENT` | `1.0%` | Standard risk per trade |
| `MAX_SINGLE_TRADE_USD` | `$2,500` | Absolute hard dollar cap per trade |
| `MAX_PORTFOLIO_VALUE` | `$500` | Maximum combined open market exposure cap |
| `MAX_OPEN_POSITIONS` | `3` | Concurrent asset holdings limit |
| `MAX_DRAWDOWN_STOP` | `-10.0%` | Portfolio peak drawdown killswitch (liquidates all holdings) |
| `DAILY_LOSS_LIMIT` | `-3.0%` | Daily stop-loss threshold |
| `PROFIT_TARGET_PCT` | `3.0%` | Take-profit threshold |
| `STOP_LOSS_PCT` | `4.0%` | Stop-loss threshold |

---

## 📊 Market Regime Classification

The bot dynamically classifies market state into three distinct operational modes:

1. **Trending** ($Hurst > 0.60$):
   * Enters buy positions on deeper pullback bounces when RSI momentum recovers.
   * Leverages position sizing expansion ($1.5\times$).
2. **Mean-Reverting** ($Hurst < 0.58$):
   * Buys oversold RSI ($<30$) and shorts/sells overbought RSI ($>80$).
   * Uses conservative position sizing ($0.8\times$).
3. **High Volatility** ($\text{ATR} > 5.0\%$):
   * Immediately stands aside to preserve capital during market turbulence.

---

## 🧠 Adaptive ML Layer (Self-Evolving Committee Weighting)

An optional adaptive meta-learner sits **on top of** the 5-brain committee
(`transformer`, `quant`, `momentum`, `sentinel`, `llm`). It learns *which brain
to trust* per market regime from realized trade outcomes, then re-weights their
votes when combining them into a final action. Signal generation, sizing, and
execution are left intact.

### How it works
- Each committee decision at trade entry is snapshotted (brains' votes, regime,
  final action) with a `decision_id` (persisted in the `decision_snapshots` table).
- When the position exits and realized PnL is known, the outcome is matched back
  to its snapshot. A brain's per-regime weight is nudged up (exponential reward)
  when its directional vote matched the profitable direction, and down otherwise.
  Hold / stand-aside votes are treated conservatively (no change); flat PnL never
  moves weights.
- Weights are clamped to `[MIN, MAX]` and normalized per regime; updates to one
  regime never bleed into another. State is saved atomically as versioned JSON
  and falls back to equal weights if missing or corrupt.

### Safety model (risk.py stays authoritative)
- **Shadow mode by default.** With `ADAPTIVE_ML_ENABLED=false` (the default), the
  learner still computes and logs weights for observability, but the committee's
  **classic** decision is what trades — nothing changes in behavior.
- **Warm-up gate.** Even when enabled, learned weights only drive live decisions
  after `ADAPTIVE_MIN_TRADES_BEFORE_LIVE` realized outcomes; until then it stays
  in shadow mode.
- **Never bypasses risk.** The Sentinel hard veto, the `WINNING_SCORE_THRESHOLD`,
  position sizing, stop-loss, and the drawdown / daily-loss killswitch in
  `src/risk.py` remain fully authoritative. The ML layer can only re-weight votes;
  it cannot enlarge a position, skip a stop, or override a veto.

### Config flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `ADAPTIVE_ML_ENABLED` | `false` | Let learned weights drive decisions. Default = paper-only shadow mode. |
| `ADAPTIVE_STATE_PATH` | `data/adaptive_meta_state.json` | Atomically-persisted learner state (versioned JSON). |
| `ADAPTIVE_LEARNING_RATE` | `0.10` | Exponential reward rate for per-brain weight updates. |
| `ADAPTIVE_MIN_WEIGHT` | `0.02` | Lower clamp for any single brain weight per regime. |
| `ADAPTIVE_MAX_WEIGHT` | `0.60` | Upper clamp for any single brain weight per regime. |
| `ADAPTIVE_MIN_TRADES_BEFORE_LIVE` | `50` | Realized outcomes required before weights go live. |

Current brain weights, model version, sample count, and last-update time are
exported to Prometheus (`bot_adaptive_*`), and a Telegram alert fires when a
regime's weights change materially.

### Committee Latency & Decision Metrics

| Metric | Type | Description |
| :--- | :--- | :--- |
| `committee_decision_seconds` | Histogram | Latency of each `run_committee()` call, labeled by symbol |
| `committee_brain_weight` | Gauge | Per-brain weight in each committee decision, labeled by brain/action/symbol |
| `committee_score` | Histogram | Distribution of final committee scores, labeled by action (buy/sell/hold) |

---

## 🧠 5-Brain Committee Architecture

The decision engine is an ensemble of five specialized "brains" that vote on
each trading decision. Their votes are combined using **regime-dependent weighting**
and an optional **adaptive meta-learner** that learns which brains to trust
per regime from realized P&L.

```
┌─────────────────────────────────────────────────────────────┐
│                      run_committee()                         │
│                                                             │
│  ┌──────────┐  ┌───────┐  ┌──────────┐  ┌───────┐  ┌──────┐ │
│  │Transformer│  │Quant  │  │Momentum  │  │Sentinel│  │LLM   │ │
│  │(PyTorch)  │  │(Stat- │  │(Trend/   │  │(Crash │  │(Qual-│ │
│  │Grok GQA   │  │ arbitrage) │ Momentum)│  │ veto) │  │ itative)│ │
│  │v9)        │  │        │  │          │  │        │  │        │ │
│  └────┬─────┘  └────┬───┘  └────┬─────┘  └────┬───┘  └────┬─┘ │
│       │            │            │            │            │   │
│       └────────────┴────────────┴────────────┴────────────┘   │
│                          │                                    │
│           ┌──────────────▼──────────────┐                    │
│           │  Dynamic Weight Matrix      │                    │
│           │  (per regime: bull/bear/    │                    │
│           │   trending/sideways/etc.)    │                    │
│           └──────────────┬──────────────┘                    │
│                          │                                    │
│           ┌──────────────▼──────────────┐                    │
│           │  Adaptive Meta-Learner      │                    │
│           │  (shadow mode by default)    │                    │
│           └──────────────┬──────────────┘                    │
│                          │                                    │
│           ┌──────────────▼──────────────┐                    │
│           │  Final Decision +           │                    │
│           │  Position Sizing Multiplier │                    │
│           └─────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### Brain Details

| Brain | Module | Type | Weight | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Transformer** | `transformer_brain.py` | PyTorch GQA | 15–40% | Grok GQA v9 model for directional probability. Loaded lazily, runs in background thread. Includes MC-dropout uncertainty estimation and SHAP-style gradient attribution. |
| **Quant** | `quant_brain.py` | Statistical arb | 15–30% | Mean-reversion signal based on z-score of price relative to Bollinger Bands. |
| **Momentum** | `momentum_brain.py` | Trend following | 10–30% | Trend-following signal using rolling momentum and MACD. |
| **Sentinel** | `sentinel_brain.py` | Hard veto | 5–40% | Market regime filter and hard veto for crash/volatility events. Can force "stand_aside" when market stress is detected. |
| **LLM** | `llm_brain.py` | Qualitative | 5–25% | Rule-assisted qualitative review layer (stub; can be connected to Gemini/Groq/Claude for full reasoning). |

### Voting & Weighting

1. **Concurrent execution**: All 5 brains run concurrently via `asyncio.gather`
   (the transformer brain uses `asyncio.to_thread` for PyTorch inference).
2. **Regime weight matrix**: Each market regime (bull, bear, trending, sideways,
   high_volatility, low_volatility, neutral) has a static weight distribution
   defined in `REGIME_WEIGHT_MATRIX`.
3. **Adaptive override**: The optional `AdaptiveMetaLearner` computes learned
   weights per regime from historical outcomes. When `ADAPTIVE_ML_ENABLED=true`
   and both validation and warm-up gates pass, learned weights replace the static
   matrix; otherwise the classic matrix is used with learned weights logged as
   shadow.
4. **Entropy penalty**: Vote disagreement (Shannon entropy across brain actions)
   reduces the position size multiplier by up to 25%.
5. **Hard vetoes**: Sentinel can force `stand_aside` regardless of other votes
   when confidence ≥ 0.85 and the signal is `stand_aside`.

### Entry Criteria

| Criterion | Threshold |
| :--- | :--- |
| `score` (weighted) | ≥ per-symbol adaptive threshold (default 0.15) |
| `winner` action | Must be `buy` or `sell` (not `stand_aside`/`skip`/`hold`) |
| Net edge after costs | ≥ `TX_COST_MIN_EDGE_BPS` (default 10 bps round-trip) |
| Risk checks | `RiskManager` exposure, drawdown, and daily loss gates must pass |

## 📈 Backtesting & Parameter Sweeps

### Live Backtest Simulation
Run a historical simulation on past bars:
```bash
python run_live_backtest.py
```

### Grid-Search Parameter Optimization
Run parameter grid sweeps to optimize hold times, stop-loss percentages, and take-profit targets:
```bash
python sweep_params.py
```

---

## 🩺 Monitoring & Metrics API

When running, the bot launches an integrated **FastAPI** HTTP server (default port `8000`):

- **Health Check**: `GET http://localhost:8000/health`
- **Prometheus Metrics**: `GET http://localhost:8000/metrics`
- **Telegram Alerts**: Automatic real-time notification on order executions, trailing stops, and risk killswitches.

---

## 🧪 Running Tests

Execute the unit test suite with `pytest`:
```bash
python -m pytest
```
# test
