"""Net-expectancy research harness for Apex_oracle_bot. RESEARCH ONLY.

Standalone by design: this module does NOT import ``src/`` (so it cannot mutate
config defaults, the DB, or production state) and copies the handful of
constants it needs from ``src/config.py`` / ``src/strategies.py`` with the source
line noted next to each. It answers one question:

    Does any long-only spot design on BTC/ETH/SOL have positive NET expectancy
    after fees, on data it was not tuned on?

Win rate is deliberately NOT the success metric. The pre-registered candidate
list, cost model, dev/holdout split, and pass criteria live in
``research/RESULTS.md`` (Section 0) and were fixed before any result was seen.

Usage:
    python -m research.net_expectancy fetch     # download + cache bars/funding
    python -m research.net_expectancy dev       # dev-only: baselines + candidates
    python -m research.net_expectancy holdout   # one frozen run per family
"""

from __future__ import annotations

import argparse
import io
import json
import os
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
DEV_RESULTS = os.path.join(HERE, "DEV_RESULTS.json")
HOLDOUT_RESULTS = os.path.join(HERE, "HOLDOUT_RESULTS.json")

SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]
PERP = {"BTC/USD": "BTCUSDT", "ETH/USD": "ETHUSDT", "SOL/USD": "SOLUSDT"}

ALPACA_BARS = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
BINANCE_FUNDING = (
    "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
    "{sym}/{sym}-fundingRate-{ym}.zip"
)

# ── Constants replicated from src/config.py (NOT imported) ──────────────────
PROFIT_TARGET_PCT = 0.03      # src/config.py PROFIT_TARGET_PCT default
STOP_LOSS_PCT = 0.04          # src/config.py STOP_LOSS_PCT default
MAX_HOLD_HOURS = 8.0          # src/config.py MAX_HOLD_HOURS default
MIN_HOLD_MINUTES = 30         # src/config.py MIN_HOLD_MINUTES default
HURST_TREND_UP = 0.60         # src/config.py HURST_TREND_UP
HURST_MEAN_REVERT = 0.58      # src/config.py HURST_MEAN_REVERT
HIGH_VOLATILITY_PCT = 5.0     # src/config.py HIGH_VOLATILITY_PCT
RSI_OVERBOUGHT = 80.0         # src/config.py RSI_OVERBOUGHT
RSI_OVERSOLD = 25.0           # src/config.py RSI_OVERSOLD
RSI_NEUTRAL_BUY = 25.0        # src/config.py RSI_NEUTRAL_BUY
RSI_NEUTRAL_SELL = 55.0       # src/config.py RSI_NEUTRAL_SELL
BB_ZSCORE_THRESHOLD = 1.5     # src/config.py BB_ZSCORE_THRESHOLD
MIN_WEIGHT = 0.01             # src/strategy_selector.py StrategyMetaLearner min_weight
MAX_WEIGHT = 0.80             # src/strategy_selector.py StrategyMetaLearner max_weight

# src/strategy_selector.py _REGIME_PRIORS + _NEUTRAL_PRIOR
REGIME_PRIORS = {
    "trending": {"trend_following": 0.5, "momentum": 0.3},
    "bull": {"trend_following": 0.5, "momentum": 0.3},
    "bear": {"trend_following": 0.5, "momentum": 0.3},
    "sideways": {"mean_reversion": 0.5, "grid": 0.3},
    "high_volatility": {"breakout": 0.5, "scalping": 0.3},
    "low_volatility": {"grid": 0.5, "mean_reversion": 0.3},
}
NEUTRAL_PRIOR = {"trend_following": 0.2, "mean_reversion": 0.2, "momentum": 0.2}
ALL_STRATEGIES = ["trend_following", "mean_reversion", "momentum", "breakout", "grid", "scalping"]
# src/strategy_selector.py _estimate_strategy_costs freq multipliers
FREQ_MULT = {
    "trend_following": 1.0, "mean_reversion": 1.5, "momentum": 1.3,
    "breakout": 1.2, "grid": 3.0, "scalping": 5.0,
}

BAR_HOURS = 1.0  # Alpaca 1Hour bars
HOLDOUT_DAYS = 183  # "last 6 months"


# ════════════════════════════════════════════════════════════════════════════
# Cost model
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class CostModel:
    """All costs in basis points. Configurable; echoed into every report."""
    taker_bps: float = 25.0       # Alpaca Tier-1 taker 0.25%  (docs.alpaca.markets/us/docs/crypto-fees)
    maker_bps: float = 15.0       # Alpaca Tier-1 maker 0.15%
    slippage_bps: float = 5.0     # market-order slippage, per side
    # A maker (limit) entry pays the maker fee and no slippage; the market exit
    # pays taker fee + slippage. A market entry pays taker + slippage.
    @property
    def taker_round_trip_bps(self) -> float:
        return 2.0 * (self.taker_bps + self.slippage_bps)

    def entry_cost_bps(self, entry_type: str) -> float:
        if entry_type == "limit":
            return self.maker_bps
        return self.taker_bps + self.slippage_bps

    def exit_cost_bps(self) -> float:
        return self.taker_bps + self.slippage_bps


# ════════════════════════════════════════════════════════════════════════════
# Data fetch + cache
# ════════════════════════════════════════════════════════════════════════════
def _http_json(url: str, tries: int = 4) -> dict:
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # transient network/rate-limit
            last = e
    raise RuntimeError(f"GET failed: {url}: {last}")


def _http_bytes(url: str, tries: int = 4) -> bytes:
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:
            last = e
    raise RuntimeError(f"GET failed: {url}: {last}")


def fetch_alpaca_bars(symbol: str, start="2021-01-01T00:00:00Z", end="2026-10-09T00:00:00Z") -> pd.DataFrame:
    """Paginate Alpaca 1Hour crypto bars. Returns DataFrame indexed by UTC ts."""
    rows, token = [], None
    for _ in range(2000):
        params = {"symbols": symbol, "timeframe": "1Hour", "start": start, "end": end, "limit": 10000}
        if token:
            params["page_token"] = token
        d = _http_json(ALPACA_BARS + "?" + urllib.parse.urlencode(params))
        bars = d.get("bars", {}).get(symbol, []) or []
        rows.extend(bars)
        token = d.get("next_page_token")
        if not token or not bars:
            break
    if not rows:
        raise RuntimeError(f"no bars for {symbol}")
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["t"], utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["ts", "open", "high", "low", "close", "volume"]].drop_duplicates("ts").sort_values("ts")
    return df.reset_index(drop=True)


def fetch_binance_funding(perp: str, start="2021-01", end="2026-10") -> pd.Series:
    """Concatenate Binance USD-M monthly funding archives -> Series by ts (UTC)."""
    months = pd.date_range(start=start, end=end, freq="MS")
    frames = []
    for m in months:
        ym = m.strftime("%Y-%m")
        url = BINANCE_FUNDING.format(sym=perp, ym=ym)
        try:
            raw = _http_bytes(url)
        except Exception:
            continue  # month not yet published
        z = zipfile.ZipFile(io.BytesIO(raw))
        csv = z.read(z.namelist()[0]).decode()
        f = pd.read_csv(io.StringIO(csv))
        f["ts"] = pd.to_datetime(f["calc_time"].astype("int64"), unit="ms", utc=True)
        frames.append(f[["ts", "last_funding_rate"]])
    if not frames:
        raise RuntimeError(f"no funding for {perp}")
    out = pd.concat(frames).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return out.set_index("ts")["last_funding_rate"]


def cache_path(kind: str, key: str) -> str:
    safe = key.replace("/", "_")
    return os.path.join(CACHE, f"{kind}_{safe}.parquet")


def load_or_fetch_bars(symbol: str) -> pd.DataFrame:
    p = cache_path("bars", symbol)
    if os.path.exists(p):
        return pd.read_parquet(p)
    df = fetch_alpaca_bars(symbol)
    df.to_parquet(p, index=False)
    return df


def load_or_fetch_funding(symbol: str) -> pd.Series:
    p = cache_path("funding", PERP[symbol])
    if os.path.exists(p):
        return pd.read_parquet(p)["last_funding_rate"]
    s = fetch_binance_funding(PERP[symbol])
    s.to_frame().to_parquet(p)
    return s


# ════════════════════════════════════════════════════════════════════════════
# Feature engineering (replicates src/feature_engineering.py + analyze_market_regime)
# ════════════════════════════════════════════════════════════════════════════
def _zscore(s: pd.Series, window: int = 20) -> pd.Series:
    mu = s.rolling(window, min_periods=2).mean()
    sig = s.rolling(window, min_periods=2).std().replace(0.0, np.nan)
    return ((s - mu) / sig).fillna(0.0)


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI matching src/strategies.py _calculate_rsi (SMMA recursion)."""
    diff = close.diff()
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    avg_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).clip(0.0, 100.0)


def _rolling_autocorr(s: pd.Series, window: int = 10) -> pd.Series:
    lag = s.shift(1)
    cov = s.rolling(window, min_periods=4).cov(lag)
    sx = s.rolling(window, min_periods=4).std()
    sl = lag.rolling(window, min_periods=4).std()
    ac = cov / (sx * sl).replace(0.0, np.nan)
    return ac.fillna(0.0).clip(-1.0, 1.0)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar features + regime label matching the bot's regime classifier."""
    close, high, low = df["close"], df["high"], df["low"]
    prev_close = close.shift(1)
    log_ret = np.log(close / prev_close)

    rsi = _rsi_wilder(close, 14)
    prev_rsi = rsi.shift(1).fillna(50.0)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=2).mean().fillna(0.0)

    z_return = _zscore(log_ret, 20)
    _rolling_autocorr(log_ret, 10)
    z_ret_autocorr = _rolling_autocorr(z_return, 10)   # hurst proxy (analyze_market_regime)
    hurst = ((z_ret_autocorr + 1.0) / 2.0).clip(0.0, 1.0)

    price_z = _zscore(close, 20)
    ema20 = close.ewm(span=20, adjust=False).mean()
    htf_bull = close > ema20
    atr_pct = (atr / close * 100.0).fillna(0.0)

    # ── Regime classification with hysteresis (sequential) ──────────────────
    regime = np.empty(len(df), dtype=object)
    prev = "neutral"
    hb = htf_bull.to_numpy()
    ap = atr_pct.to_numpy()
    hu = hurst.to_numpy()
    for i in range(len(df)):
        a = ap[i]
        if a > HIGH_VOLATILITY_PCT:
            r = "high_volatility"
        elif a < HIGH_VOLATILITY_PCT * 0.25:
            r = "low_volatility"
        else:
            trend_up = HURST_TREND_UP
            mean_rev = HURST_MEAN_REVERT
            if prev in ("bull", "bear", "trending"):
                mean_rev = max(mean_rev, HURST_MEAN_REVERT - 0.02)
            elif prev == "sideways":
                trend_up = min(trend_up, HURST_TREND_UP + 0.02)
            if hu[i] > trend_up:
                r = "bull" if hb[i] else "bear"
            elif hu[i] < mean_rev:
                r = "sideways"
            else:
                r = "neutral"
        regime[i] = r
        prev = r

    out = pd.DataFrame({
        "ts": df["ts"].to_numpy(),
        "rsi": rsi.to_numpy(),
        "prev_rsi": prev_rsi.to_numpy(),
        "atr_pct": atr_pct.to_numpy(),
        "price_zscore": price_z.to_numpy(),
        "htf_trend": np.where(hb, "bullish", "bearish"),
        "regime": regime,
    })
    return out


def select_best_strategy(regime: str) -> str:
    """Replicates src/strategy_selector.select_best_strategy at cold start
    (regime priors as the learner's initial weights). Cost penalty applied."""
    weights = dict.fromkeys(ALL_STRATEGIES, MIN_WEIGHT)
    for name, w in REGIME_PRIORS.get(regime, NEUTRAL_PRIOR).items():
        if name in weights:
            weights[name] = w
    # cost-aware penalty (needs atr_pct; use a neutral atr_pct so the ordering
    # is the documented one; the penalty never reorders within a regime here)
    for s in list(weights):
        penalty = min(0.5, (FREQ_MULT.get(s, 1.0) - 1.0) * 0.2)
        weights[s] *= (1.0 - penalty)
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return max(weights, key=weights.get)


def strategy_long_signal(strat: str, row: dict) -> bool:
    """Long-only entry rule, faithful to src/execution_strategies.py."""
    rsi, prsi = row["rsi"], row["prev_rsi"]
    if strat == "trend_following":
        return row["htf_trend"] == "bullish" and rsi < 65.0 and rsi > prsi
    if strat == "mean_reversion":
        if rsi < (RSI_OVERSOLD + 10.0) and rsi > prsi:
            return True
        if 40.0 <= rsi <= 60.0 and row["price_zscore"] <= -BB_ZSCORE_THRESHOLD:
            return True
        return False
    if strat == "momentum":
        return rsi > 60.0 and (rsi - prsi) > 5.0
    if strat == "breakout":
        return row["regime"] == "high_volatility" and rsi > 55.0
    if strat == "grid":
        return rsi < 40.0
    if strat == "scalping":
        return rsi > prsi and rsi < 55.0
    return False


def bot_entry_signals(feat: pd.DataFrame) -> np.ndarray:
    """B2: bot's own entry rules -> boolean array of entry-signal bars."""
    n = len(feat)
    sig = np.zeros(n, dtype=bool)
    reg = feat["regime"].to_numpy()
    rsi = feat["rsi"].to_numpy()
    prsi = feat["prev_rsi"].to_numpy()
    pz = feat["price_zscore"].to_numpy()
    hb = (feat["htf_trend"] == "bullish").to_numpy()
    # strategy chosen per bar from the regime
    chosen = np.empty(n, dtype=object)
    for i in range(n):
        chosen[i] = select_best_strategy(reg[i])
    for i in range(n):
        s = chosen[i]
        if s == "trend_following":
            sig[i] = hb[i] and rsi[i] < 65.0 and rsi[i] > prsi[i]
        elif s == "mean_reversion":
            sig[i] = ((rsi[i] < RSI_OVERSOLD + 10.0 and rsi[i] > prsi[i])
                      or (40.0 <= rsi[i] <= 60.0 and pz[i] <= -BB_ZSCORE_THRESHOLD))
        elif s == "momentum":
            sig[i] = rsi[i] > 60.0 and (rsi[i] - prsi[i]) > 5.0
        elif s == "breakout":
            sig[i] = reg[i] == "high_volatility" and rsi[i] > 55.0
        elif s == "grid":
            sig[i] = rsi[i] < 40.0
        elif s == "scalping":
            sig[i] = rsi[i] > prsi[i] and rsi[i] < 55.0
    return sig


# ════════════════════════════════════════════════════════════════════════════
# Trade simulator
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Trade:
    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    gross_pct: float
    net_pct: float
    reason: str
    bars_held: int
    filled: bool = True


def simulate(
    symbol: str,
    bars: pd.DataFrame,
    signal_idx: np.ndarray,
    cost: CostModel,
    target_pct: float = PROFIT_TARGET_PCT,
    stop_pct: float = STOP_LOSS_PCT,
    max_hold_hours: float = MAX_HOLD_HOURS,
    entry_type: str = "market",
    veto: np.ndarray | None = None,
) -> tuple[list[Trade], int]:
    """One position at a time. Entry fills at the NEXT bar's open (never the
    signal bar). On a bar touching both stop and target, the STOP is assumed
    first. Maker entries fill only if the next bar trades through the limit.

    Returns ``(trades, unfilled_signal_count)``."""
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    ts = bars["ts"].to_numpy()
    n = len(bars)
    max_bars = max(1, round(max_hold_hours / BAR_HOURS))
    trades: list[Trade] = []
    unfilled = 0
    i = 0
    signals = np.flatnonzero(signal_idx)
    pos = 0  # pointer into signals
    next_free_bar = 0  # earliest bar at which we may open a new position
    while pos < len(signals):
        i = signals[pos]
        pos += 1
        e = i + 1  # entry bar = next bar
        if e >= n - 1 or e < next_free_bar:
            continue
        if veto is not None and veto[i]:
            continue
        # ── entry fill ──
        if entry_type == "limit":
            limit = c[i]  # limit at signal bar's close
            if not (lo[e] <= limit):  # next bar never traded through -> no fill
                unfilled += 1
                continue
            entry_price = min(o[e], limit)  # gap-through fills better at open
        else:
            entry_price = o[e]
        if entry_price <= 0:
            continue
        tp = entry_price * (1.0 + target_pct)
        sl = entry_price * (1.0 - stop_pct)
        exit_price, reason, j = None, None, None
        # evaluate from the entry bar through the max-hold bar
        for j in range(e, min(e + max_bars + 1, n)):
            if j > e and (j - e) >= max_bars:
                exit_price, reason = o[j], "max_hold"
                break
            if lo[j] <= sl:            # stop checked FIRST on a same-bar touch
                exit_price, reason = sl, "stop_loss"
                break
            if h[j] >= tp:
                exit_price, reason = tp, "profit_target"
                break
        if exit_price is None:        # ran off the end of data
            exit_price, reason, j = c[n - 1], "eod", n - 1
        gross = (exit_price / entry_price - 1.0) * 100.0
        cost_bps = cost.entry_cost_bps(entry_type) + cost.exit_cost_bps()
        net = gross - cost_bps / 100.0
        trades.append(Trade(symbol, pd.Timestamp(ts[e]), pd.Timestamp(ts[j]),
                            float(entry_price), float(exit_price),
                            float(gross), float(net), reason, int(j - e)))
        next_free_bar = j
    return trades, unfilled


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════
def bootstrap_ci(net: np.ndarray, n_boot: int = 2000, seed: int = 12345) -> tuple[float, float]:
    if len(net) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(net), size=(n_boot, len(net)))
    means = net[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def max_drawdown(trades: list[Trade]) -> float:
    """Max drawdown (%) of the compounded equity curve, trades in exit order."""
    if not trades:
        return 0.0
    ordered = sorted(trades, key=lambda t: t.exit_ts)
    eq = np.cumprod([1.0 + t.net_pct / 100.0 for t in ordered])
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1.0) * 100.0
    return float(dd.min())


def summarize(trades: list[Trade], label: str = "") -> dict:
    net = np.array([t.net_pct for t in trades], dtype=float)
    if len(net) == 0:
        return {"label": label, "trades": 0, "win_rate": None, "avg_net_pct": None,
                "profit_factor": None, "max_dd_pct": None, "ci_lo": None, "ci_hi": None}
    wins = net[net > 0]
    losses = net[net <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    lo, hi = bootstrap_ci(net)
    return {
        "label": label,
        "trades": len(net),
        "win_rate": float((net > 0).mean()),
        "avg_net_pct": float(net.mean()),
        "profit_factor": pf,
        "max_dd_pct": max_drawdown(trades),
        "ci_lo": lo,
        "ci_hi": hi,
        "avg_gross_pct": float(np.mean([t.gross_pct for t in trades])),
    }


def equal_subperiods(trades: list[Trade], start, end, k: int = 4) -> list[dict]:
    """Split [start, end] into k equal time windows; net %/trade in each."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    edges = [start + (end - start) * i / k for i in range(k + 1)]
    rows = []
    for i in range(k):
        seg = [t for t in trades if edges[i] <= t.entry_ts < edges[i + 1]]
        net = np.array([t.net_pct for t in seg], dtype=float)
        rows.append({"window": f"{edges[i].date()}..{edges[i+1].date()}",
                     "trades": len(net),
                     "avg_net_pct": float(net.mean()) if len(net) else None})
    return rows


def count_configs() -> int:
    """Total distinct configurations evaluated (for the multiple-testing note)."""
    return (len(SYMBOLS)                     # 3 markets
            + 200                            # random seeds (B1)
            + 1                              # B2 bot rules
            + 1                              # A market
            + 1                              # A limit
            + len(PREREGISTERED_B_COMBOS)    # 20 exit geometries
            + 1)                             # C funding veto


def period_breakdown(trades: list[Trade], freq: str = "Y") -> list[dict]:
    if not trades:
        return []
    df = pd.DataFrame([{"ts": t.entry_ts, "net": t.net_pct} for t in trades])
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    df["period"] = df["ts"].dt.to_period("Y" if freq == "Y" else "Q").astype(str)
    rows = []
    for p, g in df.groupby("period"):
        rows.append({"period": p, "trades": len(g),
                     "win_rate": float((g["net"] > 0).mean()),
                     "avg_net_pct": float(g["net"].mean())})
    return rows


# ════════════════════════════════════════════════════════════════════════════
# Split
# ════════════════════════════════════════════════════════════════════════════
def split_dev_holdout(bars_by_sym: dict[str, pd.DataFrame]):
    starts = [b["ts"].min() for b in bars_by_sym.values()]
    ends = [b["ts"].max() for b in bars_by_sym.values()]
    common_start = max(starts)
    common_end = min(ends)
    cutoff = common_end - timedelta(days=HOLDOUT_DAYS)
    return common_start, common_end, cutoff


def slice_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    m = (df["ts"] >= start) & (df["ts"] <= end)
    return df.loc[m].reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# Baselines
# ════════════════════════════════════════════════════════════════════════════
def buy_and_hold(bars_by_sym: dict[str, pd.DataFrame], start, end) -> dict:
    rets = []
    for _sym, df in bars_by_sym.items():
        w = slice_window(df, start, end)
        if len(w) < 2:
            continue
        rets.append(w["close"].iloc[-1] / w["open"].iloc[0] - 1.0)
    return {"avg_net_pct": float(np.mean(rets) * 100.0), "per_symbol_pct": [float(r * 100) for r in rets]}


def random_baseline(bars_by_sym: dict[str, pd.DataFrame], n_per_sym: dict[str, int],
                    cost: CostModel, target_pct, stop_pct, max_hold_hours,
                    entry_type="market", seeds: int = 200) -> dict:
    """Random long entries, same exits/costs, same per-symbol trade COUNT."""
    means = []
    winrates = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        all_tr = []
        for sym, df in bars_by_sym.items():
            n = len(df)
            k = int(n_per_sym.get(sym, 0))
            if k <= 0 or n < 3:
                continue
            idx = rng.choice(np.arange(0, n - 2), size=min(k, n - 2), replace=False)
            sig = np.zeros(n, dtype=bool)
            sig[idx] = True
            all_tr.extend(simulate(sym, df, sig, cost, target_pct, stop_pct,
                                   max_hold_hours, entry_type)[0])
        if all_tr:
            net = np.array([t.net_pct for t in all_tr])
            means.append(float(net.mean()))
            winrates.append(float((net > 0).mean()))
    m = np.array(means)
    return {
        "seeds": seeds,
        "mean_avg_net_pct": float(m.mean()) if len(m) else None,
        "p5": float(np.percentile(m, 5)) if len(m) else None,
        "p50": float(np.percentile(m, 50)) if len(m) else None,
        "p95": float(np.percentile(m, 95)) if len(m) else None,
        "mean_win_rate": float(np.mean(winrates)) if winrates else None,
    }


# ════════════════════════════════════════════════════════════════════════════
# Orchestration
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class DataBundle:
    bars: dict[str, pd.DataFrame]
    feat: dict[str, pd.DataFrame]
    funding_z: dict[str, pd.Series]


def load_bundle() -> DataBundle:
    bars, feat, fz = {}, {}, {}
    for sym in SYMBOLS:
        b = load_or_fetch_bars(sym)
        bars[sym] = b
        feat[sym] = compute_features(b)
        fr = load_or_fetch_funding(sym)
        # 30-day (90 x 8h) rolling z-score of funding, mapped onto hourly bars
        mu = fr.rolling(90, min_periods=20).mean()
        sd = fr.rolling(90, min_periods=20).std().replace(0.0, np.nan)
        z = ((fr - mu) / sd).dropna()
        fz[sym] = z
    return DataBundle(bars, feat, fz)


def funding_veto(sym: str, bars: pd.DataFrame, bundle: DataBundle, z_thresh: float = 2.0) -> np.ndarray:
    """True where a NEW entry is vetoed (BTC market-wide funding z > thresh)."""
    btc_z = bundle.funding_z["BTC/USD"]
    if len(btc_z) == 0:
        return np.zeros(len(bars), dtype=bool)
    left = bars[["ts"]].sort_values("ts").copy()
    left["ts"] = pd.to_datetime(left["ts"], utc=True).astype("datetime64[ns, UTC]")
    zdf = btc_z.rename("z").reset_index()
    zdf["ts"] = pd.to_datetime(zdf["ts"], utc=True).astype("datetime64[ns, UTC]")
    merged = pd.merge_asof(left, zdf.sort_values("ts"), on="ts", direction="backward")
    return (merged["z"].fillna(0.0).to_numpy() > z_thresh)


def dev_windows(bundle: DataBundle):
    cs, ce, cutoff = split_dev_holdout(bundle.bars)
    return cs, cutoff, ce


def run_family_b_exits(bundle: DataBundle, start, end, cost: CostModel) -> list[dict]:
    """Grid over target x stop x max-hold; dev only. Uses B2 signals."""
    combos = PREREGISTERED_B_COMBOS
    results = []
    for (tgt, stp, mh) in combos:
        all_tr = []
        for sym in SYMBOLS:
            df = slice_window(bundle.bars[sym], start, end)
            f = slice_window(bundle.feat[sym], start, end)
            sig = bot_entry_signals(f)
            all_tr.extend(simulate(sym, df, sig, cost, tgt / 100.0, stp / 100.0, mh)[0])
        s = summarize(all_tr, f"B tgt={tgt}% stop={stp}% hold={mh}h")
        s["params"] = {"target_pct": tgt, "stop_pct": stp, "max_hold_hours": mh}
        results.append(s)
    return results


# Pre-registered 20 combos (fixed in RESULTS.md §4.2; span the grid, no cherry-pick)
PREREGISTERED_B_COMBOS = [
    (1.5, 2, 4), (1.5, 3, 8), (1.5, 4, 24), (2, 3, 8), (2, 4, 24),
    (2, 5, 72), (3, 2, 8), (3, 3, 8), (3, 4, 8), (3, 4, 24),
    (3, 5, 72), (4, 3, 8), (4, 4, 8), (4, 5, 24), (4, 8, 72),
    (6, 4, 24), (6, 5, 72), (6, 8, 72), (2, 2, 4), (4, 4, 72),
]


def run_candidates_dev(bundle: DataBundle, cost: CostModel) -> dict:
    cs, cutoff, ce = dev_windows(bundle)
    out: dict = {"window": {"start": str(cs), "dev_end": str(cutoff), "holdout_end": str(ce)},
                 "cost_model": asdict(cost)}
    # ---- B2 reference: bot entry rules + bot exits ----
    b2_tr = []
    for sym in SYMBOLS:
        df = slice_window(bundle.bars[sym], cs, cutoff)
        f = slice_window(bundle.feat[sym], cs, cutoff)
        b2_tr.extend(simulate(sym, df, bot_entry_signals(f), cost)[0])
    out["B2_bot_rules"] = summarize(b2_tr, "B2 bot entry rules + bot exits")
    out["B2_bot_rules"]["per_year"] = period_breakdown(b2_tr, "Y")
    out["B2_bot_rules"]["per_quarter"] = period_breakdown(b2_tr, "Q")
    n_per_sym = {}
    for sym in SYMBOLS:
        n_per_sym[sym] = sum(1 for t in b2_tr if t.symbol == sym)

    # ---- B0 buy & hold ----
    out["B0_buy_hold"] = buy_and_hold(bundle.bars, cs, cutoff)

    # ---- B1 random baseline (bot exits, matched count) ----
    out["B1_random_bot_exits"] = random_baseline(
        {s: slice_window(bundle.bars[s], cs, cutoff) for s in SYMBOLS},
        n_per_sym, cost, PROFIT_TARGET_PCT, STOP_LOSS_PCT, MAX_HOLD_HOURS, seeds=200)

    # ---- Candidate A: maker (limit) entries vs market, bot exits ----
    a = {"market": [], "limit": []}
    unfilled_total = 0
    filled_fwd, unfilled_fwd = [], []
    for sym in SYMBOLS:
        df = slice_window(bundle.bars[sym], cs, cutoff)
        f = slice_window(bundle.feat[sym], cs, cutoff)
        sig = bot_entry_signals(f)
        a["market"].extend(simulate(sym, df, sig, cost, entry_type="market")[0])
        tr, nf = simulate(sym, df, sig, cost, entry_type="limit")
        a["limit"].extend(tr)
        unfilled_total += nf
        # Adverse selection, measured at the SIGNAL level: for every signal,
        # would the resting limit at the signal close have filled on the next
        # bar, and what was the next bar's forward return? A limit that only
        # fills when price falls through it is adversely selected.
        o = df["open"].to_numpy()
        c = df["close"].to_numpy()
        lo = df["low"].to_numpy()
        for i in np.flatnonzero(sig):
            if i + 1 >= len(df):
                continue
            fwd = (o[i + 1] / c[i] - 1.0) * 100.0  # signal-close -> next open
            if lo[i + 1] <= c[i]:
                filled_fwd.append(fwd)
            else:
                unfilled_fwd.append(fwd)
    out["A_market"] = summarize(a["market"], "A market entries (bot exits)")
    out["A_limit"] = summarize(a["limit"], "A maker/limit entries (bot exits)")
    out["A_unfilled_signals"] = unfilled_total
    out["A_adverse_selection"] = {
        "filled_fwd_mean_pct": float(np.mean(filled_fwd)) if filled_fwd else None,
        "unfilled_fwd_mean_pct": float(np.mean(unfilled_fwd)) if unfilled_fwd else None,
        "n_filled": len(filled_fwd), "n_unfilled": len(unfilled_fwd),
    }

    # ---- Candidate B: exit geometry grid (ALL 20) ----
    out["B_exit_grid"] = run_family_b_exits(bundle, cs, cutoff, cost)

    # ---- Candidate C: funding veto (ONE variant) ----
    c_tr = []
    for sym in SYMBOLS:
        df = slice_window(bundle.bars[sym], cs, cutoff)
        f = slice_window(bundle.feat[sym], cs, cutoff)
        veto = funding_veto(sym, df, bundle, z_thresh=2.0)
        c_tr.extend(simulate(sym, df, bot_entry_signals(f), cost, veto=veto)[0])
    out["C_funding_veto"] = summarize(c_tr, "C funding z>2 veto (bot exits)")
    out["C_funding_veto"]["per_quarter"] = period_breakdown(c_tr, "Q")

    # ---- freeze best per family (dev) ----
    out["freeze"] = freeze_best(out)
    out["configs_tried"] = count_configs()
    return out


def freeze_best(out: dict) -> dict:
    """Pick the single best dev candidate per family by avg net %/trade."""
    def best_of(items, key="avg_net_pct"):
        cand = [x for x in items if x.get(key) is not None]
        return max(cand, key=lambda x: x[key]) if cand else None
    fam_b = best_of(out["B_exit_grid"])
    fam_a = best_of([out["A_market"], out["A_limit"]])
    fam_c = out["C_funding_veto"]
    return {
        "A": {"which": fam_a["label"], "avg_net_pct": fam_a["avg_net_pct"]} if fam_a else None,
        "B": {"params": fam_b["params"], "avg_net_pct": fam_b["avg_net_pct"]} if fam_b else None,
        "C": {"avg_net_pct": fam_c["avg_net_pct"]},
        "B2_reference": {"avg_net_pct": out["B2_bot_rules"]["avg_net_pct"]},
    }


def run_holdout(bundle: DataBundle, cost: CostModel, frozen: dict) -> dict:
    _cs, cutoff, ce = dev_windows(bundle)
    res: dict = {"window": {"holdout_start": str(cutoff), "holdout_end": str(ce)},
                 "cost_model": asdict(cost), "runs": {}}
    # random baseline on holdout, bot exits, matched to B2 count
    b2_tr = []
    for sym in SYMBOLS:
        df = slice_window(bundle.bars[sym], cutoff, ce)
        f = slice_window(bundle.feat[sym], cutoff, ce)
        b2_tr.extend(simulate(sym, df, bot_entry_signals(f), cost)[0])
    res["runs"]["B2_reference"] = summarize(b2_tr, "B2 reference (holdout)")
    res["runs"]["B2_reference"]["per_quarter"] = period_breakdown(b2_tr, "Q")
    n_per_sym = {s: sum(1 for t in b2_tr if t.symbol == s) for s in SYMBOLS}
    res["random"] = random_baseline(
        {s: slice_window(bundle.bars[s], cutoff, ce) for s in SYMBOLS},
        n_per_sym, cost, PROFIT_TARGET_PCT, STOP_LOSS_PCT, MAX_HOLD_HOURS, seeds=200)

    # Family B frozen params
    bp = frozen["B"]["params"]
    bt = []
    for sym in SYMBOLS:
        df = slice_window(bundle.bars[sym], cutoff, ce)
        f = slice_window(bundle.feat[sym], cutoff, ce)
        bt.extend(simulate(sym, df, bot_entry_signals(f), cost,
                           bp["target_pct"] / 100.0, bp["stop_pct"] / 100.0, bp["max_hold_hours"])[0])
    res["runs"]["B_frozen"] = summarize(bt, "B frozen (holdout)")
    res["runs"]["B_frozen"]["params"] = bp
    res["runs"]["B_frozen"]["per_quarter"] = period_breakdown(bt, "Q")

    # Family A frozen
    a_type = "limit" if frozen["A"]["which"].startswith("A maker") else "market"
    at = []
    for sym in SYMBOLS:
        df = slice_window(bundle.bars[sym], cutoff, ce)
        f = slice_window(bundle.feat[sym], cutoff, ce)
        at.extend(simulate(sym, df, bot_entry_signals(f), cost, entry_type=a_type)[0])
    res["runs"]["A_frozen"] = summarize(at, f"A frozen ({a_type}) (holdout)")
    res["runs"]["A_frozen"]["per_quarter"] = period_breakdown(at, "Q")

    # Family C frozen
    ct = []
    for sym in SYMBOLS:
        df = slice_window(bundle.bars[sym], cutoff, ce)
        f = slice_window(bundle.feat[sym], cutoff, ce)
        ct.extend(simulate(sym, df, bot_entry_signals(f), cost,
                           veto=funding_veto(sym, df, bundle, 2.0))[0])
    res["runs"]["C_frozen"] = summarize(ct, "C frozen funding veto (holdout)")
    res["runs"]["C_frozen"]["per_quarter"] = period_breakdown(ct, "Q")

    # 4 equal holdout sub-periods per frozen run (for the ">= 3 of 4" rule)
    trades_by_run = {"B2_reference": b2_tr, "B_frozen": bt, "A_frozen": at, "C_frozen": ct}
    for name, tr in trades_by_run.items():
        res["runs"][name]["sub_periods"] = equal_subperiods(tr, cutoff, ce, 4)

    res["configs_tried"] = count_configs()
    res["verdict"] = evaluate_pass(res)
    return res


def evaluate_pass(res: dict) -> dict:
    rnd_p95 = res["random"]["p95"]
    verdict = {}
    for fam in ["B_frozen", "A_frozen", "C_frozen", "B2_reference"]:
        s = res["runs"][fam]
        sub = s.get("sub_periods", [])
        pos_q = sum(1 for x in sub if (x["avg_net_pct"] or 0) > 0)
        ci_pos = (s["ci_lo"] is not None and s["ci_lo"] > 0)
        beats = (s["avg_net_pct"] is not None and rnd_p95 is not None and s["avg_net_pct"] > rnd_p95)
        verdict[fam] = {
            "avg_net_pct": s["avg_net_pct"], "ci": [s["ci_lo"], s["ci_hi"]],
            "random_p95": rnd_p95,
            "ci_excludes_zero": bool(ci_pos),
            "beats_random_p95": bool(beats),
            "positive_subperiods": f"{pos_q}/4",
            "PASS": bool(ci_pos and beats and pos_q >= 3),
        }
    return verdict


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
def cmd_fetch():
    os.makedirs(CACHE, exist_ok=True)
    for sym in SYMBOLS:
        b = fetch_alpaca_bars(sym)
        b.to_parquet(cache_path("bars", sym), index=False)
        f = fetch_binance_funding(PERP[sym])
        f.to_frame().to_parquet(cache_path("funding", PERP[sym]))


def cmd_dev():
    bundle = load_bundle()
    out = run_candidates_dev(bundle, CostModel())
    with open(DEV_RESULTS, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    for _r in out["B_exit_grid"]:
        pass


def cmd_holdout():
    bundle = load_bundle()
    with open(DEV_RESULTS) as fh:
        dev = json.load(fh)
    res = run_holdout(bundle, CostModel(), dev["freeze"])
    with open(HOLDOUT_RESULTS, "w") as fh:
        json.dump(res, fh, indent=2, default=str)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "dev", "holdout"])
    args = ap.parse_args(argv)
    {"fetch": cmd_fetch, "dev": cmd_dev, "holdout": cmd_holdout}[args.cmd]()


if __name__ == "__main__":
    main()
