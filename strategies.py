# strategies.py
import numpy as np
import pandas as pd
import config

# Per-symbol Kalman filter cache so the smoothing is truly online: the filter
# state (x, p) is preserved across calls instead of being rebuilt from scratch
# over the entire series every cycle. Keyed by symbol.
_kalman_filters = {}

def get_kalman_filter(symbol: str) -> "KalmanFilter1D":
    """Returns the persisted Kalman filter for a symbol, creating it on first use."""
    f = _kalman_filters.get(symbol)
    if f is None:
        f = KalmanFilter1D()
        _kalman_filters[symbol] = f
    return f

def reset_kalman_filter(symbol: str = None):
    """Drops cached filter state. Call on startup sync or symbol list change."""
    if symbol is None:
        _kalman_filters.clear()
    else:
        _kalman_filters.pop(symbol, None)


class KalmanFilter1D:
    """
    A simple 1-dimensional Kalman Filter for real-time price smoothing and noise reduction.
    """
    def __init__(self, process_noise=1e-5, measurement_noise=1e-3, estimate_error=1.0):
        self.q = process_noise        # Process noise covariance (how fast the true price changes)
        self.r = measurement_noise    # Measurement noise covariance (volatility/noise of observations)
        self.p = estimate_error       # Estimation error covariance
        self.x = None                 # Current state estimate (smoothed price)

    def update(self, measurement: float) -> float:
        if self.x is None:
            self.x = measurement
            return self.x
        
        # 1. Prediction Update
        self.p = self.p + self.q
        
        # 2. Measurement Update (Correction)
        k_gain = self.p / (self.p + self.r)
        self.x = self.x + k_gain * (measurement - self.x)
        self.p = (1 - k_gain) * self.p
        
        return self.x


def calculate_hurst_exponent(prices: pd.Series, max_lag=20) -> float:
    """
    Estimates the Hurst exponent (H) of a price series using variance-lag scaling.
    H < 0.5: Mean-reverting (Anti-persistent)
    H > 0.5: Trending (Persistent)
    H ~ 0.5: Random walk / Brownian Motion
    """
    try:
        # Require a minimum lookback length to run regression
        vals = prices.values
        if len(vals) < max_lag * 2:
            return 0.5  # Neutral default
        
        lags = range(2, max_lag)
        variances = []
        for lag in lags:
            diffs = vals[lag:] - vals[:-lag]
            variances.append(np.var(diffs))
            
        # Filter out zeros to prevent log issues
        valid_indices = [i for i, v in enumerate(variances) if v > 0]
        if len(valid_indices) < 3:
            return 0.5
            
        lags_filtered = [lags[i] for i in valid_indices]
        var_filtered = [variances[i] for i in valid_indices]
        
        # Linear regression: Log(Variance) vs Log(Lag)
        # Var(t) ~ t^(2H)  => Log(Var) ~ 2H * Log(t)
        poly = np.polyfit(np.log(lags_filtered), np.log(var_filtered), 1)
        hurst = poly[0] / 2.0
        
        # Clip to valid mathematical range
        return float(np.clip(hurst, 0.0, 1.0))
    except Exception:
        return 0.5


def calculate_rsi(prices: pd.Series, period=14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # Handle division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calculate_atr(df: pd.DataFrame, period=14) -> pd.Series:
    """Calculates Average True Range (ATR)."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()
    return atr.fillna(0.0)


def calculate_macd(prices: pd.Series, fast=12, slow=26, signal=9) -> tuple:
    """Calculates MACD Line and Signal Line."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def analyze_market_regime(df: pd.DataFrame) -> dict:
    """
    Analyzes the market regime based on technical metrics and mathematical models.
    Returns a dict with key indicators and classified regime.
    """
    if len(df) < 50:
        return {"regime": "NEUTRAL", "hurst": 0.5, "rsi": 50.0, "volatility_pct": 0.0}

    close_series = df['close']
    current_price = close_series.iloc[-1]
    
    # 1. Volatility (ATR %)
    atr = calculate_atr(df).iloc[-1]
    volatility_pct = (atr / current_price) * 100 if current_price > 0 else 0.0

    # 2. Hurst Exponent (Trendiness)
    hurst = calculate_hurst_exponent(close_series, max_lag=20)

    # 3. Kalman Filter smoothing (online, state persisted per symbol)
    symbol = df.attrs.get("symbol") if hasattr(df, "attrs") else None
    kf = get_kalman_filter(symbol) if symbol else KalmanFilter1D()
    smoothed_price = kf.x if (kf.x is not None) else close_series.iloc[-1]
    for price in close_series:
        smoothed_price = kf.update(price)

    # 4. RSI
    rsi = calculate_rsi(close_series).iloc[-1]

    # 5. MACD Trend Strength
    macd, macd_sig = calculate_macd(close_series)
    macd_val = macd.iloc[-1]
    macd_sig_val = macd_sig.iloc[-1]

    # Determine regime classification (thresholds from config.py)
    # High volatility stand-aside check (protect capital)
    if volatility_pct > config.HIGH_VOLATILITY_PCT:
        regime = "HIGH_VOLATILITY_STAND_SIDE"
    # TrendFollowing Regimes
    elif hurst > config.HURST_TREND_UP:
        if current_price > smoothed_price and macd_val > macd_sig_val:
            regime = "TREND_BULLISH"
        elif current_price < smoothed_price and macd_val < macd_sig_val:
            regime = "TREND_BEARISH"
        else:
            regime = "TREND_NEUTRAL"
    # MeanReverting Regimes
    elif hurst < config.HURST_MEAN_REVERT:
        regime = "MEAN_REVERSION"
    else:
        regime = "NEUTRAL"
        
    return {
        "regime": regime,
        "hurst": round(hurst, 4),
        "rsi": round(rsi, 2),
        "volatility_pct": round(volatility_pct, 4),
        "smoothed_price": round(smoothed_price, 4),
        "macd_diff": round(macd_val - macd_sig_val, 4)
    }


def generate_trading_signal(df: pd.DataFrame, regime_info: dict) -> str:
    """
    Generates a trading signal (BUY, SELL, or HOLD) based on the classified regime.
    """
    regime = regime_info["regime"]
    rsi = regime_info["rsi"]
    macd_diff = regime_info["macd_diff"]

    # Standard Rule Sets based on regime (RSI thresholds from config.py)
    if regime == "TREND_BULLISH":
        # Trend following buy: ride the momentum
        if rsi < config.RSI_OVERBOUGHT and macd_diff > 0:
            return "BUY"
    elif regime == "TREND_BEARISH":
        # Trend following sell: shorting/exiting
        if rsi > config.RSI_OVERSOLD and macd_diff < 0:
            return "SELL"

    elif regime == "MEAN_REVERSION":
        # Mean reversion buy: oversold
        if rsi < config.RSI_OVERSOLD:
            return "BUY"
        # Mean reversion sell: overbought
        elif rsi > config.RSI_OVERBOUGHT:
            return "SELL"

    elif regime == "NEUTRAL":
        # Standard filter logic for ranging markets
        if rsi < config.RSI_NEUTRAL_BUY:
            return "BUY"
        elif rsi > config.RSI_NEUTRAL_SELL:
            return "SELL"

    # Default state is hold/stand-aside
    return "HOLD"
