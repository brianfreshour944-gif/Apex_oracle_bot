"""Modern trading strategies using Polars for data analysis."""

import polars as pl
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.logging_config import get_logger
from src.exchange import AlpacaExchange

logger = get_logger(__name__)

import time

class TradingStrategy:
    """Modern trading strategy with regime classification and signal generation."""

    def __init__(self, exchange: AlpacaExchange, cache_ttl: float = 60.0, backtest: bool = False):
        self.exchange = exchange
        self.last_analysis_time = None
        self._regime_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._cache_ttl = cache_ttl
        self.backtest = backtest
        # Tracks which strategy opened the currently-open position per symbol.
        # Without this, select_best_strategy() is called fresh every cycle and
        # can hand an open position to a DIFFERENT strategy's exit logic than
        # the one that opened it -- e.g. Grid opens a position, then Scalping
        # becomes "best" next cycle and closes it under its own unrelated
        # 1%-target rule. That produces rapid, incoherent open/close churn
        # (a new position roughly every LOOP_INTERVAL_SEC) instead of any one
        # strategy's logic actually playing out.
        self._active_strategy: Dict[str, str] = {}

    async def analyze_market_regime(self, symbol: str, timeframe: str = "1D", limit: int = 100) -> Dict[str, Any]:
        """Analyze market regime using Hurst exponent, ATR, and RSI with TTL caching."""
        now = time.monotonic()
        if symbol in self._regime_cache:
            cached_time, cached_res = self._regime_cache[symbol]
            if now - cached_time < self._cache_ttl:
                return cached_res

        try:
            # Get historical data
            bars_df = await self.exchange.get_bars(symbol, timeframe, limit)

            if len(bars_df) < 20:
                logger.warning(f"Insufficient data for {symbol}, only {len(bars_df)} bars")
                res = {
                    "regime": "neutral",
                    "hurst": 0.5,
                    "atr": 0.0,
                    "rsi": 50.0,
                    "prev_rsi": 50.0,
                    "confidence": 0.0
                }
                self._regime_cache[symbol] = (now, res)
                return res


            # Extract numpy arrays explicitly (Polars Series -> numpy)
            close_arr = bars_df["close"].to_numpy()
            high_arr = bars_df["high"].to_numpy()
            low_arr = bars_df["low"].to_numpy()

            # Calculate returns for Hurst exponent
            returns = np.diff(close_arr) / close_arr[:-1]
            returns = returns[~np.isnan(returns)]

            # Calculate Hurst exponent
            hurst = self._calculate_hurst(returns)

            # Calculate ATR (pass numpy arrays)
            atr = self._calculate_atr(high_arr, low_arr, close_arr)

            # Calculate RSI
            rsi, prev_rsi = self._calculate_rsi(close_arr)

            # Multi-timeframe confirmation check (e.g. EMA slope on bars)
            htf_trend = "neutral"
            if len(close_arr) >= 20:
                # Use pandas for robust EWM to avoid Polars version API drift
                ema20 = float(bars_df["close"].to_pandas().ewm(span=20).mean().iloc[-1])
                htf_trend = "bullish" if close_arr[-1] > ema20 else "bearish"

            # Classify regime
            atr_pct = (atr / close_arr[-1]) * 100 if close_arr[-1] > 0 else 0
            if atr_pct > settings.HIGH_VOLATILITY_PCT:
                regime = "high_volatility"
            elif atr_pct < settings.HIGH_VOLATILITY_PCT * 0.25:
                regime = "low_volatility"
            elif hurst > settings.HURST_TREND_UP:
                if htf_trend == "bullish":
                    regime = "bull"
                elif htf_trend == "bearish":
                    regime = "bear"
                else:
                    regime = "trending"
            elif hurst < settings.HURST_MEAN_REVERT:
                regime = "sideways"
            else:
                regime = "neutral"

            volatility = float(np.std(returns) * 100) if len(returns) > 0 else 0.0
            logger.info(
                f"Regime: {regime.upper()} "
                f"Hurst={hurst:.3f} "
                f"ATR={atr_pct:.2f}% "
                f"Volatility={volatility:.2f}%"
            )

            # Fetch On-Chain Derivatives Data
            if not self.backtest:
                try:
                    from src.onchain_data import fetch_derivatives_data
                    import asyncio
                    
                    deriv_data = await fetch_derivatives_data(symbol)
                    funding_rate = float(deriv_data.get("funding_rate", 0.0))
                    open_interest = float(deriv_data.get("open_interest", 0.0))
                    long_short_ratio = float(deriv_data.get("long_short_ratio", 1.0))
                    bid_ask_imbalance = float(deriv_data.get("bid_ask_imbalance", 0.0))
                except Exception as e:
                    logger.warning(f"Failed to fetch derivatives data for {symbol}: {e}")
                    funding_rate = 0.0
                    open_interest = 0.0
                    long_short_ratio = 1.0
                    bid_ask_imbalance = 0.0
            else:
                funding_rate = 0.0
                open_interest = 0.0
                long_short_ratio = 1.0
                bid_ask_imbalance = 0.0
                
            # Fetch News Sentiment Alternative Data
            if not self.backtest:
                try:
                    from src.sentiment_analyzer import extract_sentiment
                    sentiment_data = await extract_sentiment(symbol)
                    sentiment_score = float(sentiment_data.get("sentiment_score", 0.0))
                    event_type = str(sentiment_data.get("event_type", "none"))
                    sentiment_conf = float(sentiment_data.get("confidence", 0.0))
                except Exception as e:
                    logger.warning(f"Failed to fetch sentiment data for {symbol}: {e}")
                    sentiment_score = 0.0
                    event_type = "none"
                    sentiment_conf = 0.0
            else:
                sentiment_score = 0.0
                event_type = "none"
                sentiment_conf = 0.0

            res = {
                "regime": regime,
                "hurst": float(hurst),
                "atr": float(atr),
                "rsi": float(rsi),
                "prev_rsi": float(prev_rsi),
                "htf_trend": htf_trend,
                "confidence": self._calculate_regime_confidence(regime, hurst, atr, rsi),
                "funding_rate": funding_rate,
                "open_interest": open_interest,
                "long_short_ratio": long_short_ratio,
                "bid_ask_imbalance": bid_ask_imbalance,
                "sentiment_score": sentiment_score,
                "event_type": event_type,
                "sentiment_conf": sentiment_conf,
            }
            self._regime_cache[symbol] = (now, res)
            return res



        except Exception as e:
            logger.error(f"Regime analysis failed: {e}")
            return {
                "regime": "neutral",
                "hurst": 0.5,
                "atr": 0.0,
                "rsi": 50.0,
                "prev_rsi": 50.0,
                "confidence": 0.0
            }

    def _calculate_hurst(self, returns: np.ndarray) -> float:
        """Calculate Hurst exponent using rescaled range (R/S) analysis with bias correction."""
        if len(returns) < 20:
            return 0.5

        # Extended lag range for multi-scale memory detection
        max_lag = min(50, len(returns) // 2)
        lags = list(range(2, max_lag))
        tau = []
        valid_lags = []

        for lag in lags:
            n_windows = len(returns) // lag
            if n_windows < 1:
                continue

            rs_values = []
            for i in range(n_windows):
                window = returns[i * lag: (i + 1) * lag]
                if len(window) < 2:
                    continue

                mean_adj = window - np.mean(window)
                cum_dev = np.cumsum(mean_adj)
                r = np.max(cum_dev) - np.min(cum_dev)
                s = np.std(window, ddof=1) if len(window) > 1 else np.std(window)

                if s > 1e-8:
                    rs_values.append(r / s)

            if rs_values:
                # Anis-Lloyd theoretical expectation correction factor for small n
                # E[R/S] ~ gamma(0.5 * (lag - 1)) / (sqrt(pi) * gamma(0.5 * lag)) * sum(sqrt((lag - i) / i))
                n_float = float(lag)
                expected_rs = (n_float - 0.5) / n_float * np.sqrt(np.pi * n_float / 2.0) if n_float > 2 else 1.0
                adjusted_rs = np.mean(rs_values) / expected_rs * np.sqrt(n_float ** 0.5)
                tau.append(adjusted_rs)
                valid_lags.append(lag)

        if len(tau) < 3:
            return 0.5

        try:
            poly = np.polyfit(np.log(valid_lags), np.log(tau), 1)
            hurst = poly[0]
            return float(np.clip(hurst, 0.0, 1.0))
        except Exception:
            return 0.5


    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Calculate Average True Range from numpy arrays."""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ])
        # Guard against a single bad/NaN bar corrupting the entire ATR value.
        # np.mean propagates one NaN through the whole result, which would
        # then silently disable downstream `atr > threshold` veto checks
        # (NaN comparisons are always False in Python/numpy) exactly when
        # the input data is worst.
        atr = np.nanmean(tr) if np.any(np.isfinite(tr)) else 0.0
        return float(np.nan_to_num(atr, nan=0.0, posinf=0.0, neginf=0.0))

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> Tuple[float, float]:
        """Calculate Relative Strength Index. Returns (current_rsi, prev_rsi)."""
        if len(prices) < period + 1:
            return 50.0, 50.0

        diffs = np.diff(prices)
        # Calculate initial SMMA
        up = np.mean(np.where(diffs[:period] > 0, diffs[:period], 0))
        down = np.mean(np.where(diffs[:period] < 0, -diffs[:period], 0))
        
        if down == 0:
            rsi = [100.0]
        else:
            rs = up / down
            rsi = [100.0 - (100.0 / (1.0 + rs))]
        
        for i in range(period, len(prices)-1):
            delta = diffs[i]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
                
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            if down == 0:
                rsi.append(100.0)
            else:
                rs = up / down
                rsi.append(100.0 - (100.0 / (1.0 + rs)))

        if len(rsi) >= 2:
            return float(rsi[-1]), float(rsi[-2])
        return float(rsi[-1]), 50.0

    def _calculate_regime_confidence(self, regime: str, hurst: float, atr: float, rsi: float) -> float:
        """Calculate confidence in regime classification."""
        if regime in ["bull", "bear", "trending"]:
            return min(1.0, (hurst - settings.HURST_TREND_UP) * 2.0)
        elif regime == "sideways":
            return min(1.0, (settings.HURST_MEAN_REVERT - hurst) * 2.0)
        elif regime == "high_volatility":
            return min(1.0, (atr - settings.HIGH_VOLATILITY_PCT) / settings.HIGH_VOLATILITY_PCT)
        elif regime == "low_volatility":
            return min(1.0, (settings.HIGH_VOLATILITY_PCT * 0.25 - atr) / (settings.HIGH_VOLATILITY_PCT * 0.25))
        else:
            return 1.0 - abs(0.5 - hurst) * 2.0

    async def generate_trading_signal(self, symbol: str, current_price: float, position: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate trading signal based on current regime and market conditions."""
        try:
            # Get current market regime
            regime_data = await self.analyze_market_regime(symbol)
            regime = regime_data["regime"]
            rsi = regime_data["rsi"]
            prev_rsi = regime_data.get("prev_rsi", rsi)

            # AI Strategy Selection
            from src.strategy_selector import select_best_strategy
            from src.execution_strategies import STRATEGIES

            if position:
                # Reuse whichever strategy opened this position (if we have a
                # record of it) instead of re-selecting every cycle -- see the
                # note on self._active_strategy in __init__ for why.
                best_strategy_name = self._active_strategy.get(symbol) or select_best_strategy(regime)
                self._active_strategy[symbol] = best_strategy_name
            else:
                best_strategy_name = select_best_strategy(regime)
            active_strategy = STRATEGIES.get(best_strategy_name)

            if not active_strategy:
                raise ValueError(f"Selected strategy {best_strategy_name} not found")

            # Generate signal from the selected strategy
            strat_signal = active_strategy.generate_signal(symbol, current_price, position, regime_data)
            
            action = strat_signal.get("action", "hold")
            reason = f"[{best_strategy_name.upper()}] " + strat_signal.get("reason", "No signal")

            # If we are in a position, verify price-based exits (SL/TP overrides strategy)
            if position:
                exit_signal = self._check_price_based_exits(symbol, current_price, position)
                if exit_signal:
                    exit_signal["regime"] = regime
                    exit_signal["rsi"] = rsi
                    exit_signal["atr"] = regime_data.get("atr", 0.0)
                    exit_signal["features"] = regime_data
                    exit_signal["selected_strategy"] = best_strategy_name
                    self._active_strategy.pop(symbol, None)  # position is closing
                    return exit_signal

                # Strategy-based exit
                if action == "close":
                    self._active_strategy.pop(symbol, None)  # position is closing
                    return {
                        "symbol": symbol,
                        "action": "close",
                        "reason": reason,
                        "regime": regime,
                        "rsi": rsi,
                        "atr": regime_data.get("atr", 0.0),
                        "features": regime_data,
                        "selected_strategy": best_strategy_name
                    }

            if action in ["buy", "sell"]:
                self._active_strategy[symbol] = best_strategy_name  # remember who's opening this
                return {
                    "symbol": symbol,
                    "action": action,
                    "reason": reason,
                    "regime": regime,
                    "rsi": rsi,
                    "atr": regime_data.get("atr", 0.0),
                    "htf_trend": regime_data.get("htf_trend"),
                    "features": regime_data,
                    "selected_strategy": best_strategy_name
                }

            return {
                "symbol": symbol,
                "action": "hold",
                "reason": "no_signal",
                "regime": regime,
                "rsi": rsi,
                "atr": regime_data.get("atr", 0.0),
                "features": regime_data
            }

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return {
                "symbol": symbol,
                "action": "stand_aside",
                "reason": "error",
                "error": str(e),
                "regime": "neutral",
                "rsi": 50.0,
                "atr": 0.0,
                "features": {}
            }

    def _check_price_based_exits(self, symbol: str, current_price: float, position: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check price-based exit conditions: stop loss, profit target, trailing stop, max hold."""
        try:
            entry_price = float(position.get("avg_entry_price", 0))
            qty = float(position.get("qty", 0))
            side = "long" if qty > 0 else "short"

            if entry_price <= 0:
                return None

            # Calculate P&L percentage
            if side == "long":
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100

            # Check profit target
            if pnl_pct >= settings.PROFIT_TARGET_PCT * 100:
                return {
                    "symbol": symbol,
                    "action": "close",
                    "reason": "profit_target_reached",
                    "pnl_pct": pnl_pct
                }

            # Check stop loss
            if pnl_pct <= -settings.STOP_LOSS_PCT * 100:
                return {
                    "symbol": symbol,
                    "action": "close",
                    "reason": "stop_loss_hit",
                    "pnl_pct": pnl_pct
                }

            # Trailing stop (if enabled)
            if settings.TRAILING_STOP_ENABLED:
                # Track peak price for trailing stop
                if not hasattr(self, '_trailing_peaks'):
                    self._trailing_peaks = {}

                if symbol not in self._trailing_peaks:
                    self._trailing_peaks[symbol] = current_price

                # Update peak
                if side == "long":
                    self._trailing_peaks[symbol] = max(self._trailing_peaks[symbol], current_price)
                    peak = self._trailing_peaks[symbol]
                    # Check if price dropped from peak by trailing distance
                    if (peak - current_price) / peak * 100 >= settings.TRAILING_DISTANCE_PCT * 100:
                        # Only activate after profit threshold reached
                        if (peak - entry_price) / entry_price * 100 >= settings.TRAILING_ACTIVATION_PCT * 100:
                            return {
                                "symbol": symbol,
                                "action": "close",
                                "reason": "trailing_stop_hit",
                                "peak": peak,
                                "pnl_pct": pnl_pct
                            }
                else:  # short position
                    self._trailing_peaks[symbol] = min(self._trailing_peaks[symbol], current_price)
                    peak = self._trailing_peaks[symbol]
                    if (current_price - peak) / peak * 100 >= settings.TRAILING_DISTANCE_PCT * 100:
                        if (entry_price - peak) / entry_price * 100 >= settings.TRAILING_ACTIVATION_PCT * 100:
                            return {
                                "symbol": symbol,
                                "action": "close",
                                "reason": "trailing_stop_hit",
                                "peak": peak,
                                "pnl_pct": pnl_pct
                            }

            # Max hold time check
            if "created_at" in position:
                from datetime import datetime, timezone
                created = datetime.fromisoformat(position["created_at"].replace("Z", "+00:00"))
                hold_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
                if hold_hours >= settings.MAX_HOLD_HOURS:
                    return {
                        "symbol": symbol,
                        "action": "close",
                        "reason": "max_hold_time_exceeded",
                        "hold_hours": hold_hours
                    }

            return None

        except Exception as e:
            logger.error(f"Price-based exit check failed: {e}")
            return None
