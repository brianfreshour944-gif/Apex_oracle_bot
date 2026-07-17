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

class TradingStrategy:
    """Modern trading strategy with regime classification and signal generation."""

    def __init__(self, exchange: AlpacaExchange):
        self.exchange = exchange
        self.current_regime = "neutral"
        self.last_analysis_time = None

    async def analyze_market_regime(self, symbol: str, timeframe: str = "1D", limit: int = 100) -> Dict[str, Any]:
        """Analyze market regime using Hurst exponent, ATR, and RSI."""
        try:
            # Get historical data
            bars_df = await self.exchange.get_bars(symbol, timeframe, limit)

            if len(bars_df) < 20:
                logger.warning(f"Insufficient data for {symbol}, only {len(bars_df)} bars")
                return {"regime": "neutral", "confidence": 0.0}

            # Calculate returns for Hurst exponent
            bars_df = bars_df.with_columns(
                (pl.col("close").pct_change()).alias("returns")
            ).drop_nulls()

            # Calculate Hurst exponent (simplified)
            returns = bars_df["returns"].to_numpy()
            hurst = self._calculate_hurst(returns)

            # Calculate ATR
            atr = self._calculate_atr(bars_df)

            # Calculate RSI
            rsi = self._calculate_rsi(bars_df["close"].to_numpy())

            # Classify regime
            if hurst > settings.HURST_TREND_UP:
                regime = "trending"
            elif hurst < settings.HURST_MEAN_REVERT:
                regime = "mean_reverting"
            elif atr > settings.HIGH_VOLATILITY_PCT:
                regime = "high_volatility"
            else:
                regime = "neutral"

            self.current_regime = regime

            return {
                "regime": regime,
                "hurst": float(hurst),
                "atr": float(atr),
                "rsi": float(rsi),
                "confidence": self._calculate_regime_confidence(regime, hurst, atr, rsi)
            }

        except Exception as e:
            logger.error(f"Regime analysis failed: {e}")
            return {"regime": "neutral", "confidence": 0.0}

    def _calculate_hurst(self, returns: np.ndarray) -> float:
        """Calculate Hurst exponent using rescaled range (R/S) analysis."""
        if len(returns) < 20:
            return 0.5

        # Use a proper R/S analysis implementation
        lags = range(2, min(20, len(returns) // 2))
        tau = []

        for lag in lags:
            # Calculate rescaled range for this lag
            # Split returns into non-overlapping windows of size lag
            n_windows = len(returns) // lag
            if n_windows < 1:
                continue

            rs_values = []
            for i in range(n_windows):
                window = returns[i * lag: (i + 1) * lag]
                if len(window) < 2:
                    continue

                # Mean-adjusted series
                mean_adj = window - np.mean(window)

                # Cumulative deviation
                cum_dev = np.cumsum(mean_adj)

                # Range
                r = np.max(cum_dev) - np.min(cum_dev)

                # Standard deviation
                s = np.std(window)

                if s > 0:
                    rs_values.append(r / s)

            if rs_values:
                tau.append(np.mean(rs_values))

        if len(tau) < 2:
            return 0.5

        # Fit line to log-log plot
        poly = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)
        hurst = poly[0]

        # Clamp to reasonable range
        return float(np.clip(hurst, 0.0, 1.0))

    def _calculate_atr(self, bars_df: pl.DataFrame) -> float:
        """Calculate Average True Range."""
        high = bars_df["high"]
        low = bars_df["low"]
        close = bars_df["close"]

        tr = pl.max_horizontal(
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        )

        atr = tr.mean()
        return float(atr)

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period:
            return 50.0

        deltas = np.diff(prices)
        seed = deltas[:period+1]

        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period

        rs = up / down
        rsi = np.where(down == 0, 100, 100 - (100 / (1 + rs)))

        for i in range(period+1, len(prices)):
            delta = deltas[i-1]

            if delta > 0:
                up_val = delta
                down_val = 0.0
            else:
                up_val = 0.0
                down_val = -delta

            up = (up * (period - 1) + up_val) / period
            down = (down * (period - 1) + down_val) / period

            rs = up / down if down != 0 else float('inf')
            rsi = np.append(rsi, 100 - (100 / (1 + rs)) if rs != float('inf') else 100.0)

        return float(rsi[-1])

    def _calculate_regime_confidence(self, regime: str, hurst: float, atr: float, rsi: float) -> float:
        """Calculate confidence in regime classification."""
        if regime == "trending":
            return min(1.0, (hurst - settings.HURST_TREND_UP) * 2.0)
        elif regime == "mean_reverting":
            return min(1.0, (settings.HURST_MEAN_REVERT - hurst) * 2.0)
        elif regime == "high_volatility":
            return min(1.0, (atr - settings.HIGH_VOLATILITY_PCT) / settings.HIGH_VOLATILITY_PCT)
        else:
            return 1.0 - abs(0.5 - hurst) * 2.0

    async def generate_trading_signal(self, symbol: str, current_price: float, position: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate trading signal based on current regime and market conditions."""
        try:
            # Get current market regime
            regime_data = await self.analyze_market_regime(symbol)
            regime = regime_data["regime"]
            rsi = regime_data["rsi"]

            # Skip trading in high volatility regime
            if regime == "high_volatility":
                return {
                    "symbol": symbol,
                    "action": "stand_aside",
                    "reason": "high_volatility_regime",
                    "regime": regime,
                    "rsi": rsi
                }

            # Check if we should enter a position
            if not position:
                if regime == "trending" and rsi < settings.RSI_NEUTRAL_BUY:
                    return {
                        "symbol": symbol,
                        "action": "buy",
                        "reason": "trending_regime_buy_signal",
                        "regime": regime,
                        "rsi": rsi
                    }
                elif regime == "mean_reverting" and rsi > settings.RSI_NEUTRAL_SELL:
                    return {
                        "symbol": symbol,
                        "action": "sell",
                        "reason": "mean_reverting_regime_sell_signal",
                        "regime": regime,
                        "rsi": rsi
                    }

            # Check if we should exit a position (regime flip OR price-based exits)
            if position:
                # Regime-based exit
                if (regime == "trending" and rsi > settings.RSI_NEUTRAL_SELL) or \
                   (regime == "mean_reverting" and rsi < settings.RSI_NEUTRAL_BUY):
                    return {
                        "symbol": symbol,
                        "action": "close",
                        "reason": "regime_change_exit_signal",
                        "regime": regime,
                        "rsi": rsi
                    }

                # Price-based exit checks
                exit_signal = self._check_price_based_exits(symbol, current_price, position)
                if exit_signal:
                    return exit_signal

            return {
                "symbol": symbol,
                "action": "hold",
                "reason": "no_signal",
                "regime": regime,
                "rsi": rsi
            }

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return {
                "symbol": symbol,
                "action": "stand_aside",
                "reason": "error",
                "error": str(e)
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
