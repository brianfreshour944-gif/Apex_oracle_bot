"""Distinct trading strategy implementations for the AI Strategy Selector."""

from typing import Dict, Any, Optional
from src.config import settings
from src.logging_config import get_logger

logger = get_logger("execution_strategies")

class BaseExecutionStrategy:
    """Base class for all execution strategies."""
    def __init__(self):
        self.name = "base"

    def generate_signal(self, symbol: str, current_price: float, position: Optional[Dict], features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a trading signal based on current features."""
        raise NotImplementedError


class TrendFollowingStrategy(BaseExecutionStrategy):
    """Rides strong trends, buying pullbacks in uptrends and shorting rallies in downtrends."""
    def __init__(self):
        super().__init__()
        self.name = "trend_following"

    def generate_signal(self, symbol: str, current_price: float, position: Optional[Dict], features: Dict[str, Any]) -> Dict[str, Any]:
        rsi = features.get("rsi", 50.0)
        prev_rsi = features.get("prev_rsi", 50.0)
        htf_trend = features.get("htf_trend", "neutral")

        if not position:
            # Trend following buys upward momentum or pullbacks in a bull trend.
            # Using 65/35 instead of 55/45 to be less restrictive.
            if htf_trend == "bullish" and rsi < 65.0 and rsi > prev_rsi:
                return {"action": "buy", "reason": f"Trend Following: Bullish momentum (RSI {rsi:.1f})"}
            elif htf_trend == "bearish" and rsi > 35.0 and rsi < prev_rsi:
                return {"action": "sell", "reason": f"Trend Following: Bearish momentum (RSI {rsi:.1f})"}
        else:
            qty = float(position.get("qty", 0))
            if qty > 0 and htf_trend == "bearish":
                return {"action": "close", "reason": "Trend Following: Macro trend flipped bearish"}
            elif qty < 0 and htf_trend == "bullish":
                return {"action": "close", "reason": "Trend Following: Macro trend flipped bullish"}

        return {"action": "hold", "reason": "No trend signal"}


class MeanReversionStrategy(BaseExecutionStrategy):
    """Fades extremes. Buys when oversold, sells when overbought. Best in sideways regimes."""
    def __init__(self):
        super().__init__()
        self.name = "mean_reversion"

    def generate_signal(self, symbol: str, current_price: float, position: Optional[Dict], features: Dict[str, Any]) -> Dict[str, Any]:
        rsi = features.get("rsi", 50.0)
        prev_rsi = features.get("prev_rsi", 50.0)
        price_zscore = features.get("price_zscore", 0.0)
        z_thresh = getattr(settings, "BB_ZSCORE_THRESHOLD", 1.5)

        if not position:
            # Primary: RSI oversold/overbought bounce
            if rsi < (settings.RSI_OVERSOLD + 10.0) and rsi > prev_rsi:
                return {"action": "buy", "reason": f"Mean Reversion: Oversold bounce (RSI {rsi:.1f})"}
            elif rsi > (settings.RSI_OVERBOUGHT - 10.0) and rsi < prev_rsi:
                return {"action": "sell", "reason": f"Mean Reversion: Overbought rejection (RSI {rsi:.1f})"}
            # Fallback: price z-score when RSI is neutral (40-60)
            elif 40.0 <= rsi <= 60.0:
                if price_zscore <= -z_thresh:
                    return {"action": "buy", "reason": f"Mean Reversion: Price {price_zscore:.1f}sd below mean, RSI neutral ({rsi:.1f})"}
                elif price_zscore >= z_thresh:
                    return {"action": "sell", "reason": f"Mean Reversion: Price {price_zscore:.1f}sd above mean, RSI neutral ({rsi:.1f})"}
        else:
            qty = float(position.get("qty", 0))
            # Require BOTH the RSI recovery AND actual price reversion
            # toward the mean (z-score no longer meaningfully on the entry
            # side) before exiting. RSI alone let z-score-fallback entries
            # (which can start with RSI already at 50-54, since that's the
            # whole point of the neutral-band fallback) close out again on
            # the very next cycle after a trivial 1-2 point RSI tick, with
            # no real price movement - producing rapid buy/sell churn that
            # loses money on spread alone.
            if qty > 0 and rsi > settings.RSI_NEUTRAL_SELL and price_zscore > -0.5:
                return {"action": "close", "reason": f"Mean Reversion: Target RSI reached ({rsi:.1f}), price reverted"}
            elif qty < 0 and rsi < settings.RSI_NEUTRAL_BUY and price_zscore < 0.5:
                return {"action": "close", "reason": f"Mean Reversion: Target RSI reached ({rsi:.1f}), price reverted"}

        return {"action": "hold", "reason": "No mean reversion signal"}


class MomentumStrategy(BaseExecutionStrategy):
    """Aggressive entries on acceleration."""
    def __init__(self):
        super().__init__()
        self.name = "momentum"

    def generate_signal(self, symbol: str, current_price: float, position: Optional[Dict], features: Dict[str, Any]) -> Dict[str, Any]:
        rsi = features.get("rsi", 50.0)
        prev_rsi = features.get("prev_rsi", 50.0)
        rsi_roc = rsi - prev_rsi

        if not position:
            if rsi > 60.0 and rsi_roc > 5.0:
                return {"action": "buy", "reason": f"Momentum: Strong bullish acceleration (RSI ROC +{rsi_roc:.1f})"}
            elif rsi < 40.0 and rsi_roc < -5.0:
                return {"action": "sell", "reason": f"Momentum: Strong bearish acceleration (RSI ROC {rsi_roc:.1f})"}
        else:
            qty = float(position.get("qty", 0))
            if qty > 0 and rsi_roc < -2.0:
                return {"action": "close", "reason": "Momentum: Loss of bullish momentum"}
            elif qty < 0 and rsi_roc > 2.0:
                return {"action": "close", "reason": "Momentum: Loss of bearish momentum"}

        return {"action": "hold", "reason": "No momentum signal"}


class BreakoutStrategy(BaseExecutionStrategy):
    """Enters when volatility spikes in the direction of the trend."""
    def __init__(self):
        super().__init__()
        self.name = "breakout"

    def generate_signal(self, symbol: str, current_price: float, position: Optional[Dict], features: Dict[str, Any]) -> Dict[str, Any]:
        regime = features.get("regime", "neutral")
        rsi = features.get("rsi", 50.0)

        # Basic proxy for breakout: high volatility combined with directional RSI
        if not position:
            if regime == "high_volatility":
                if rsi > 55.0:
                    return {"action": "buy", "reason": "Breakout: High Volatility Bullish Break"}
                elif rsi < 45.0:
                    return {"action": "sell", "reason": "Breakout: High Volatility Bearish Break"}
        else:
            qty = float(position.get("qty", 0))
            if qty > 0 and rsi < 50.0:
                return {"action": "close", "reason": "Breakout: Breakout failed / retraced"}
            elif qty < 0 and rsi > 50.0:
                return {"action": "close", "reason": "Breakout: Breakout failed / retraced"}

        return {"action": "hold", "reason": "No breakout signal"}


class GridStrategy(BaseExecutionStrategy):
    """Simplified systematic mean-reversion with fixed grid bands."""
    def __init__(self):
        super().__init__()
        self.name = "grid"

    def generate_signal(self, symbol: str, current_price: float, position: Optional[Dict], features: Dict[str, Any]) -> Dict[str, Any]:
        # Pseudo-grid using RSI bands for continuous accumulation.
        rsi = features.get("rsi", 50.0)
        
        if not position:
            if rsi < 40.0:
                return {"action": "buy", "reason": "Grid: Lower bound touched"}
            elif rsi > 60.0:
                return {"action": "sell", "reason": "Grid: Upper bound touched"}
        else:
            entry = float(position.get("avg_entry_price", 0.0))
            qty = float(position.get("qty", 0))
            if entry > 0:
                pct_change = (current_price - entry) / entry
                if qty > 0 and pct_change >= settings.PROFIT_TARGET_PCT * 0.5:
                    return {"action": "close", "reason": f"Grid: Profit band hit (+{pct_change*100:.1f}%)"}
                elif qty < 0 and pct_change <= -settings.PROFIT_TARGET_PCT * 0.5:
                    return {"action": "close", "reason": f"Grid: Profit band hit ({-pct_change*100:.1f}%)"}

        return {"action": "hold", "reason": "Within grid bands"}


class ScalpingStrategy(BaseExecutionStrategy):
    """Very fast signals, looking for minor ticks."""
    def __init__(self):
        super().__init__()
        self.name = "scalping"

    def generate_signal(self, symbol: str, current_price: float, position: Optional[Dict], features: Dict[str, Any]) -> Dict[str, Any]:
        rsi = features.get("rsi", 50.0)
        prev_rsi = features.get("prev_rsi", 50.0)

        if not position:
            if rsi > prev_rsi and rsi < 55.0:
                return {"action": "buy", "reason": "Scalp: Micro-bounce detected"}
            elif rsi < prev_rsi and rsi > 45.0:
                return {"action": "sell", "reason": "Scalp: Micro-rejection detected"}
        else:
            entry = float(position.get("avg_entry_price", 0.0))
            qty = float(position.get("qty", 0))
            if entry > 0:
                pct_change = (current_price - entry) / entry
                # Very tight scalping take profit (1/3rd normal)
                tight_tp = settings.PROFIT_TARGET_PCT / 3.0
                if qty > 0 and pct_change >= tight_tp:
                    return {"action": "close", "reason": "Scalp: Quick target reached"}
                elif qty < 0 and pct_change <= -tight_tp:
                    return {"action": "close", "reason": "Scalp: Quick target reached"}

        return {"action": "hold", "reason": "No scalp opportunity"}


# Registry of available strategies
STRATEGIES = {
    "trend_following": TrendFollowingStrategy(),
    "mean_reversion": MeanReversionStrategy(),
    "momentum": MomentumStrategy(),
    "breakout": BreakoutStrategy(),
    "grid": GridStrategy(),
    "scalping": ScalpingStrategy()
}
