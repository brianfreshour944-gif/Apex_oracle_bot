"""Modern risk management module with comprehensive type hints."""

from typing import Tuple
from src.config import settings

def check_account_killswitches(equity: float, starting_equity: float) -> Tuple[bool, str]:
    """Check if account killswitches have been breached."""
    # Check drawdown limit
    drawdown_pct = ((equity - starting_equity) / starting_equity) * 100.0
    if drawdown_pct <= settings.MAX_DRAWDOWN_STOP:
        return True, f"Max drawdown limit breached: {drawdown_pct:.2f}%"

    # Check daily loss limit (simplified - would need daily equity tracking in real implementation)
    if drawdown_pct <= settings.DAILY_LOSS_LIMIT:
        return True, f"Daily loss limit breached: {drawdown_pct:.2f}%"

    return False, ""

def calculate_position_size(
    equity: float,
    current_price: float,
    atr: float,
    multiplier: float = 1.0
) -> float:
    """Calculate position size based on risk management rules."""
    # Calculate risk amount
    risk_amount = equity * settings.BASE_RISK_PERCENT

    # Calculate position size using ATR-based stop
    stop_distance = atr * multiplier
    position_size = (risk_amount / stop_distance) if stop_distance > 0 else 0.0

    # Apply hard caps
    max_trade_size = min(settings.MAX_SINGLE_TRADE_USD / current_price, position_size)
    return min(max_trade_size, settings.MAX_SINGLE_TRADE_USD / current_price)