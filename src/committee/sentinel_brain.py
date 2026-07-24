"""Brain 4: Sentinel Brain.

Hard veto filter for extreme market risk, severe volatility, and emergency flags.
Cannot vote BUY or SELL directly — acts as a protection gatekeeper.
"""

from .models import BrainVote

async def sentinel_brain(symbol: str, price: float, signal: dict) -> BrainVote:
    """Evaluates safety conditions and enforces hard veto if market conditions are dangerous."""
    regime = signal.get("regime", "neutral")
    atr = signal.get("atr", 0.0)

    # Check for extreme risk conditions
    is_veto = False
    reason = "Sentinel cleared trade"
    action = "hold"

    confidence = 0.50

    if signal.get("flash_crash", False):
        is_veto = True
        action = "stand_aside"
        reason = "Hard Veto: Flash crash condition detected"
    elif signal.get("halted", False):
        is_veto = True
        action = "stand_aside"
        reason = "Hard Veto: Trading halted"
    elif price > 0 and atr > 0 and (atr / price) > 0.10:
        # ATR > 10% of price indicates abnormal price swing / gap risk
        is_veto = True
        action = "stand_aside"
        reason = f"Hard Veto: Abnormal ATR volatility ratio ({(atr/price)*100:.1f}%)"
    elif regime == "high_volatility":
        # Soft veto: reduce conviction without outright blocking
        is_veto = False
        action = "stand_aside"
        confidence = 0.50
        reason = f"Soft Veto: High volatility regime '{regime}'"

    return BrainVote(
        name="sentinel",
        action=action,
        confidence=0.95 if is_veto else confidence,
        weight=0.10,
        regime=regime,
        reason=reason,
        is_veto=is_veto
    )
