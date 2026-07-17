"""Modern notifier module with comprehensive type hints."""

from typing import Optional
from src.logging_config import get_logger
from src.config import settings

logger = get_logger(__name__)

async def send_trade_alert(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    order_id: Optional[str] = None,
    pnl: Optional[float] = None,
    is_entry: bool = True,
) -> None:
    """Send trade alert notification."""
    try:
        # In a real implementation, this would send to Telegram, etc.
        logger.info(f"Trade alert: {side} {qty:.6f} {symbol} @ {price}")
    except Exception as e:
        logger.error(f"Failed to send trade alert: {e}")

async def send_heartbeat_alert(
    equity: float,
    daily_pnl_pct: float,
    open_positions: int,
    buying_power: float,
) -> None:
    """Send heartbeat alert notification."""
    try:
        # In a real implementation, this would send to monitoring systems
        logger.info(f"Heartbeat: Equity ${equity:,.2f}, PnL {daily_pnl_pct:.2f}%, Positions {open_positions}")
    except Exception as e:
        logger.error(f"Failed to send heartbeat alert: {e}")

async def send_killswitch_alert(reason: str) -> None:
    """Send killswitch alert notification."""
    try:
        # In a real implementation, this would send urgent alerts
        logger.critical(f"KILLSWITCH ACTIVATED: {reason}")
    except Exception as e:
        logger.error(f"Failed to send killswitch alert: {e}")