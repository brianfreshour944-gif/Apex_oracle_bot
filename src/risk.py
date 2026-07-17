"""Modern risk management with position sizing and killswitch logic."""

import asyncio
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from src.config import settings
from src.logging_config import get_logger
from src.exchange import AlpacaExchange

logger = get_logger(__name__)

class RiskManager:
    """Modern risk management with position sizing and drawdown protection."""

    def __init__(self, exchange: AlpacaExchange):
        self.exchange = exchange
        self.peak_equity = 0.0
        self.daily_pnl = 0.0
        self.open_positions = []
        self.last_check_time = datetime.now(timezone.utc)

    async def update_account_status(self) -> Dict[str, Any]:
        """Update account status and check risk limits."""
        try:
            account = await self.exchange.get_account()
            positions = await self.exchange.get_positions()

            equity = float(account.get("equity", 0))
            cash = float(account.get("cash", 0))
            portfolio_value = float(account.get("portfolio_value", 0))

            # Update peak equity for drawdown calculation
            if equity > self.peak_equity:
                self.peak_equity = equity

            # Calculate drawdown from peak
            drawdown_pct = ((equity - self.peak_equity) / self.peak_equity) * 100 if self.peak_equity > 0 else 0

            # Check if we've hit max drawdown limit
            if drawdown_pct < settings.MAX_DRAWDOWN_STOP:
                logger.critical(f"MAX DRAWDOWN LIMIT HIT: {drawdown_pct:.2f}%")
                return {
                    "status": "killswitch_activated",
                    "reason": "max_drawdown_exceeded",
                    "drawdown_pct": drawdown_pct,
                    "action": "liquidate_all"
                }

            # Reset daily PnL if new day
            now = datetime.now(timezone.utc)
            if now.day != self.last_check_time.day:
                self.daily_pnl = 0.0
                self.last_check_time = now
                # Reset peak equity to current equity at start of new day
                self.peak_equity = equity

            # Update daily PnL (actual change since start of day)
            # Track start-of-day equity separately
            if not hasattr(self, 'start_of_day_equity'):
                self.start_of_day_equity = equity
            self.daily_pnl = equity - self.start_of_day_equity

            # Check daily loss limit (scaled to actual equity)
            daily_loss_limit_abs = settings.DAILY_LOSS_LIMIT / 100.0 * equity
            if self.daily_pnl < daily_loss_limit_abs:
                logger.critical(f"DAILY LOSS LIMIT HIT: ${self.daily_pnl:.2f} (limit: ${daily_loss_limit_abs:.2f})")
                return {
                    "status": "killswitch_activated",
                    "reason": "daily_loss_limit_exceeded",
                    "daily_pnl": self.daily_pnl,
                    "action": "liquidate_all"
                }

            # Check position limits
            position_count = len(positions)
            if position_count > settings.MAX_OPEN_POSITIONS:
                logger.warning(f"Position limit exceeded: {position_count}/{settings.MAX_OPEN_POSITIONS}")
                return {
                    "status": "position_limit_exceeded",
                    "reason": "too_many_open_positions",
                    "action": "reduce_positions"
                }

            # Calculate current exposure
            current_exposure = sum(float(p.get("market_value", 0)) for p in positions)

            # Check portfolio value cap (absolute dollar amount)
            max_portfolio_abs = float(settings.MAX_PORTFOLIO_VALUE)
            if current_exposure > max_portfolio_abs:
                logger.warning(f"Portfolio value cap exceeded: ${current_exposure:.2f} (cap: ${max_portfolio_abs:.2f})")
                return {
                    "status": "exposure_limit_exceeded",
                    "reason": "max_portfolio_value_exceeded",
                    "action": "reduce_exposure"
                }

            return {
                "status": "risk_ok",
                "equity": equity,
                "cash": cash,
                "portfolio_value": portfolio_value,
                "drawdown_pct": drawdown_pct,
                "daily_pnl": self.daily_pnl,
                "open_positions": position_count,
                "current_exposure": current_exposure
            }

        except Exception as e:
            logger.error(f"Risk update failed: {e}")
            # Don't treat transient errors as killswitch - return error but no liquidation
            return {
                "status": "error",
                "error": str(e),
                "action": "stand_aside"
            }

    def calculate_position_size(self, symbol: str, current_price: float, regime: str) -> Tuple[float, str]:
        """Calculate position size based on risk parameters and regime."""
        try:
            # Base position size calculation
            risk_amount = settings.ACCOUNT_BASE * settings.BASE_RISK_PERCENT
            position_size = risk_amount / current_price

            # Apply regime-specific adjustments
            if regime == "trending":
                # More aggressive in trending markets
                position_size = min(position_size * 1.5, settings.MAX_SINGLE_TRADE_USD / current_price)
            elif regime == "mean_reverting":
                # More conservative in mean-reverting markets
                position_size = min(position_size * 0.8, settings.MAX_SINGLE_TRADE_USD / current_price)
            else:
                # Neutral regime
                position_size = min(position_size, settings.MAX_SINGLE_TRADE_USD / current_price)

            # Apply hard cap
            position_size = min(position_size, settings.MAX_SINGLE_TRADE_USD / current_price)

            # Round to reasonable precision
            position_size = round(position_size, 6)

            return position_size, "ok"

        except Exception as e:
            logger.error(f"Position sizing failed: {e}")
            return 0.0, f"error: {e}"

    async def check_killswitch_conditions(self) -> bool:
        """Check if killswitch conditions are met (only real risk breaches, not errors)."""
        status = await self.update_account_status()
        # Only activate on actual risk breaches, NOT on transient errors
        return status.get("status") == "killswitch_activated"

    async def liquidate_all_positions(self) -> Dict[str, Any]:
        """Liquidate all open positions."""
        try:
            positions = await self.exchange.get_positions()

            results = []
            for position in positions:
                symbol = position["symbol"]
                qty = position["qty"]

                # Close position (sell if long, buy if short)
                side = "sell" if float(qty) > 0 else "buy"
                qty_abs = abs(float(qty))

                order_result = await self.exchange.create_order(
                    symbol=symbol,
                    qty=qty_abs,
                    side=side,
                    type="market"
                )

                results.append({
                    "symbol": symbol,
                    "original_qty": qty,
                    "close_order": order_result
                })

            logger.critical("ALL POSITIONS LIQUIDATED - KILLSWITCH ACTIVATED")
            return {
                "status": "liquidation_complete",
                "results": results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Liquidation failed: {e}")
            return {
                "status": "liquidation_failed",
                "error": str(e)
            }