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
        self.peak_equity = self._load_peak_equity()
        self.daily_pnl = 0.0
        self.open_positions = []
        self.peak_prices: Dict[str, float] = {}  # Tracks highest price seen while in position
        self.last_check_time = datetime.now(timezone.utc)

    def _load_peak_equity(self) -> float:
        import os, json
        filepath = "/app/data/risk_state.json" if os.path.exists("/app/data") else "risk_state.json"
        try:
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    data = json.load(f)
                    return float(data.get("peak_equity", 0.0))
        except Exception as e:
            logger.error(f"Failed to load peak_equity: {e}")
        return 0.0

    def _save_peak_equity(self) -> None:
        import os, json
        filepath = "/app/data/risk_state.json" if os.path.exists("/app/data") else "risk_state.json"
        try:
            with open(filepath, "w") as f:
                json.dump({"peak_equity": self.peak_equity}, f)
        except Exception as e:
            logger.error(f"Failed to save peak_equity: {e}")

    async def update_account_status(self) -> Dict[str, Any]:
        """Update account status and check risk limits."""
        try:
            account = await self.exchange.get_account()
            positions = await self.exchange.get_positions()

            equity = float(account.get("equity", 0))
            cash = float(account.get("cash", 0))
            portfolio_value = float(account.get("portfolio_value", 0))

            # Update peak equity for drawdown calculation
            if self.peak_equity == 0.0:
                self.peak_equity = equity
                self._save_peak_equity()
            elif equity > self.peak_equity:
                self.peak_equity = equity
                self._save_peak_equity()

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
                self.start_of_day_equity = equity  # FIX: must reset here too --
                                                     # previously only set once
                                                     # ever via hasattr, so this
                                                     # day-boundary check updated
                                                     # last_check_time but never
                                                     # actually refreshed the
                                                     # baseline daily_pnl is
                                                     # computed from, making the
                                                     # "daily" limit silently
                                                     # track cumulative loss
                                                     # since inception instead
                self.last_check_time = now

            # Update daily PnL (actual change since start of day)
            if not hasattr(self, 'start_of_day_equity'):
                self.start_of_day_equity = equity  # first-ever call only
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

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        regime: str,
        atr: Optional[float] = None,
        confidence: float = 1.0,
        returns_matrix: Optional[Dict[str, np.ndarray]] = None,
        equity: float = 10000.0,
    ) -> Tuple[float, str]:
        """Calculate position size with ATR volatility parity, confidence weighting, and correlation check."""
        try:
            risk_amount = equity * settings.BASE_RISK_PERCENT

            # ATR Volatility Parity Sizing (if ATR available)
            if atr is not None and atr > 0:
                stop_distance = atr * getattr(settings, "ATR_STOP_MULTIPLIER", 2.0)
                position_size = risk_amount / stop_distance
            else:
                position_size = risk_amount / current_price

            # Apply regime-specific adjustments
            if regime == "trending":
                position_size *= 1.5
            elif regime == "mean_reverting":
                position_size *= 0.8

            # Confidence weighting (scale between 0.5 and 1.5)
            conf_weight = np.clip(confidence, 0.5, 1.5)
            position_size *= conf_weight

            # Cross-symbol correlation check (downscale if high correlation > 0.85)
            if returns_matrix and len(returns_matrix) > 1:
                keys = list(returns_matrix.keys())
                corrs = []
                for k in keys:
                    if k != symbol and len(returns_matrix[k]) == len(returns_matrix.get(symbol, [])):
                        c = np.corrcoef(returns_matrix[symbol], returns_matrix[k])[0, 1]
                        if not np.isnan(c):
                            corrs.append(c)
                if corrs and np.mean(corrs) > 0.85:
                    logger.warning(f"High portfolio correlation ({np.mean(corrs):.2f}) for {symbol}. Downscaling size by 30%.")
                    position_size *= 0.7

            # Apply hard dollar cap per trade
            position_size = min(position_size, settings.MAX_SINGLE_TRADE_USD / current_price)
            position_size = round(position_size, 6)

            return position_size, "ok"

        except Exception as e:
            logger.error(f"Position sizing failed: {e}")
            return 0.0, f"error: {e}"


    def check_trailing_stop(self, symbol: str, current_price: float, avg_entry_price: float, qty: float) -> str:
        """
        Check if a trailing stop should be activated or triggered.
        Returns: 'close' if triggered, 'hold' otherwise.
        """
        if not settings.TRAILING_STOP_ENABLED or avg_entry_price <= 0:
            return "hold"

        # Note: This logic works best for long positions. 
        # Short positions would track lowest price. Assuming longs for crypto default.
        is_long = float(qty) > 0
        if not is_long:
            return "hold" # Simplified: only trail longs

        # Calculate unrealized PnL %
        unrealized_pct = (current_price - avg_entry_price) / avg_entry_price

        # Have we reached the activation threshold?
        if unrealized_pct >= settings.TRAILING_ACTIVATION_PCT:
            # Track peak price
            if symbol not in self.peak_prices or current_price > self.peak_prices[symbol]:
                self.peak_prices[symbol] = current_price
                
        # If we are tracking a peak price, check if we've fallen below the distance
        if symbol in self.peak_prices:
            peak = self.peak_prices[symbol]
            drawdown_from_peak = (peak - current_price) / peak
            
            if drawdown_from_peak >= settings.TRAILING_DISTANCE_PCT:
                logger.info(f"Trailing stop triggered for {symbol}: Peak {peak:.2f}, Current {current_price:.2f}")
                # Reset peak tracker
                del self.peak_prices[symbol]
                return "close"
                
        return "hold"

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