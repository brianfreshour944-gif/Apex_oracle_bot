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
        self.peak_prices: Dict[str, float] = {}  # Tracks highest price seen while in position
        self.last_check_time = datetime.now(timezone.utc)
        # Protects against a race condition where multiple symbols are evaluated
        # concurrently (asyncio tasks) and could each independently pass an
        # exposure-cap check before any of their sibling orders have actually
        # settled on the exchange.
        self._exposure_lock = asyncio.Lock()
        self._equity_lock = asyncio.Lock()
        self._reserved_exposure = []  # list of (notional_amount, reserved_at) tuples

    async def update_account_status(
        self,
        account: Optional[Dict[str, Any]] = None,
        positions: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Update account status and check risk limits.

        Accepts optionally pre-fetched `account`/`positions` so callers that
        already have fresh data from an earlier call in the same processing
        pass can avoid a redundant network round-trip to Alpaca. Measured:
        without this, a single symbol's buy-path evaluation called
        get_account()/get_positions() 2-3x each (bot.py's own fetch, this
        method's fetch, and check_and_reserve_exposure's internal fetch) for
        no additional data -- 45 redundant calls across 15 concurrently
        evaluated symbols in one cycle.
        """
        try:
            if account is None:
                account = await self.exchange.get_account()
            if positions is None:
                positions = await self.exchange.get_positions()

            equity = float(account.get("equity", 0))
            cash = float(account.get("cash", 0))
            portfolio_value = float(account.get("portfolio_value", 0))

            # Update peak equity for drawdown calculation (protected by lock)
            async with self._equity_lock:
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
                async with self._equity_lock:
                    self.daily_pnl = 0.0
                    self.start_of_day_equity = equity
                self.last_check_time = now

            # Update daily PnL (actual change since start of day)
            if not hasattr(self, 'start_of_day_equity'):
                async with self._equity_lock:
                    self.start_of_day_equity = equity
            async with self._equity_lock:
                self.daily_pnl = equity - self.start_of_day_equity
                daily_pnl = self.daily_pnl

            # Check daily loss limit (scaled to actual equity)
            daily_loss_limit_abs = settings.DAILY_LOSS_LIMIT / 100.0 * equity
            if daily_pnl < daily_loss_limit_abs:
                logger.critical(f"DAILY LOSS LIMIT HIT: ${daily_pnl:.2f} (limit: ${daily_loss_limit_abs:.2f})")
                return {
                    "status": "killswitch_activated",
                    "reason": "daily_loss_limit_exceeded",
                    "daily_pnl": daily_pnl,
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
                "daily_pnl": daily_pnl,
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
    ) -> Tuple[float, str]:
        """Portfolio Optimization: Volatility Parity, Correlation VaR, and Cash Allocation."""
        try:
            # 1. Dynamic Cash Allocation (base risk scales with actual equity, not hardcoded base)
            account_equity = self.peak_equity if self.peak_equity > 0 else settings.ACCOUNT_BASE
            risk_amount = account_equity * settings.BASE_RISK_PERCENT

            # 2. Apply regime-specific adjustments to RISK AMOUNT (not position size),
            # so the effective dollar risk is transparent and matches the intended percentage.
            if regime == "trending":
                risk_amount *= 1.5
            elif regime == "mean_reverting":
                risk_amount *= 0.8
            elif regime == "high_volatility":
                risk_amount *= 0.5

            # Confidence weighting applied to risk amount so the effective risk
            # is base_risk * conf_weight, not an opaque position-size inflation.
            conf_weight = np.clip(confidence, 0.5, 1.5)
            risk_amount *= conf_weight

            # 3. Volatility Targeting (Equal risk contribution)
            # Use the actual stop-loss percentage distance for sizing so that
            # the position size matches the real SL trigger distance.
            stop_distance = current_price * settings.STOP_LOSS_PCT
            if atr is not None and atr > 0:
                atr_stop_distance = atr * getattr(settings, "ATR_STOP_MULTIPLIER", 2.0)
                # Use the smaller of ATR-based and %-based stop distances
                # so the tighter stop governs the actual risk.
                stop_distance = min(stop_distance, atr_stop_distance)

            position_size = risk_amount / stop_distance

            # 4. Correlation Matrix & Portfolio VaR
            if returns_matrix and len(returns_matrix) > 1:
                keys = list(returns_matrix.keys())
                
                # Build correlation matrix
                valid_keys = [k for k in keys if len(returns_matrix[k]) == len(returns_matrix.get(symbol, []))]
                if len(valid_keys) > 1:
                    returns_arr = np.array([returns_matrix[k] for k in valid_keys])
                    corr_matrix = np.corrcoef(returns_arr)
                    
                    # Find our symbol's index
                    if symbol in valid_keys:
                        idx = valid_keys.index(symbol)
                        symbol_corrs = corr_matrix[idx]
                        avg_corr = (np.sum(symbol_corrs) - 1.0) / (len(valid_keys) - 1)
                        
                        # Maximize diversification: scale down heavily if entering correlated cluster
                        if avg_corr > 0.5:
                            penalty = max(0.2, 1.0 - (avg_corr - 0.5) * 2) # e.g. 0.8 corr -> 0.4 penalty
                            logger.info(f"Portfolio Optimizer: {symbol} correlation cluster ({avg_corr:.2f}). VaR constraint -> size x{penalty:.2f}.")
                            position_size *= penalty

            # 5. Maximum Sector/Direction Exposure (Hard Cap)
            position_size = min(position_size, settings.MAX_SINGLE_TRADE_USD / current_price)
            position_size = round(position_size, 6)

            # Guard: NaN propagates silently through np.clip/min/round without raising.
            # If confidence was NaN (e.g. scoring math underflowed) we must reject here
            # before a NaN qty reaches the exchange API.
            if position_size != position_size:  # IEEE 754: only NaN != NaN
                logger.error(
                    f"calculate_position_size({symbol}): NaN position size detected "
                    f"(confidence={confidence}). Rejecting trade."
                )
                return 0.0, "error: NaN position size (confidence was NaN)"

            return position_size, "ok"

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return 0.0, f"error: {e}"


    def check_trailing_stop(self, symbol: str, current_price: float, avg_entry_price: float, qty: float) -> str:
        """
        Check if a trailing stop should be activated or triggered.
        Returns: 'close' if triggered, 'hold' otherwise.
        Handles both long and short positions.
        """
        if not settings.TRAILING_STOP_ENABLED or avg_entry_price <= 0:
            return "hold"

        is_long = float(qty) > 0

        if is_long:
            unrealized_pct = (current_price - avg_entry_price) / avg_entry_price

            if unrealized_pct >= settings.TRAILING_ACTIVATION_PCT:
                if symbol not in self.peak_prices or current_price > self.peak_prices[symbol]:
                    self.peak_prices[symbol] = current_price

            if symbol in self.peak_prices:
                peak = self.peak_prices[symbol]
                drawdown_from_peak = (peak - current_price) / peak

                if drawdown_from_peak >= settings.TRAILING_DISTANCE_PCT:
                    logger.info(f"Trailing stop triggered for {symbol}: Peak {peak:.2f}, Current {current_price:.2f}")
                    del self.peak_prices[symbol]
                    return "close"

        else:
            unrealized_pct = (avg_entry_price - current_price) / avg_entry_price

            if unrealized_pct >= settings.TRAILING_ACTIVATION_PCT:
                if symbol not in self.peak_prices or current_price < self.peak_prices[symbol]:
                    self.peak_prices[symbol] = current_price

            if symbol in self.peak_prices:
                trough = self.peak_prices[symbol]
                drawdown_from_trough = (current_price - trough) / trough

                if drawdown_from_trough >= settings.TRAILING_DISTANCE_PCT:
                    logger.info(f"Trailing stop triggered for {symbol} (short): Trough {trough:.2f}, Current {current_price:.2f}")
                    del self.peak_prices[symbol]
                    return "close"

        return "hold"

    async def check_killswitch_conditions(self) -> bool:
        """Check if killswitch conditions are met (only real risk breaches, not errors)."""
        status = await self.update_account_status()
        # Only activate on actual risk breaches, NOT on transient errors
        return status.get("status") == "killswitch_activated"

    async def check_and_reserve_exposure(
        self,
        requested_notional: float,
        min_notional: float = 10.0,
        current_exposure: Optional[float] = None,
    ) -> Tuple[float, str]:
        """Returns the APPROVED notional (may be smaller than requested,
        capped to remaining headroom) rather than a strict pass/fail, so
        available capacity gets used instead of sitting idle. Returns
        (0.0, reason) if headroom is below min_notional.

        `current_exposure` should be passed in by the caller (from a
        `update_account_status()` call already made earlier in the same
        processing pass) whenever available. Measured impact of NOT doing
        this: the account/position re-fetch previously ran *inside*
        `self._exposure_lock`, so it serialized across every concurrently
        evaluated symbol -- 15 concurrent calls took ~2.85s (matching the
        fully-serialized prediction of ~2.4s) instead of running in
        parallel, because only one task can hold the lock while it waits on
        the network. With `current_exposure` supplied, the locked section is
        pure in-memory arithmetic and the same 15 concurrent calls complete
        in <1ms combined.
        """
        if current_exposure is None:
            # Fallback path for callers that don't have a fresh status handy.
            # This still does I/O, but OUTSIDE the lock, so it can run
            # concurrently with other symbols' fallback fetches instead of
            # serializing them.
            status = await self.update_account_status()
            if status.get("status") != "risk_ok":
                return 0.0, status.get("reason", status.get("status", "risk_check_failed"))
            current_exposure = status["current_exposure"]

        async with self._exposure_lock:
            now = datetime.now(timezone.utc)
            self._reserved_exposure = [(a, t) for a, t in self._reserved_exposure if (now - t).total_seconds() < 60]
            reserved_total = sum(a for a, _ in self._reserved_exposure)
            max_portfolio_abs = float(settings.MAX_PORTFOLIO_VALUE)
            headroom = max_portfolio_abs - current_exposure - reserved_total
            if headroom < min_notional:
                logger.warning(f"Exposure reservation denied: current=${current_exposure:.2f} reserved=${reserved_total:.2f} headroom=${headroom:.2f} (below min ${min_notional:.2f}) cap=${max_portfolio_abs:.2f}")
                return 0.0, "max_portfolio_value_would_be_exceeded"
            approved = min(requested_notional, headroom)
            if approved < requested_notional:
                logger.info(f"Exposure reservation partially approved: requested=${requested_notional:.2f} approved=${approved:.2f} (headroom-limited)")
            self._reserved_exposure.append((approved, now))
            return approved, "ok"

    async def reduce_exposure_to_cap(self) -> Dict[str, Any]:
        """Close positions, worst unrealized P&L first, until exposure is
        back under the cap. Targeted/incremental, unlike liquidate_all_positions."""
        try:
            positions = await self.exchange.get_positions()
            max_portfolio_abs = float(settings.MAX_PORTFOLIO_VALUE)
            current_exposure = sum(float(p.get("market_value", 0)) for p in positions)
            if current_exposure <= max_portfolio_abs:
                return {"status": "no_action_needed"}
            positions_sorted = sorted(positions, key=lambda p: float(p.get("unrealized_pl", 0)))
            results = []
            for position in positions_sorted:
                if current_exposure <= max_portfolio_abs:
                    break
                symbol = position["symbol"]
                qty = position["qty"]
                side = "sell" if float(qty) > 0 else "buy"
                qty_abs = abs(float(qty))
                market_value = float(position.get("market_value", 0))
                order_result = await self.exchange.create_order(symbol=symbol, qty=qty_abs, side=side, type="market")
                filled_price = order_result.get("filled_avg_price", 0.0)
                filled_qty = order_result.get("filled_qty", qty_abs)
                actual_value = filled_price * filled_qty if filled_price > 0 else market_value
                current_exposure -= actual_value
                results.append({"symbol": symbol, "closed_qty": qty, "order": order_result})
                logger.warning(f"Closed {symbol} to reduce exposure (was ${market_value:.2f}, actual=${actual_value:.2f})")
            logger.warning(f"Exposure reduction complete. New exposure: ${current_exposure:.2f} (cap: ${max_portfolio_abs:.2f})")
            return {"status": "exposure_reduced", "results": results, "final_exposure": current_exposure}
        except Exception as e:
            logger.error(f"Exposure reduction failed: {e}")
            return {"status": "reduction_failed", "error": str(e)}

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