"""Modern risk management with position sizing and killswitch logic."""

import asyncio
import math
import threading
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from src.config import settings
from src.logging_config import get_logger
from src.exchange import AlpacaExchange

logger = get_logger(__name__)

# Default transaction cost model (used as fallback when dynamic not available)
DEFAULT_TX_COSTS = {
    "fee_bps": 5.0,
    "slippage_bps": 3.0,
    "spread_bps": 2.0,
}

# Regime -> (activation_multiplier, distance_multiplier) for check_trailing_stop.
# Unrecognized/None regime always maps to (1.0, 1.0) -- exact match to the
# static (unscaled) behavior. Starting values, easy to tune once observed.
_TRAILING_REGIME_MULTIPLIERS = {
    "high_volatility": (1.5, 1.5),  # wider both ways -- avoid noise-driven stopouts
    "low_volatility":  (0.6, 0.6),  # tighter -- capture smaller moves when noise is low
    "trending":        (1.0, 1.3),  # let winners run further, same entry bar
    "bull":            (1.0, 1.3),
    "bear":            (1.0, 1.3),
    "sideways":        (0.8, 0.8),  # lock in gains faster, trends don't persist
    "mean_reverting":  (0.8, 0.8),
}

# Correlation isn't actually wired up anywhere live (calculate_position_size's
# returns_matrix parameter has no real caller), so this uses regime as a cheap,
# already-available proxy: symbols currently classified in the same regime
# tend to move together. Cap how much of the position-count budget can pile
# into one regime cluster instead of requiring real return-series correlation.
# floor() (not ceil()) so this has a real effect even at small MAX_OPEN_POSITIONS
# values like the current default of 3 -- ceil(3*0.67)=3 (no-op), floor=2.
_MAX_SAME_REGIME_FRACTION = 0.67

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
        # Lock for peak_prices dict to prevent concurrent modification issues
        # Using threading.Lock since check_trailing_stop is synchronous
        self._peak_prices_lock = threading.Lock()
        self._reserved_exposure = []  # list of (notional_amount, reserved_at) tuples
        # Transaction cost tracking for dynamic model
        self._realized_tx_costs: Dict[str, Dict[str, float]] = {}  # symbol -> {fee_bps, slippage_bps, spread_bps}
        # Reserved new-position slots: symbol -> reserved_at. Mirrors
        # _reserved_exposure's pattern (same lock, same TTL) but for
        # MAX_OPEN_POSITIONS instead of dollar exposure -- without this,
        # multiple concurrently-evaluated symbols in the same cycle all see
        # the same stale, cycle-start position count and can all pass the
        # naive position-count check simultaneously.
        self._reserved_new_position_symbols: Dict[str, datetime] = {}

    def get_transaction_costs(self, symbol: str) -> Dict[str, float]:
        """Get transaction cost estimates for a symbol.
        
        Uses dynamic model (recent realized costs) if enabled and available,
        otherwise falls back to static defaults from settings.
        
        Returns dict with: fee_bps, slippage_bps, spread_bps, total_bps
        """
        if settings.TX_COST_USE_DYNAMIC and symbol in self._realized_tx_costs:
            costs = self._realized_tx_costs[symbol]
            total = costs["fee_bps"] + costs["slippage_bps"] + costs["spread_bps"]
            return {**costs, "total_bps": total}
        
        # Fallback to static defaults
        fee = getattr(settings, "TX_COST_FEE_BPS", DEFAULT_TX_COSTS["fee_bps"])
        slippage = getattr(settings, "TX_COST_SLIPPAGE_BPS", DEFAULT_TX_COSTS["slippage_bps"])
        spread = getattr(settings, "TX_COST_SPREAD_BPS", DEFAULT_TX_COSTS["spread_bps"])
        return {"fee_bps": fee, "slippage_bps": slippage, "spread_bps": spread, "total_bps": fee + slippage + spread}

    def _get_max_portfolio_cap(self) -> float:
        """Return the effective portfolio exposure cap in USD.
        
        If MAX_PORTFOLIO_PCT is set (default 0.5), the cap is
        ACCOUNT_BASE * MAX_PORTFOLIO_PCT. Otherwise falls back to the
        static MAX_PORTFOLIO_VALUE.
        """
        pct = getattr(settings, "MAX_PORTFOLIO_PCT", None)
        if pct is not None and pct > 0:
            return float(settings.ACCOUNT_BASE) * float(pct)
        return float(settings.MAX_PORTFOLIO_VALUE)

    def record_fill_costs(self, symbol: str, fee_bps: float, slippage_bps: float, spread_bps: float = None) -> None:
        """Record realized transaction costs for dynamic model updates.
        
        Called after each order fill to update the dynamic cost model.
        Uses exponential moving average to blend old and new observations.
        """
        if symbol not in self._realized_tx_costs:
            # spread_bps isn't cleanly derivable from a single fill (no bid/ask
            # captured at execution time), and the only production caller
            # (bot.py) never supplies it. Default to the static estimate rather
            # than 0.0 so total_bps doesn't silently drop the spread cost once
            # dynamic tracking activates for a symbol -- a future caller that
            # does have a real spread reading can still override it here.
            default_spread = spread_bps if spread_bps is not None else getattr(settings, "TX_COST_SPREAD_BPS", DEFAULT_TX_COSTS["spread_bps"])
            self._realized_tx_costs[symbol] = {"fee_bps": fee_bps, "slippage_bps": slippage_bps, "spread_bps": default_spread}
        else:
            # EMA with alpha=0.3 (moderate adaptation)
            alpha = 0.3
            old = self._realized_tx_costs[symbol]
            old["fee_bps"] = (1 - alpha) * old["fee_bps"] + alpha * fee_bps
            old["slippage_bps"] = (1 - alpha) * old["slippage_bps"] + alpha * slippage_bps
            if spread_bps is not None:
                old["spread_bps"] = (1 - alpha) * old.get("spread_bps", 0.0) + alpha * spread_bps

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

            if not np.isfinite(equity) or equity <= 0:
                logger.critical(f"Corrupted equity read from exchange: {equity!r}. Refusing to trade until this clears.")
                return {
                    "status": "error",
                    "error": f"non-finite or non-positive equity: {equity!r}",
                    "action": "stand_aside",
                }

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

            # Check portfolio value cap (dynamic: percentage of account base if configured)
            max_portfolio_abs = self._get_max_portfolio_cap()
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

    def _get_drawdown_taper_multiplier(self, drawdown_pct: Optional[float]) -> float:
        """Additional risk-amount taper as equity drawdown approaches the killswitch.

        current_equity (see calculate_position_size) already shrinks
        risk_amount proportionally with equity. This tapers MORE
        aggressively as drawdown approaches MAX_DRAWDOWN_STOP -- a soft
        landing so the bot pulls back further before the hard killswitch
        trips, instead of trading at full (if proportionally-smaller) size
        right up until the moment it's forcibly liquidated.

        No taper (1.0x) below 50% of the way to the killswitch threshold;
        linear taper down to a 0.25x floor from 50% to 100%+ of the way
        there. None or a non-negative drawdown (at/above peak equity) is a
        no-op, matching the previous (untapered) behavior exactly.

        A non-finite drawdown_pct (NaN/inf) also no-ops rather than silently
        propagating NaN into risk_amount: `nan >= 0` is False in Python, so
        an un-guarded NaN would fall through the checks below instead of
        being caught by them, poisoning every downstream multiplication.
        No live caller can trigger this today (update_account_status already
        rejects non-finite equity before drawdown_pct is derived from it),
        but this method takes an Optional[float] from any caller, present
        or future, so it guards its own input rather than relying on that.
        """
        if drawdown_pct is None or not math.isfinite(drawdown_pct) or drawdown_pct >= 0:
            return 1.0
        max_drawdown = settings.MAX_DRAWDOWN_STOP  # negative, e.g. -10.0
        if max_drawdown >= 0 or not math.isfinite(max_drawdown):
            return 1.0
        pct_of_max = drawdown_pct / max_drawdown  # both negative -> positive ratio
        if pct_of_max <= 0.5:
            return 1.0
        floor = 0.25
        progress = min((pct_of_max - 0.5) / 0.5, 1.0)
        return 1.0 - progress * (1.0 - floor)

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        regime: str,
        atr: Optional[float] = None,
        confidence: float = 1.0,
        returns_matrix: Optional[Dict[str, np.ndarray]] = None,
        expected_return_pct: float = 0.0,
        current_equity: Optional[float] = None,
        drawdown_pct: Optional[float] = None,
    ) -> Tuple[float, str]:
        """Portfolio Optimization: Volatility Parity, Correlation VaR, and Cash Allocation.

        Now includes transaction cost model:
        - Round-trip costs = 2 * (fee + slippage + half_spread) in bps
        - Effective risk reduced by expected costs
        - Trades rejected if expected edge < TX_COST_MIN_EDGE_BPS

        `current_equity` should be passed in by the caller (from the same
        update_account_status() snapshot used for the exposure/position-limit
        checks) whenever available, so risk-per-trade scales down with actual
        capital during a drawdown instead of staying pinned to the account's
        historical peak_equity -- otherwise risk as a fraction of *current*
        equity silently grows the deeper a drawdown gets. Falls back to
        peak_equity (previous behavior) when not supplied, so existing
        callers (backtests, tests) are unaffected.

        `drawdown_pct` (from the same snapshot, if available) additionally
        tapers risk as the account approaches MAX_DRAWDOWN_STOP -- see
        _get_drawdown_taper_multiplier. None is a no-op.
        """
        try:
            # 1. Dynamic Cash Allocation (base risk scales with actual equity, not hardcoded base)
            if current_equity is not None and current_equity > 0:
                account_equity = current_equity
            else:
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

            # 2b. Additional taper as drawdown approaches the killswitch --
            # soft landing on top of the equity-proportional shrink above.
            taper_mult = self._get_drawdown_taper_multiplier(drawdown_pct)
            if taper_mult < 1.0:
                logger.info(
                    f"Drawdown taper active for {symbol}: {taper_mult:.2f}x "
                    f"(drawdown={drawdown_pct:.2f}%, killswitch at {settings.MAX_DRAWDOWN_STOP:.2f}%)"
                )
            risk_amount *= taper_mult

            # 3. Transaction Cost Model - Get costs and adjust effective risk
            tx_costs = self.get_transaction_costs(symbol)
            total_cost_bps = tx_costs["total_bps"]  # fee + slippage + spread (one-way)
            round_trip_cost_bps = 2.0 * total_cost_bps  # entry + exit
            
            # Expected return in bps (from strategy signal if provided, else from profit target)
            if expected_return_pct > 0:
                expected_edge_bps = expected_return_pct * 10000  # Convert to bps
            else:
                expected_edge_bps = settings.PROFIT_TARGET_PCT * 10000  # Use take-profit as proxy
            
            # Net edge after round-trip costs
            net_edge_bps = expected_edge_bps - round_trip_cost_bps
            
            # Check minimum edge threshold
            min_edge_bps = getattr(settings, "TX_COST_MIN_EDGE_BPS", 10.0)
            if net_edge_bps < min_edge_bps:
                logger.warning(
                    f"Trade rejected for {symbol}: net edge {net_edge_bps:.1f}bps < min {min_edge_bps:.1f}bps "
                    f"(expected={expected_edge_bps:.1f}, round_trip_cost={round_trip_cost_bps:.1f})"
                )
                return 0.0, f"rejected: insufficient edge after costs ({net_edge_bps:.1f}bps < {min_edge_bps:.1f}bps)"

            # 4. Adjust effective risk by expected transaction costs
            # The risk amount is reduced by the expected cost as a fraction of position
            cost_fraction = round_trip_cost_bps / 10000  # Convert bps to fraction
            effective_risk_amount = risk_amount * (1.0 - cost_fraction)
            effective_risk_amount = max(effective_risk_amount, risk_amount * 0.5)  # Floor at 50% of original

            # 5. Volatility Targeting (Equal risk contribution)
            # Use the actual stop-loss percentage distance for sizing so that
            # the position size matches the real SL trigger distance.
            stop_distance = current_price * settings.STOP_LOSS_PCT
            if atr is not None and atr > 0:
                atr_stop_distance = atr * getattr(settings, "ATR_STOP_MULTIPLIER", 2.0)
                # Use the smaller of ATR-based and %-based stop distances
                # so the tighter stop governs the actual risk.
                stop_distance = min(stop_distance, atr_stop_distance)

            position_size = effective_risk_amount / stop_distance

            # 6. Correlation Matrix & Portfolio VaR
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

            # 7. Maximum Sector/Direction Exposure (Hard Cap)
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

            logger.debug(
                f"Position sizing {symbol}: risk=${risk_amount:.2f} effective=${effective_risk_amount:.2f} "
                f"costs={total_cost_bps:.1f}bps round_trip={round_trip_cost_bps:.1f}bps "
                f"edge={expected_edge_bps:.1f}bps net={net_edge_bps:.1f}bps size={position_size:.6f}"
            )
            return position_size, "ok"

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return 0.0, f"error: {e}"


    def _get_trailing_params(self, regime: Optional[str]) -> Tuple[float, float]:
        """Scale TRAILING_ACTIVATION_PCT/TRAILING_DISTANCE_PCT by regime.

        Unrecognized or missing regime maps to (1.0, 1.0) -- exact match to
        the previous static behavior. Enforces the same invariant
        config.py validates statically for the base settings (distance must
        stay smaller than activation) since a multiplier combination could
        otherwise produce a stop that triggers immediately upon activation
        instead of trailing.
        """
        act_mult, dist_mult = _TRAILING_REGIME_MULTIPLIERS.get(regime, (1.0, 1.0))
        activation = settings.TRAILING_ACTIVATION_PCT * act_mult
        distance = settings.TRAILING_DISTANCE_PCT * dist_mult
        if distance >= activation:
            logger.warning(
                f"Trailing distance ({distance:.4f}) >= activation ({activation:.4f}) "
                f"for regime {regime!r} -- clamping distance"
            )
            distance = activation * 0.9
        return activation, distance

    def check_trailing_stop(self, symbol: str, current_price: float, avg_entry_price: float, qty: float, regime: Optional[str] = None) -> str:
        """
        Check if a trailing stop should be activated or triggered.
        Returns: 'close' if triggered, 'hold' otherwise.
        Handles both long and short positions.

        `regime` (if supplied) scales the activation/distance percentages
        via _TRAILING_REGIME_MULTIPLIERS -- e.g. wider in high_volatility to
        avoid noise-driven stopouts, tighter in low_volatility/sideways to
        lock in gains faster. None or an unrecognized regime is a no-op,
        identical to the previous unscaled behavior.
        """
        if not settings.TRAILING_STOP_ENABLED or avg_entry_price <= 0:
            return "hold"

        if not np.isfinite(current_price) or current_price <= 0:
            logger.warning(f"[{symbol}] check_trailing_stop got a non-finite price ({current_price!r}); skipping this check.")
            return "hold"

        activation_pct, distance_pct = self._get_trailing_params(regime)

        is_long = float(qty) > 0

        if is_long:
            unrealized_pct = (current_price - avg_entry_price) / avg_entry_price

            if unrealized_pct >= activation_pct:
                with self._peak_prices_lock:
                    if symbol not in self.peak_prices or current_price > self.peak_prices[symbol]:
                        self.peak_prices[symbol] = current_price

            with self._peak_prices_lock:
                if symbol in self.peak_prices:
                    peak = self.peak_prices[symbol]
                    drawdown_from_peak = (peak - current_price) / peak

                    if drawdown_from_peak >= distance_pct:
                        logger.info(f"Trailing stop triggered for {symbol}: Peak {peak:.2f}, Current {current_price:.2f}")
                        del self.peak_prices[symbol]
                        return "close"

        else:
            unrealized_pct = (avg_entry_price - current_price) / avg_entry_price

            if unrealized_pct >= activation_pct:
                with self._peak_prices_lock:
                    if symbol not in self.peak_prices or current_price < self.peak_prices[symbol]:
                        self.peak_prices[symbol] = current_price

            with self._peak_prices_lock:
                if symbol in self.peak_prices:
                    trough = self.peak_prices[symbol]
                    drawdown_from_trough = (current_price - trough) / trough

                    if drawdown_from_trough >= distance_pct:
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
        # If caller didn't provide current_exposure, fetch it now (outside lock
        # to allow concurrent fetches). We'll re-verify inside the lock.
        if current_exposure is None:
            status = await self.update_account_status()
            if status.get("status") != "risk_ok":
                return 0.0, status.get("reason", status.get("status", "risk_check_failed"))
            current_exposure = status["current_exposure"]

        async with self._exposure_lock:
            # Re-fetch exposure if it wasn't provided by caller, to close the
            # race window where another task could have reserved capacity
            # between our fetch and acquiring this lock.
            if current_exposure is None:
                status = await self.update_account_status()
                if status.get("status") != "risk_ok":
                    return 0.0, status.get("reason", status.get("status", "risk_check_failed"))
                current_exposure = status["current_exposure"]

            now = datetime.now(timezone.utc)
            # Prune stale reservations aggressively (30s TTL instead of 60s)
            # so failed/cancelled orders release their headroom faster.
            self._reserved_exposure = [(a, t) for a, t in self._reserved_exposure if (now - t).total_seconds() < 30]
            reserved_total = sum(a for a, _ in self._reserved_exposure)
            max_portfolio_abs = self._get_max_portfolio_cap()
            headroom = max_portfolio_abs - current_exposure - reserved_total
            if headroom < min_notional:
                logger.warning(f"Exposure reservation denied: current=${current_exposure:.2f} reserved=${reserved_total:.2f} headroom=${headroom:.2f} (below min ${min_notional:.2f}) cap=${max_portfolio_abs:.2f}")
                return 0.0, "max_portfolio_value_would_be_exceeded"
            approved = min(requested_notional, headroom)
            if approved < requested_notional:
                logger.info(f"Exposure reservation partially approved: requested=${requested_notional:.2f} approved=${approved:.2f} (headroom-limited)")
            self._reserved_exposure.append((approved, now))
            return approved, "ok"

    def release_reserved_exposure(self, notional: float) -> None:
        """Release a previously-reserved exposure amount back to the pool.

        Call this when an order fails, is cancelled, or the reserved amount
        was never actually used. Do NOT call this after a successful fill --
        reproduced with a real asyncio race test that doing so reopens the
        exact race check_and_reserve_exposure exists to prevent: a
        fast-filling symbol's release mid-cycle lets a slower-evaluated
        sibling (still reading the same stale `current_exposure` snapshot)
        reserve against it too, and the two together can exceed the
        portfolio exposure cap within one cycle (measured: $13.4k landed
        exposure against a $10k cap, $8k already-stale, 3 concurrent $1.8k
        reservations racing a release-on-success). A successful reservation
        is instead left to expire via the 30s TTL prune in
        check_and_reserve_exposure -- the next cycle's fresh
        update_account_status() reflects the real fill once that clears it.
        """
        if notional <= 0:
            return
        # Remove the most recent matching reservation (LIFO)
        for i in range(len(self._reserved_exposure) - 1, -1, -1):
            amt, _ = self._reserved_exposure[i]
            if abs(amt - notional) < 0.01:
                del self._reserved_exposure[i]
                logger.debug(f"Released reserved exposure: ${notional:.2f}")
                return
        # Fallback: remove the largest reservation
        if self._reserved_exposure:
            idx = max(range(len(self._reserved_exposure)), key=lambda i: self._reserved_exposure[i][0])
            del self._reserved_exposure[idx]
            logger.debug(f"Released largest reserved exposure: ${notional:.2f}")

    async def reserve_position_slot(
        self,
        symbol: str,
        open_position_count: int,
        same_regime_open_count: int = 0,
    ) -> Tuple[bool, str]:
        """Atomically check and reserve a new-position slot against MAX_OPEN_POSITIONS.

        Call this once per symbol, only for a genuinely NEW entry (not a
        scale-in add to an existing position), immediately before placing
        the order -- mirrors check_and_reserve_exposure's reservation
        pattern for the same reason: `open_position_count` is a stale,
        cycle-start snapshot shared by every concurrently-evaluated symbol,
        so without a reservation, multiple symbols could all see the same
        under-cap count and all pass simultaneously.

        Unlike check_and_reserve_exposure/release_reserved_exposure, do NOT
        release this on a successful order -- only call release_position_slot()
        on a failure to place/confirm the order. Releasing on success was
        tried first and reproducibly reopens the same race this exists to
        prevent: a fast-filling symbol's release mid-cycle lets a
        slower-evaluated sibling (still reading the same stale
        open_position_count) reserve a slot too, collectively exceeding
        MAX_OPEN_POSITIONS within one cycle. A successful reservation is
        instead left to expire via the 30s TTL below -- a symbol briefly
        looks "reserved" to its siblings for up to 30s after it actually
        fills, which only costs a little new-entry throughput, never a cap
        breach.

        `same_regime_open_count` (default 0, a no-op) is the number of
        currently open positions classified in the same regime as `symbol` --
        see _MAX_SAME_REGIME_FRACTION. Caps how concentrated the portfolio
        can get in one regime cluster, independent of the overall count cap.
        """
        async with self._exposure_lock:
            now = datetime.now(timezone.utc)
            # Prune stale reservations (30s TTL, matching _reserved_exposure)
            self._reserved_new_position_symbols = {
                s: t for s, t in self._reserved_new_position_symbols.items()
                if (now - t).total_seconds() < 30
            }
            if symbol in self._reserved_new_position_symbols:
                return False, "position_slot_already_reserved"
            reserved_count = len(self._reserved_new_position_symbols)
            if open_position_count + reserved_count >= settings.MAX_OPEN_POSITIONS:
                logger.warning(
                    f"Position slot reservation denied for {symbol}: open={open_position_count} "
                    f"reserved={reserved_count} cap={settings.MAX_OPEN_POSITIONS}"
                )
                return False, "max_open_positions_would_be_exceeded"

            same_regime_cap = max(1, math.floor(settings.MAX_OPEN_POSITIONS * _MAX_SAME_REGIME_FRACTION))
            if same_regime_open_count >= same_regime_cap:
                logger.warning(
                    f"Position slot reservation denied for {symbol}: {same_regime_open_count} open "
                    f"position(s) already share its regime (cluster cap {same_regime_cap})"
                )
                return False, "same_regime_cluster_cap_reached"

            self._reserved_new_position_symbols[symbol] = now
            return True, "ok"

    def release_position_slot(self, symbol: str) -> None:
        """Release a previously-reserved new-position slot."""
        self._reserved_new_position_symbols.pop(symbol, None)

    async def reduce_exposure_to_cap(self) -> Dict[str, Any]:
        """Close positions, worst unrealized P&L first, until exposure is
        back under the cap. Targeted/incremental, unlike liquidate_all_positions."""
        try:
            positions = await self.exchange.get_positions()
            max_portfolio_abs = self._get_max_portfolio_cap()
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
                client_order_id = f"emergency_{symbol}_{side}_{qty_abs}_{int(time.time())}"
                order_result = await self.exchange.create_order(symbol=symbol, qty=qty_abs, side=side, type="market", client_order_id=client_order_id)
                filled_price = order_result.get("filled_avg_price", 0.0)
                filled_qty = order_result.get("filled_qty", qty_abs)
                actual_value = filled_price * filled_qty if filled_price > 0 else market_value
                current_exposure -= actual_value
                results.append({"symbol": symbol, "closed_qty": qty, "order": order_result})
                logger.warning(f"Closed {symbol} to reduce exposure (was ${market_value:.2f}, actual=${actual_value:.2f})")
                # Clear stale peak-price tracking for this symbol so a future
                # re-entry doesn't inherit a leftover peak from this closed
                # position (same reasoning as _record_committee_outcome's
                # cleanup for normal closes -- this path bypasses that).
                self.peak_prices.pop(symbol, None)
                self._reserved_new_position_symbols.pop(symbol, None)
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

                client_order_id = f"emergency_{symbol}_{side}_{qty_abs}_{int(time.time())}"
                order_result = await self.exchange.create_order(
                    symbol=symbol,
                    qty=qty_abs,
                    side=side,
                    type="market",
                    client_order_id=client_order_id,
                )

                results.append({
                    "symbol": symbol,
                    "original_qty": qty,
                    "close_order": order_result
                })
                # Clear stale peak-price tracking for this symbol so a future
                # re-entry doesn't inherit a leftover peak from this closed
                # position (same reasoning as _record_committee_outcome's
                # cleanup for normal closes -- this path bypasses that).
                self.peak_prices.pop(symbol, None)
                self._reserved_new_position_symbols.pop(symbol, None)

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

    def cleanup_stale_state(self, max_age_seconds: float = 3600, active_symbols: Optional[list] = None) -> Dict[str, int]:
        """Clean up stale state entries to prevent memory leaks.
        
        Args:
            max_age_seconds: Maximum age of peak_prices entries before cleanup
            active_symbols: List of currently active trading symbols (from settings)
            
        Returns:
            Dict with counts of cleaned entries per category
        """
        now = datetime.now(timezone.utc)
        active_symbols = set(active_symbols) if active_symbols else set()
        cleaned = {
            "peak_prices": 0,
            "realized_tx_costs": 0,
            "reserved_exposure": 0,
            "reserved_position_slots": 0,
        }
        
        # Clean peak_prices for symbols not in active trading or very old
        # We can't easily track age, so we clean symbols not in active_symbols
        stale_peaks = [k for k in self.peak_prices.keys() if k not in active_symbols]
        for k in stale_peaks:
            del self.peak_prices[k]
        cleaned["peak_prices"] = len(stale_peaks)
        
        # Clean realized_tx_costs for symbols not in active trading
        stale_costs = [k for k in self._realized_tx_costs.keys() if k not in active_symbols]
        for k in stale_costs:
            del self._realized_tx_costs[k]
        cleaned["realized_tx_costs"] = len(stale_costs)
        
        # Clean reserved_exposure (already has 30s TTL in check_and_reserve_exposure)
        # This is an additional safety net
        old_reserved = len(self._reserved_exposure)
        self._reserved_exposure = [
            (a, t) for a, t in self._reserved_exposure 
            if (now - t).total_seconds() < max(30, max_age_seconds)
        ]
        cleaned["reserved_exposure"] = old_reserved - len(self._reserved_exposure)
        
        # Clean reserved_new_position_symbols for inactive symbols
        stale_slots = [k for k in self._reserved_new_position_symbols.keys() if k not in active_symbols]
        for k in stale_slots:
            del self._reserved_new_position_symbols[k]
        cleaned["reserved_position_slots"] = len(stale_slots)
        
        if any(v > 0 for v in cleaned.values()):
            logger.debug(f"RiskManager cleaned stale state: {cleaned}")
        
        return cleaned