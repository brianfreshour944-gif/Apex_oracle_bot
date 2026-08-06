"""Main trading bot implementation."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from src.config import settings
from src.logging_config import get_logger
from src.db import init_db
from src.exchange import AlpacaExchange
from src.api import start_fastapi_server_async
from src.strategies import TradingStrategy
from src.risk import RiskManager
from src.telegram_alerts import send_telegram_alert

logger = get_logger("bot")


async def _record_committee_outcome(symbol: str, exit_price: float, exit_reason: Optional[str] = None) -> None:
    """On position exit, close the open decision snapshot and update the learner.

    Fully fail-safe: realized-PnL bookkeeping for the adaptive layer must never
    interfere with trading. risk.py stays authoritative for the exit itself.
    """
    try:
        from datetime import datetime, timezone
        from src.db import get_open_snapshot, close_decision_snapshot
        from src.committee.committee import get_meta_learner
        from src.metrics import update_adaptive_metrics, alert_weight_change

        snap = await asyncio.to_thread(get_open_snapshot, symbol)
        if not snap:
            return

        entry_price = float(snap.get("entry_price", 0.0))
        qty = float(snap.get("qty", 0.0))
        action = snap.get("final_action", "buy")
        if entry_price <= 0 or qty == 0:
            return

        if action == "buy":
            realized_pnl = (exit_price - entry_price) * qty
            return_pct = (exit_price - entry_price) / entry_price * 100.0
        else:  # sell / short
            realized_pnl = (entry_price - exit_price) * qty
            return_pct = (entry_price - exit_price) / entry_price * 100.0

        holding_sec = 0.0
        created = snap.get("created_at")
        if created:
            try:
                started = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                holding_sec = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
            except Exception:
                pass

        max_fav_pct = 0.0
        max_adv_pct = 0.0
        global strategy
        if strategy is not None and hasattr(strategy, '_trailing_peaks') and symbol in strategy._trailing_peaks:
            peak = strategy._trailing_peaks[symbol]
            if action == "buy":
                max_fav_pct = (peak - entry_price) / entry_price * 100.0 if peak > entry_price else 0.0
            else:
                max_fav_pct = (entry_price - peak) / entry_price * 100.0 if peak < entry_price else 0.0

        await asyncio.to_thread(
            close_decision_snapshot,
            snap["decision_id"],
            realized_pnl=realized_pnl,
            return_pct=return_pct,
            holding_period_sec=holding_sec,
            exit_reason=exit_reason,
            max_favorable_pct=max_fav_pct,
            max_adverse_pct=max_adv_pct,
        )

        # Clear peak price tracking for this symbol on any position close
        # to prevent stale peak prices from affecting future positions
        if risk_manager is not None:
            risk_manager.peak_prices.pop(symbol, None)

        learner = get_meta_learner()
        if learner is not None:
            report = learner.update(
                {
                    "regime": snap.get("regime", "default"),
                    "final_action": action,
                    "brain_votes": snap.get("brain_votes", {}),
                },
                {"net_pnl": realized_pnl, "return_pct": return_pct},
            )
            update_adaptive_metrics(learner.snapshot())
            if report.material_change:
                await alert_weight_change(
                    report.regime, report.old_weights, report.new_weights, learner.sample_count
                )
                
        # Update strategy learner
        selected_strategy = snap.get("feature_snapshot", {}).get("selected_strategy")
        if selected_strategy:
            from src.strategy_selector import record_strategy_outcome
            record_strategy_outcome(
                regime=snap.get("regime", "default"),
                strategy_name=selected_strategy,
                action=action,
                pnl=realized_pnl,
                return_pct=return_pct
            )

        # Ã¢â€�?â‚¬Ã¢â€�?â‚¬ Trigger Causal Attribution Ã¢â€�?â‚¬Ã¢â€�?â‚¬
        async def _run_attribution():
            try:
                from src.attribution import analyze_closed_trade
                res = await analyze_closed_trade(snap["decision_id"])
                if res:
                    logger.info(f"Ã°Å¸Â§Â  Causal Attribution for {symbol} ({action}):")
                    logger.info(f"   Success Factors: {res.get('success_factors')}")
                    logger.info(f"   Key Signals: {res.get('key_signals')}")
                    logger.info(f"   MVP Member: {res.get('mvp_member')}")
                    logger.info(f"   Robustness: {res.get('robustness_score')}")
                    logger.info(f"   Lessons: {res.get('lessons_learned')}")
                    
                    msg = (
                        f"Ã°Å¸Â§Â  <b>Trade Attribution: {symbol} ({action})</b>\n"
                        f"PnL: ${realized_pnl:.2f}\n"
                        f"<i>{res.get('success_factors')}</i>\n\n"
                        f"<b>MVP:</b> {res.get('mvp_member')}\n"
                        f"<b>Robustness:</b> {res.get('robustness_score')}\n"
                        f"<b>Lessons:</b> {res.get('lessons_learned')}"
                    )
                    await send_telegram_alert(msg)
            except Exception as e:
                logger.error(f"Attribution engine failed: {e}")
                
        asyncio.create_task(_run_attribution())

    except Exception as e:
        logger.error(f"Adaptive outcome recording failed for {symbol} (non-fatal): {e}")


# Global state
ex: Optional[AlpacaExchange] = None
strategy: Optional[TradingStrategy] = None
risk_manager: Optional[RiskManager] = None
latest_scan_results: Dict[str, dict] = {}
scan_cycle_count: int = 0
# Per-symbol cooldown: timestamp (time.time()) before which a new entry on
# that symbol is blocked, set right after closing a position. See
# settings.COOLDOWN_SECONDS_BUY for why this exists.
cooldowns: Dict[str, float] = {}
_symbol_locks: Dict[str, asyncio.Lock] = {}

import json
import os as _os

_REGIME_FLAG_PATH = "data/regime_flag.txt"
_regime_flag_cache: Dict[str, Any] = {}
_regime_flag_cache_mtime: float = -1.0

def read_regime_flag():
    """Read the regime flag file, cached and only re-read when the file's
    mtime changes. This is called on every symbol evaluation every cycle
    (measured ~1.5ms/call for open()+json.load() when uncached, fully
    blocking the event loop each time) even though the file is only written
    hours/days apart by the weekly analyzer / regime-switch scripts."""
    global _regime_flag_cache, _regime_flag_cache_mtime
    default = {
        "pause_grok": False,
        "pause_oracle": False,
        "grok_multiplier": 1.0,
        "oracle_multiplier": 1.0,
        "regime": "normal"
    }
    try:
        mtime = _os.path.getmtime(_REGIME_FLAG_PATH)
    except OSError:
        return default
    if mtime == _regime_flag_cache_mtime and _regime_flag_cache:
        return _regime_flag_cache
    try:
        with open(_REGIME_FLAG_PATH, "r") as f:
            data = json.load(f)
        _regime_flag_cache = data
        _regime_flag_cache_mtime = mtime
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return default

_BANNED_SYMBOLS_PATH = _os.path.join(_os.path.dirname(__file__), '..', 'data', 'banned_symbols.json')
_banned_symbols_cache: set = set()
_banned_symbols_cache_mtime: float = -1.0

def get_banned_symbols():
    """Read the banned symbols list generated by the weekly analyzer, cached
    and only re-read when the file's mtime changes (see read_regime_flag)."""
    global _banned_symbols_cache, _banned_symbols_cache_mtime
    try:
        mtime = _os.path.getmtime(_BANNED_SYMBOLS_PATH)
    except OSError:
        return set()
    if mtime == _banned_symbols_cache_mtime:
        return _banned_symbols_cache
    try:
        with open(_BANNED_SYMBOLS_PATH, 'r') as f:
            bans = json.load(f)
        _banned_symbols_cache = {b["symbol"] for b in bans}
        _banned_symbols_cache_mtime = mtime
        return _banned_symbols_cache
    except Exception as e:
        logger.warning(f"Failed to read banned symbols: {e}")
        return _banned_symbols_cache

async def process_signal_for_symbol(symbol: str, current_price: float, risk_manager: RiskManager, strategy: TradingStrategy, ex: AlpacaExchange, regime_flag: dict = None, banned_symbols: set = None) -> None:
    """Processes signal for a single symbol asynchronously."""
    # Get or create lock for this symbol
    lock = _symbol_locks.setdefault(symbol, asyncio.Lock())
    async with lock:
        try:
            # Get current positions
            positions = await ex.get_positions()
            position_dict = {p["symbol"].replace("/", ""): p for p in positions}
            current_position = position_dict.get(symbol.replace("/", ""))
    
            # Cooldown check: block a fresh entry right after this symbol closed a
            # position, but never block evaluation of an EXISTING position's exit
            # logic (a position we're already holding must always be free to close).
            if not current_position:
                now_ts = time.time()
                cooldown_until = cooldowns.get(symbol, 0)
                if now_ts < cooldown_until:
                    remaining = int(cooldown_until - now_ts)
                    logger.debug(f"[{symbol}] On entry cooldown, {remaining}s remaining -- skipping")
                    return
    
            # Trailing Stop Check
            if current_position:
                avg_entry_price = float(current_position.get("avg_entry_price", 0))
                qty = float(current_position.get("qty", 0))
                trailing_action = risk_manager.check_trailing_stop(symbol, current_price, avg_entry_price, qty)
                
                if trailing_action == "close":
                    side = "sell" if qty > 0 else "buy"
                    qty_abs = abs(qty)
                    order_result = await ex.create_order(
                        symbol=symbol,
                        qty=qty_abs,
                        side=side,
                        type="market"
                    )
                    logger.info(f"Trailing Stop Executed: {symbol}")
                    await send_telegram_alert(f"Ã°Å¸Å¡Â¨ <b>Trailing Stop Triggered</b>\nSymbol: {symbol}\nClosed {qty} @ ${current_price:.2f}")
                    cooldowns[symbol] = time.time() + settings.COOLDOWN_SECONDS_BUY
                    await _record_committee_outcome(symbol, current_price, exit_reason="trailing_stop")
                    return  # Skip standard signals
    
            # Generate trading signal
            signal = await strategy.generate_trading_signal(
                symbol,
                current_price,
                current_position
            )
    
            logger.debug("[SCAN] Signal for {symbol} @ ${current_price:.2f}: {action} (regime: {regime}, RSI: {rsi:.2f})",
                         symbol=symbol, current_price=current_price, action=signal["action"],
                         regime=signal["regime"], rsi=signal["rsi"])
    
            # Bypass committee for hard exits (SL/TP)
            if signal["action"] == "close":
                pass # proceed directly to close logic below
            else:
                # Ã¢â€�?â‚¬Ã¢â€�?â‚¬ 5-BRAIN ENSEMBLE COMMITTEE EVALUATION Ã¢â€�?â‚¬Ã¢â€�?â‚¬
                from src.committee.committee import run_committee
                committee_result = await run_committee(symbol, current_price, signal)
    
                # Ã¢â€�?â‚¬Ã¢â€�?â‚¬ BUILD REGIME DASHBOARD Ã¢â€�?â‚¬Ã¢â€�?â‚¬
                dashboard = []
                dashboard.append("==============================")
                dashboard.append(f"{symbol}")
                dashboard.append("")
                
                regime_str = signal.get("regime", "UNKNOWN").upper()
                hurst = signal.get("features", {}).get("hurst", 0.0)
                atr = signal.get("atr", 0.0)
                atr_pct = (atr / current_price * 100) if current_price > 0 else 0.0
                
                dashboard.append(f"Regime........ {regime_str}")
                dashboard.append(f"Hurst......... {hurst:.2f}")
                dashboard.append(f"ATR........... {atr_pct:.1f}%")
                dashboard.append("")
                
                for v in committee_result.votes:
                    action_str = v.action.upper()
                    if action_str == "STAND_ASIDE":
                        action_str = "PASS"
                    elif action_str not in ["PASS", "HOLD", "SKIP"]:
                        action_str = f"{action_str} {v.confidence:.2f}"
                    dashboard.append(f"{v.name.capitalize():<14} {action_str}")
                     
                dashboard.append("")
                
                committee_action = committee_result.action.upper()
                if committee_result.vetoed:
                    committee_action = "VETO"
                elif committee_action == "STAND_ASIDE":
                    committee_action = "PASS"
                     
                dashboard.append(f"Committee...... {committee_action}")
                dashboard.append("")
    
                if committee_result.vetoed:
                    dashboard.append("FINAL.......... NO TRADE")
                    dashboard.append(f"Reason......... {committee_result.veto_reason}")
                    dashboard.append("==============================")
                    print("\n".join(dashboard), flush=True)
                    latest_scan_results[symbol] = {"score": committee_result.score, "action": "VETO", "price": current_price}
                    return
    
                if committee_result.action in ["stand_aside", "skip", "hold"]:
                    dashboard.append("FINAL.......... NO TRADE")
                    dashboard.append("Reason......... Committee Consensus")
                    dashboard.append("==============================")
                    print("\n".join(dashboard), flush=True)
                    latest_scan_results[symbol] = {"score": committee_result.score, "action": committee_result.action.upper(), "price": current_price}
                    return
    
                latest_scan_results[symbol] = {"score": committee_result.score, "action": committee_result.action.upper(), "price": current_price}
    
                # Override original signal action & confidence with committee's consensus decision
                signal["action"] = committee_result.action
                signal["confidence"] = committee_result.score
    
            if signal["action"] in ["buy", "sell"]:
                # Ã¢â€�?â‚¬Ã¢â€�?â‚¬ REGIME SWITCH CHECK (Only for ENTRY, not EXIT) Ã¢â€�?â‚¬Ã¢â€�?â‚¬
                if signal["action"] == "buy":
                    if regime_flag is None:
                        regime_flag = read_regime_flag()
                    if regime_flag.get("pause_oracle", False):
                        dashboard.append("Risk........... VETO")
                        dashboard.append("FINAL.......... NO TRADE")
                        dashboard.append("Reason......... Oracle Paused (Regime Switch)")
                        dashboard.append("==============================")
                        print("\n".join(dashboard), flush=True)
                        return
    
                # Ã¢â€�?â‚¬Ã¢â€�?â‚¬ BANNED SYMBOLS CHECK Ã¢â€�?â‚¬Ã¢â€�?â‚¬
                if banned_symbols is None:
                    banned = get_banned_symbols()
                else:
                    banned = banned_symbols
                if symbol in banned:
                    dashboard.append("Risk........... VETO")
                    dashboard.append("FINAL.......... NO TRADE")
                    dashboard.append("Reason......... Symbol Banned")
                    dashboard.append("==============================")
                    print("\n".join(dashboard), flush=True)
                    return
    
                # Check position limit before entering
                # Reuse the `positions` list already fetched at the top of this
                # function (bot.py:213) instead of letting update_account_status()
                # fetch it again -- measured: this redundancy previously cost 2-3x
                # get_positions()/get_account() calls per symbol per cycle.
                risk_status = await risk_manager.update_account_status(positions=positions)
                if risk_status["status"] == "position_limit_exceeded":
                    dashboard.append("Risk........... VETO")
                    dashboard.append("FINAL.......... NO TRADE")
                    dashboard.append("Reason......... Position limit reached")
                    dashboard.append("==============================")
                    print("\n".join(dashboard), flush=True)
                    return
                if risk_status["status"] == "exposure_limit_exceeded":
                    dashboard.append("Risk........... VETO")
                    dashboard.append("FINAL.......... NO TRADE")
                    dashboard.append("Reason......... Exposure cap reached")
                    dashboard.append("==============================")
                    print("\n".join(dashboard), flush=True)
                    return
    
                # Calculate position size
                position_size, sizing_status = risk_manager.calculate_position_size(
                    symbol,
                    current_price,
                    signal["regime"],
                    atr=signal.get("atr"),
                    confidence=signal.get("confidence", 1.0)
                )
    
                if sizing_status != "ok":
                    dashboard.append("Risk........... VETO")
                    dashboard.append("FINAL.......... NO TRADE")
                    dashboard.append(f"Reason......... {sizing_status}")
                    dashboard.append("==============================")
                    print("\n".join(dashboard), flush=True)
                    return
                    
                dashboard.append("Risk........... PASS")
                dashboard.append("")
                dashboard.append(f"FINAL.......... EXECUTE {signal['action'].upper()}")
                dashboard.append("==============================")
                print("\n".join(dashboard), flush=True)
                    
                # Apply Regime Switch Multiplier if buying
                if signal["action"] == "buy":
                    multiplier = regime_flag.get("oracle_multiplier", 1.0)
                    position_size = position_size * multiplier
    
                # Apply Committee Confidence Sizing Multiplier (Higher confidence = Larger trade size)
                committee_mult = getattr(committee_result, "size_multiplier", 1.0)
                position_size = position_size * committee_mult
                position_size = round(position_size, 6)
                logger.info(f"Ã°Å¸â€œÅ  Applied Committee Sizing Multiplier ({committee_mult:.2f}x based on score {committee_result.score:.2f}) Ã¢â€ â€™ Final Qty: {position_size}")
    
                # Atomically check and reserve exposure to prevent a race condition
                # where concurrently-evaluated symbols could each pass an individual
                # exposure check before any of their sibling orders have settled.
                # This may approve a SMALLER notional than requested (capped to
                # remaining headroom) rather than an all-or-nothing veto, so
                # available capacity actually gets used instead of sitting idle.
                notional = current_price * position_size
                # Pass the current_exposure already computed by update_account_status()
                # above so this doesn't re-fetch account/positions a third time AND
                # so the exposure lock's critical section is pure in-memory
                # arithmetic instead of holding the lock across a network round
                # trip (measured: this previously serialized ~2.85s across 15
                # concurrently-evaluated symbols in the same cycle).
                approved_notional, reserve_reason = await risk_manager.check_and_reserve_exposure(
                    notional, current_exposure=risk_status.get("current_exposure")
                )
                if approved_notional <= 0:
                    logger.warning(f"[{symbol}] Order vetoed: {reserve_reason}")
                    return
    
                if approved_notional < notional:
                    scale = approved_notional / notional
                    original_size = position_size
                    position_size = round(position_size * scale, 6)
                    logger.info(f"[{symbol}] Position size reduced to fit exposure headroom: {position_size} (was {original_size}, ${approved_notional:.2f} of ${notional:.2f} requested)")
                    if position_size <= 0:
                        logger.warning(f"[{symbol}] Order vetoed: scaled position size rounded to zero")
                        return
    
                # Place order
                order_result = await ex.create_order(
                    symbol=symbol,
                    qty=position_size,
                    side=signal["action"],
                    type="market"
                )
    
                filled_price = order_result.get("filled_avg_price", 0.0)
                commission = order_result.get("commission", 0.0)
                if filled_price > 0:
                    expected_price = current_price
                    slippage_bps = abs(filled_price - expected_price) / expected_price * 10000
                    if slippage_bps > 1.0:
                        logger.warning(f"[SLIPPAGE] {symbol} {signal['action']}: expected=${expected_price:.2f} actual=${filled_price:.2f} slippage={slippage_bps:.1f}bps commission=${commission:.4f}")
                    else:
                        logger.info(f"[FILL] {symbol} {signal['action']}: filled=${filled_price:.2f} commission=${commission:.4f}")
                    if commission > 0:
                        logger.info(f"[FEE] {symbol} {signal['action']}: commission=${commission:.4f}")

                logger.info(f"[TRADE] Order executed: {signal['action'].upper()} {position_size} {symbol} @ ${current_price:.2f}")
                logger.debug(f"[TRADE] Order result: {order_result}")
                await send_telegram_alert(f"Ã°Å¸â€œË† <b>Order Executed</b>\nSymbol: {symbol}\nAction: {signal['action'].upper()}\nQty: {position_size}\nPrice: ${current_price:.2f}")
    
                # Persist committee decision snapshot for the adaptive meta-learner
                # (fail-safe; closed out with realized PnL when the position exits).
                try:
                    from src.db import save_decision_snapshot
                    await asyncio.to_thread(
                        save_decision_snapshot,
                        decision_id=committee_result.decision_id,
                        symbol=symbol,
                        regime=signal.get("regime", "default"),
                        final_action=signal["action"],
                        confidence=committee_result.score,
                        size_multiplier=getattr(committee_result, "size_multiplier", 1.0),
                        entry_price=current_price,
                        qty=position_size,
                        brain_votes={v.name: v.action for v in committee_result.votes},
                        feature_snapshot_json=json.dumps({
                            "atr": signal.get("atr"),
                            "rsi": signal.get("rsi"),
                            "macd": signal.get("macd"),
                            "selected_strategy": signal.get("selected_strategy"),
                        }),
                        causal_reasoning_json=json.dumps({
                            v.name: getattr(v, "causal_reasoning", None) 
                            for v in committee_result.votes if getattr(v, "causal_reasoning", None)
                        })
                    )
                except Exception as db_e:
                    logger.warning(f"Decision snapshot persist failed for {symbol} (non-fatal): {db_e}")
    
            elif signal["action"] == "close" and current_position:
    
                # Close existing position
                qty = float(current_position["qty"])
                side = "sell" if qty > 0 else "buy"
                qty_abs = abs(qty)
    
                order_result = await ex.create_order(
                    symbol=symbol,
                    qty=qty_abs,
                    side=side,
                    type="market"
                )
    
                filled_price = order_result.get("filled_avg_price", 0.0)
                commission = order_result.get("commission", 0.0)
                if filled_price > 0:
                    expected_price = current_price
                    slippage_bps = abs(filled_price - expected_price) / expected_price * 10000
                    if slippage_bps > 1.0:
                        logger.warning(f"[SLIPPAGE] {symbol} close: expected=${expected_price:.2f} actual=${filled_price:.2f} slippage={slippage_bps:.1f}bps commission=${commission:.4f}")
                    else:
                        logger.info(f"[FILL] {symbol} close: filled=${filled_price:.2f} commission=${commission:.4f}")
                    if commission > 0:
                        logger.info(f"[FEE] {symbol} close: commission=${commission:.4f}")

                logger.info(f"[TRADE] Position closed: {symbol} (was {qty}) - reason: {signal.get('reason', 'unknown')}")
                logger.debug(f"[TRADE] Close order result: {order_result}")
                await send_telegram_alert(f"Ã°Å¸â€œâ€° <b>Position Closed</b>\nSymbol: {symbol}\nReason: {signal.get('reason', 'unknown')}")
                cooldowns[symbol] = time.time() + settings.COOLDOWN_SECONDS_BUY
                await _record_committee_outcome(symbol, current_price, exit_reason=signal.get('reason', 'unknown'))
    
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    
async def scan_heartbeat_loop() -> None:
    """Background task to print periodic scan summary."""
    global scan_cycle_count, latest_scan_results
    while True:
        try:
            await asyncio.sleep(settings.LOOP_INTERVAL_SEC)
            scan_cycle_count += 1
            
            if not latest_scan_results:
                continue
                
            trades_this_cycle = False
            out = []
            out.append("========================================")
            out.append(f"Cycle {scan_cycle_count}")
            out.append(f"{len(latest_scan_results)} Symbols")
            out.append("Scanning...")
            out.append("========================================")
            
            for sym, data in latest_scan_results.items():
                action = data.get("action", "UNKNOWN")
                score = data.get("score", 0.0)
                out.append(f"{sym:<10} Score {score:.3f} {action}")
                if action in ["BUY", "SELL"]:
                    trades_this_cycle = True
            
            if not trades_this_cycle:
                out.append("\nNo trades this cycle.")
            
            # Use raw print to format nicely as a block without structlog prefix wrapping
            print("\n" + "\n".join(out) + "\n", flush=True)
            
        except Exception as e:
            logger.error(f"[HEARTBEAT] Error printing heartbeat: {e}")
            await asyncio.sleep(10)

async def monitor_killswitch(risk_manager: RiskManager) -> None:
    """Background task to monitor killswitch continuously."""
    import os
    while True:
        try:
            if await risk_manager.check_killswitch_conditions():
                logger.critical("KILLSWITCH ACTIVATED - Liquidating all positions")
                await send_telegram_alert(f"Ã°Å¸â€™â‚¬ <b>KILLSWITCH ACTIVATED</b>\nLiquidating all positions immediately.")
                await risk_manager.liquidate_all_positions()
                os._exit(1)

            exposure_status = await risk_manager.update_account_status()
            if exposure_status.get("status") == "exposure_limit_exceeded":
                logger.warning("Exposure cap breached - reducing")
                await risk_manager.reduce_exposure_to_cap()
            await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"Killswitch monitor error: {e}")
            await asyncio.sleep(10)


async def run_periodic_analyzer() -> None:
    """Background task to run the analyzer script periodically."""
    import sys
    import os
    script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'weekly_analyzer.py')
    
    while True:
        try:
            logger.info("Running periodic analyzer script...")
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Analyzer completed successfully:\n{stdout.decode().strip()}")
            else:
                logger.error(f"Analyzer failed with code {process.returncode}:\n{stderr.decode().strip()}")
                
        except Exception as e:
            logger.error(f"Error running periodic analyzer: {e}")
            
        # Run every 12 hours
        await asyncio.sleep(12 * 3600)

async def run_periodic_automl() -> None:
    """Background task to run the AutoML pipeline every Saturday night."""
    import sys
    import os
    from datetime import datetime, timedelta
    
    script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'automl_pipeline.py')
    
    while True:
        try:
            now = datetime.now()
            # Calculate days until Saturday (5 = Saturday)
            days_ahead = 5 - now.weekday()
            if days_ahead < 0 or (days_ahead == 0 and now.hour >= 2):
                days_ahead += 7
            
            # Target 2 AM on Saturday
            target_time = now + timedelta(days=days_ahead)
            target_time = target_time.replace(hour=2, minute=0, second=0, microsecond=0)
            
            sleep_seconds = (target_time - now).total_seconds()
            logger.info(f"AutoML pipeline scheduled for {target_time} (in {sleep_seconds/3600:.1f} hours)")
            
            await asyncio.sleep(sleep_seconds)
            
            logger.info("Running AutoML pipeline...")
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"AutoML pipeline completed successfully:\n{stdout.decode().strip()}")
            else:
                logger.error(f"AutoML pipeline failed with code {process.returncode}:\n{stderr.decode().strip()}")
                
        except Exception as e:
            logger.error(f"Error running AutoML pipeline: {e}")
            await asyncio.sleep(3600)

async def run_periodic_cull() -> None:
    """Background task to run the Evolution Cull on the 1st of every month."""
    import sys
    import os
    from datetime import datetime
    
    script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'evolution_cull.py')
    
    while True:
        try:
            now = datetime.now()
            # Calculate time until 1st of next month at 4 AM
            if now.month == 12:
                target_month = 1
                target_year = now.year + 1
            else:
                target_month = now.month + 1
                target_year = now.year
                
            target_time = datetime(target_year, target_month, 1, 4, 0, 0)
            
            sleep_seconds = (target_time - now).total_seconds()
            logger.info(f"Evolution Cull scheduled for {target_time} (in {sleep_seconds/86400:.1f} days)")
            
            await asyncio.sleep(sleep_seconds)
            
            logger.info("Running Evolution Cull pipeline...")
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Evolution Cull completed successfully:\n{stdout.decode().strip()}")
            else:
                logger.error(f"Evolution Cull failed with code {process.returncode}:\n{stderr.decode().strip()}")
                
        except Exception as e:
            logger.error(f"Error running Evolution Cull: {e}")
            await asyncio.sleep(86400)

async def run_periodic_research() -> None:
    """Background task to run the Automatic Researcher every Sunday morning."""
    import sys
    import os
    from datetime import datetime, timedelta
    
    script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'automatic_researcher.py')
    
    while True:
        try:
            now = datetime.now()
            # Calculate days until Sunday (6 = Sunday)
            days_ahead = 6 - now.weekday()
            if days_ahead < 0 or (days_ahead == 0 and now.hour >= 4):
                days_ahead += 7
                
            # Target 4 AM on Sunday
            target_time = now + timedelta(days=days_ahead)
            target_time = target_time.replace(hour=4, minute=0, second=0, microsecond=0)
            
            sleep_seconds = (target_time - now).total_seconds()
            logger.info(f"Automatic Research scheduled for {target_time} (in {sleep_seconds/3600:.1f} hours)")
            
            await asyncio.sleep(sleep_seconds)
            
            logger.info("Running Automatic Research...")
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Automatic Research completed successfully:\n{stdout.decode().strip()}")
            else:
                logger.error(f"Automatic Research failed with code {process.returncode}:\n{stderr.decode().strip()}")
                
        except Exception as e:
            logger.error(f"Error running Automatic Research: {e}")
            await asyncio.sleep(3600)

async def run_periodic_post_mortem() -> None:
    """Background task to run the Post-Mortem AI every Saturday morning."""
    import sys
    import os
    from datetime import datetime, timedelta
    
    script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'post_mortem.py')
    
    while True:
        try:
            now = datetime.now()
            # Calculate days until Saturday (5 = Saturday)
            days_ahead = 5 - now.weekday()
            if days_ahead < 0 or (days_ahead == 0 and now.hour >= 4):
                days_ahead += 7
                
            # Target 4 AM on Saturday
            target_time = now + timedelta(days=days_ahead)
            target_time = target_time.replace(hour=4, minute=0, second=0, microsecond=0)
            
            sleep_seconds = (target_time - now).total_seconds()
            logger.info(f"Post-Mortem AI scheduled for {target_time} (in {sleep_seconds/3600:.1f} hours)")
            
            await asyncio.sleep(sleep_seconds)
            
            logger.info("Running Post-Mortem AI...")
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Post-Mortem AI completed successfully:\n{stdout.decode().strip()}")
            else:
                logger.error(f"Post-Mortem AI failed with code {process.returncode}:\n{stderr.decode().strip()}")
                
        except Exception as e:
            logger.error(f"Error running Post-Mortem AI: {e}")
            await asyncio.sleep(3600)

async def run_periodic_db_maintenance() -> None:
    """Periodically clean up old database records to prevent unbounded growth.

    Deletes closed DecisionSnapshot and ShadowTrade records older than
    DB_RETENTION_DAYS (default 90 days) to maintain database performance.
    Runs weekly.
    """
    from datetime import timedelta

    retention_days = getattr(settings, 'DB_RETENTION_DAYS', 90)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_iso = cutoff_date.isoformat()

    while True:
        cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        try:
            logger.info(f"Starting database maintenance - removing records older than {retention_days} days ({cutoff_iso})")

            from src.db import get_engine
            from sqlalchemy import text, delete
            from src.db import DecisionSnapshot, ShadowTrade

            with get_engine().begin() as conn:
                result = conn.execute(
                    delete(DecisionSnapshot).where(
                        DecisionSnapshot.status == "closed",
                        DecisionSnapshot.closed_at < cutoff_iso
                    )
                )
                deleted_snapshots = result.rowcount

                result = conn.execute(
                    delete(ShadowTrade).where(
                        ShadowTrade.status == "closed",
                        ShadowTrade.closed_at < cutoff_iso
                    )
                )
                deleted_shadow_trades = result.rowcount

                if str(get_engine().url).startswith("sqlite"):
                    conn.execute(text("VACUUM"))

            logger.info(f"Database maintenance completed: {deleted_snapshots} snapshots, {deleted_shadow_trades} shadow trades deleted")

        except Exception as e:
            logger.error(f"Error during database maintenance: {e}")

        await asyncio.sleep(7 * 24 * 3600)

async def run_trading_bot() -> None:
    """Main trading bot loop."""
    global ex, strategy, risk_manager

    try:
        logger.info("Initializing Apex Oracle Bot v2.0.0")
        logger.info("=================================")
        logger.info("Configuration:")
        logger.info(f"Bot Name: {settings.BOT_NAME}")
        logger.info(f"Database: {settings.DATABASE_URL}")
        logger.info(f"Exchange: Alpaca Crypto (Paper={settings.ALPACA_BASE_URL.endswith('paper-api.alpaca.markets')})")
        logger.info(f"Symbols: {settings.SYMBOLS}")
        logger.info("=================================")

        # Kick off the transformer model/scaler load in the background as
        # early as possible. Measured cold-load cost: ~5-13s (dominated by
        # `import torch` ~7-8s and joblib.load() pulling in a cold sklearn
        # import ~3.5s), occasionally much worse (62s observed) under system
        # load. Previously this only happened lazily on the FIRST call to
        # transformer_brain() -- i.e. on the bot's first live trading
        # decision -- which froze the entire event loop (0 scheduler ticks
        # recorded during the load) for that whole duration: no other
        # symbol could be evaluated, the killswitch monitor couldn't run,
        # and the FastAPI health server couldn't respond.
        #
        # Firing it here, wrapped in to_thread, lets the slow imports/disk
        # I/O overlap with the DB/exchange-auth startup work below instead
        # of adding to the critical path serially, and guarantees it's
        # warm before the main loop's first trading decision.
        from src.committee.transformer_brain import get_ml_predictor
        model_warmup_task = asyncio.create_task(
            asyncio.to_thread(get_ml_predictor), name="model_warmup"
        )

        # Initialize database (handle connection failures gracefully)
        try:
            init_db()
            logger.info("Database connected successfully")
        except Exception as e:
            logger.warning(f"Database connection failed (will retry later): {e}")
            logger.info("Running in offline mode - some features may be limited")

        logger.info(settings.log_config())

        # Initialize exchange
        ex = AlpacaExchange()
        try:
            await ex.load()
            logger.info("Alpaca exchange connected")
        except RuntimeError:
            # Re-raise configuration and auth errors which are fatal
            raise
        except Exception as e:
            logger.warning(f"Alpaca exchange connection failed on startup: {e}. Bot will start in offline/retry mode.")

        # Initialize trading strategy and risk manager
        strategy = TradingStrategy(ex)
        risk_manager = RiskManager(ex)
        logger.info("Trading strategy and risk manager initialized")

        # Start API server
        await start_fastapi_server_async()
        logger.info("FastAPI server started")

        # Make sure the model warmup (fired above) has actually finished
        # before we start evaluating live signals. By this point it has
        # been running concurrently with DB init, exchange auth, and the
        # FastAPI server start, so this await is typically a no-op.
        try:
            await model_warmup_task
            logger.info("Transformer model warmup complete")
        except Exception as e:
            logger.warning(f"Transformer model warmup failed (will fall back to signal-only voting): {e}")

        # Helper: log any unhandled exception from a background task before
        # removing it from active_tasks. Without this, a crashing killswitch /
        # heartbeat / analyzer silently disappears with no log entry and never
        # restarts Ã¢â‚¬â€�? leaving safety mechanisms offline.
        def _on_task_done(task: asyncio.Task) -> None:
            active_tasks.discard(task)
            if not task.cancelled():
                try:
                    exc = task.exception()
                    if exc:
                        logger.critical(
                            f"Background task '{task.get_name()}' died with an unhandled "
                            f"exception and will NOT restart: {exc}",
                            exc_info=exc,
                        )
                except asyncio.CancelledError:
                    pass

        # Start Killswitch monitor
        active_tasks: set[asyncio.Task] = set()
        ks_task = asyncio.create_task(monitor_killswitch(risk_manager), name="killswitch")
        active_tasks.add(ks_task)
        ks_task.add_done_callback(_on_task_done)
        logger.info("Killswitch monitor started")

        # Start Scan Heartbeat
        heartbeat_task = asyncio.create_task(scan_heartbeat_loop(), name="heartbeat")
        active_tasks.add(heartbeat_task)
        heartbeat_task.add_done_callback(_on_task_done)
        logger.info("Scan heartbeat monitor started")

        # Start periodic analyzer
        analyzer_task = asyncio.create_task(run_periodic_analyzer(), name="analyzer")
        active_tasks.add(analyzer_task)
        analyzer_task.add_done_callback(_on_task_done)
        logger.info("Periodic analyzer task started")

        # Start periodic AutoML pipeline
        automl_task = asyncio.create_task(run_periodic_automl(), name="automl")
        active_tasks.add(automl_task)
        automl_task.add_done_callback(_on_task_done)
        logger.info("Periodic AutoML pipeline task started")

        # Start periodic Evolution Cull
        cull_task = asyncio.create_task(run_periodic_cull(), name="cull")
        active_tasks.add(cull_task)
        cull_task.add_done_callback(_on_task_done)
        logger.info("Periodic Evolution Cull task started")

        # Start periodic Automatic Research
        research_task = asyncio.create_task(run_periodic_research(), name="research")
        active_tasks.add(research_task)
        research_task.add_done_callback(_on_task_done)
        logger.info("Periodic Automatic Research task started")

        # Start periodic Post-Mortem AI
        post_mortem_task = asyncio.create_task(run_periodic_post_mortem(), name="post_mortem")
        active_tasks.add(post_mortem_task)
        post_mortem_task.add_done_callback(_on_task_done)
        logger.info("Periodic Post-Mortem AI task started")

        # Start periodic DB maintenance
        db_maintenance_task = asyncio.create_task(run_periodic_db_maintenance(), name="db_maintenance")
        active_tasks.add(db_maintenance_task)
        db_maintenance_task.add_done_callback(_on_task_done)
        logger.info("Periodic Database Maintenance task started")

        # Main trading loop
        logger.info(f"Bot initialization complete. Starting stateless REST polling loop for {settings.SYMBOLS} (interval: {settings.LOOP_INTERVAL_SEC}s).")

        try:
            while True:
                regime_flag = read_regime_flag()
                banned_symbols = get_banned_symbols()
                for symbol in settings.SYMBOLS:
                    try:
                        latest_bar_df = await ex.get_latest_bar(symbol)
                        if latest_bar_df.is_empty():
                            continue
                        
                        current_price = latest_bar_df["close"][0]

                        # Prune expired cooldown entries to prevent
                        # the dict from growing without bound.
                        now_ts = time.time()
                        expired = [k for k, v in cooldowns.items() if v < now_ts]
                        for k in expired:
                            del cooldowns[k]

                        # Dispatch signal processing to a background task
                        task = asyncio.create_task(
                            process_signal_for_symbol(
                                symbol, current_price, risk_manager, strategy, ex,
                                regime_flag=regime_flag,
                                banned_symbols=banned_symbols,
                            )
                        )
                        active_tasks.add(task)
                        task.add_done_callback(active_tasks.discard)
                    except Exception as sym_e:
                        logger.error(f"[MAIN_LOOP] Error polling {symbol}: {sym_e}")
                        
                # Wait for the next evaluation cycle
                await asyncio.sleep(settings.LOOP_INTERVAL_SEC)

        except Exception as e:
            logger.error(f"Stateless polling loop error: {e}")
            await asyncio.sleep(5)

    except KeyboardInterrupt:
        logger.info("Shutdown requested. Exiting gracefully.")
    except Exception as e:
        logger.error(f"Fatal error in bot: {e}", exc_info=True)
        raise
    finally:
        logger.info("Cleaning up background tasks and closing exchange...")
        for task in list(active_tasks):
            task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        if ex:
            await ex.close()
        logger.info("Shutdown complete.")


def run_bot() -> None:
    """Synchronous wrapper for async bot."""
    asyncio.run(run_trading_bot())
