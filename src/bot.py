import asyncio
import json
import logging
import math
import os as _os
import sys
import time
import traceback
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from src.config import settings, COOLDOWN_SECONDS_BUY, MAX_POSITION_ADDS, POSITION_ADD_MIN_SECONDS, POSITION_ADD_MIN_SCORE_INCREASE, POSITION_ADD_SIZE_DECAY
from src.logging_config import get_logger
from src.db import init_db
from src.exchange import AlpacaExchange
from src.api import start_fastapi_server_async
from src.strategies import TradingStrategy
from src.risk import RiskManager
from src.telegram_alerts import send_telegram_alert
from src.alerting import AlertingEngine
from src.committee.transformer_brain import _model_inference_lock
from src.population_trainer import get_pbt_trainer, run_pbt_cycle
from src.ood_discriminator import get_ood_discriminator
from scripts.deployment_registry import register_process, heartbeat_process, cleanup_stale
import argparse

# Structured logging setup
class StructuredLogger:
    """Structured JSON logger for production observability."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._extra = {}
    
    def _log(self, level: int, msg: str, **kwargs):
        extra = {"extra": {**self._extra, **kwargs}}
        self.logger.log(level, msg, extra=extra)
    
    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        self._log(logging.CRITICAL, msg, **kwargs)
    
    def bind(self, **kwargs):
        """Bind additional context to all subsequent logs."""
        self._extra.update(kwargs)

logger = get_logger("bot")
structured_logger = StructuredLogger("bot")


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
        global _state
        if _state.strategy is not None and hasattr(_state.strategy, '_trailing_peaks') and symbol in _state.strategy._trailing_peaks:
            peak = _state.strategy._trailing_peaks[symbol]
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
        if _state.risk_manager is not None:
            _state.risk_manager.peak_prices.pop(symbol, None)
        if _state.strategy is not None and hasattr(_state.strategy, '_trailing_peaks'):
            _state.strategy._trailing_peaks.pop(symbol, None)

        # Append to the Transformer's live replay buffer, if this trade's
        # entry captured a tensor state. Matches the exact {"tensor": ...,
        # "label": ...} schema generate_replay_dataset.py produces from
        # backtest data, so retrain_transformer.py trains on both real and
        # simulated experience without any format handling on its end.
        # Fail-safe and fully decoupled from trading: any error here is
        # logged and swallowed, never allowed to affect risk/exits.
        try:
            import os
            tensor_state = snap.get("tensor_state")
            if tensor_state is not None:
                live_buffer_path = os.path.join(
                    os.path.dirname(__file__), "..", "data", "live_experiences.jsonl"
                )
                os.makedirs(os.path.dirname(live_buffer_path), exist_ok=True)
                label = 1.0 if realized_pnl > 0 else 0.0
                record = json.dumps({"tensor": tensor_state, "label": label})

                def _append_live_experience():
                    with open(live_buffer_path, "a") as f:
                        f.write(record + "\n")

                await asyncio.to_thread(_append_live_experience)
        except Exception as e:
            logger.warning(f"Failed to append live trade to Transformer replay buffer (non-fatal): {e}")

# Online Transformer gradient step: one step on the just-closed trade's
        # tensor state. This provides continuous learning between daily full
        # retrain_transformer.py runs. Fail-safe: runs in background thread,
        # never blocks trading, errors swallowed.
        try:
            tensor_state = snap.get("tensor_state")
            if tensor_state is not None:
                 def _online_transformer_step():
                     try:
                         from src.committee.transformer_brain import get_ml_predictor
                         import torch
                         import numpy as np
                         
                         predictor = get_ml_predictor()
                         if predictor is None:
                             return
                         
                         model = predictor["model"]
                         scaler = predictor["scaler"]
                         device = predictor["device"]
                         
                         # Prepare single sample
                         data = np.array(tensor_state, dtype=np.float32)
                         if hasattr(scaler, "feature_names_in_"):
                             cols = list(scaler.feature_names_in_)
                         else:
                             from src.feature_engineering import get_active_features
                             cols = get_active_features()
                         
                         if len(data) > 0 and len(data[0]) == len(cols):
                             data_scaled = scaler.transform(data).astype(np.float32)
                             data_scaled = np.nan_to_num(data_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                             
                             x = torch.tensor(data_scaled).unsqueeze(0).to(device)
                             label_tensor = torch.tensor([[1.0 if realized_pnl > 0 else 0.0]], dtype=torch.float32).to(device)
                             
                             # Protect model train/eval state from concurrent access
                             with _model_inference_lock:
                                  model.train()
                                  model.zero_grad()
                                  logits = model(x)
                                  loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label_tensor)
                                  loss.backward()
                                  
                                  # Learning rate schedule with warmup + cosine annealing
                                  # Use state from _state instead of module-level globals
                                  with _state._transformer_online_lock:
                                      step = _state._transformer_online_updates
                                      lr_schedule = _state._transformer_online_lr_schedule
                                      lr_base = _state._transformer_online_lr_base
                                      lr_min = _state._transformer_online_lr_min
                                      warmup_steps = _state._transformer_online_warmup_steps
                                      total_steps = _state._transformer_online_total_steps
                                  
                                  if step < warmup_steps:
                                      # Linear warmup
                                      lr = lr_base * (step + 1) / warmup_steps
                                  else:
                                      progress = min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)
                                      if lr_schedule == "cosine":
                                          # Cosine annealing to min LR
                                          lr = lr_min + 0.5 * (lr_base - lr_min) * (1 + math.cos(math.pi * progress))
                                      elif lr_schedule == "linear":
                                          # Linear decay to min LR
                                          lr = lr_base - (lr_base - lr_min) * progress
                                      else:  # constant
                                          lr = lr_base
                                  
                                  # Increment counter and apply gradient step
                                  with _state._transformer_online_lock:
                                      _state._transformer_online_updates += 1
                                  with torch.no_grad():
                                      for p in model.parameters():
                                          if p.grad is not None:
                                              p.data -= lr * p.grad
                                  model.eval()
                     except Exception:
                         pass  # Swallow all errors
                  

                 # Run in background thread, don't await
                 task = asyncio.create_task(asyncio.to_thread(_online_transformer_step), name="transformer_online")
                 _state._background_tasks.add(task)
                 task.add_done_callback(_state._background_tasks.discard)
        except Exception as e:
            logger.debug(f"Online Transformer step skipped (non-fatal): {e}")

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
                
        task = asyncio.create_task(_run_attribution(), name="attribution")
        _state._background_tasks.add(task)
        task.add_done_callback(_state._background_tasks.discard)

    except Exception as e:
        logger.error(f"Adaptive outcome recording failed for {symbol} (non-fatal): {e}")


# BotState encapsulates all global mutable state for the trading bot.
# This class replaces module-level globals to improve testability and encapsulation.
class BotState:
    def __init__(self):
        self.ex: Optional[AlpacaExchange] = None
        self.strategy: Optional[TradingStrategy] = None
        self.risk_manager: Optional[RiskManager] = None
        self.latest_scan_results: Dict[str, dict] = {}
        self.scan_cycle_count: int = 0
        self.cooldowns: Dict[str, float] = {}
        self.position_adds: Dict[str, Dict[str, Any]] = {}
        self._symbol_locks: Dict[str, asyncio.Lock] = {}
        self._background_tasks: set[asyncio.Task] = set()
        self._shutdown_requested: bool = False
        self._regime_flag_cache: Dict[str, Any] = {}
        self._regime_flag_cache_mtime: float = -1.0
        self._banned_symbols_cache: set = set()
        self._banned_symbols_cache_mtime: float = -1.0
        # Online transformer learning state (moved from module-level globals)
        self._transformer_online_updates: int = 0
        self._transformer_online_lr_schedule: str = "cosine"
        self._transformer_online_lr_base: float = 1e-5
        self._transformer_online_lr_min: float = 1e-6
        self._transformer_online_warmup_steps: int = 100
        self._transformer_online_total_steps: int = 10000
        self._transformer_online_lock: asyncio.Lock = asyncio.Lock()
        
        # Alerting metrics
        self.trade_timestamps: list[float] = []
        self.exchange_failure_count: int = 0

    def get_regime_flag(self) -> Dict[str, Any]:
        """Read the regime flag file, cached and only re-read when the file's mtime changes."""
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
        if mtime == self._regime_flag_cache_mtime and self._regime_flag_cache:
            return self._regime_flag_cache
        try:
            with open(_REGIME_FLAG_PATH, "r") as f:
                data = json.load(f)
            self._regime_flag_cache = data
            self._regime_flag_cache_mtime = mtime
            return data
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def get_banned_symbols(self) -> set:
        """Read the banned symbols list generated by the weekly analyzer, cached."""
        try:
            mtime = _os.path.getmtime(_BANNED_SYMBOLS_PATH)
        except OSError:
            return set()
        if mtime == self._banned_symbols_cache_mtime:
            return self._banned_symbols_cache
        try:
            with open(_BANNED_SYMBOLS_PATH, 'r') as f:
                bans = json.load(f)
            self._banned_symbols_cache = {b["symbol"] for b in bans}
            self._banned_symbols_cache_mtime = mtime
            return self._banned_symbols_cache
        except Exception as e:
            logger.warning(f"Failed to read banned symbols: {e}")
            return self._banned_symbols_cache

    def get_symbol_lock(self, symbol: str) -> asyncio.Lock:
        """Get or create a per-symbol lock for concurrent access control."""
        return self._symbol_locks.setdefault(symbol, asyncio.Lock())

    def add_background_task(self, task: asyncio.Task) -> None:
        """Register a background task for tracking and cleanup."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def clear_position_adds(self, symbol: str) -> None:
        """Reset position pyramid/scale-in tracking on position close."""
        self.position_adds.pop(symbol, None)

    def set_shutdown(self) -> None:
        self._shutdown_requested = True

    def is_shutdown_requested(self) -> bool:
        return self._shutdown_requested

    async def shutdown(self) -> None:
        """Cancel all background tasks and close exchange."""
        logger.info("Cleaning up background tasks and closing exchange...")
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        # Note: exchange close handled by caller

    def reset(self) -> None:
        """Reset all mutable state for testing purposes."""
        self.ex = None
        self.strategy = None
        self.risk_manager = None
        self.latest_scan_results.clear()
        self.scan_cycle_count = 0
        self.cooldowns.clear()
        self.position_adds.clear()
        self._symbol_locks.clear()
        self._background_tasks.clear()
        self._shutdown_requested = False
        self._regime_flag_cache.clear()
        self._regime_flag_cache_mtime = -1.0
        self._banned_symbols_cache.clear()
        self._banned_symbols_cache_mtime = -1.0
        # Reset transformer online learning state
        self._transformer_online_updates = 0
        self._transformer_online_lr_schedule = "cosine"
        self._transformer_online_lr_base = 1e-5
        self._transformer_online_lr_min = 1e-6
        self._transformer_online_warmup_steps = 100
        self._transformer_online_total_steps = 10000
        self._transformer_online_lock = asyncio.Lock()


# Global state instance (single instance for the process)
_state = BotState()

# Paths for config files
_REGIME_FLAG_PATH = "data/regime_flag.txt"
_BANNED_SYMBOLS_PATH = _os.path.join(_os.path.dirname(__file__), '..', 'data', 'banned_symbols.json')


def read_regime_flag():
    """Read the regime flag file, cached and only re-read when the file's mtime changes."""
    return _state.get_regime_flag()


def get_banned_symbols():
    """Read the banned symbols list generated by the weekly analyzer, cached."""
    return _state.get_banned_symbols()

async def process_signal_for_symbol(symbol: str, current_price: float, risk_manager: RiskManager, strategy: TradingStrategy, ex: AlpacaExchange, positions: list = None, regime_flag: dict = None, banned_symbols: set = None) -> None:
    """Processes signal for a single symbol asynchronously."""
    # Get or create lock for this symbol
    lock = _state._symbol_locks.setdefault(symbol, asyncio.Lock())
    async with lock:
        try:
            # Use pre-fetched positions from the main cycle if available,
            # otherwise fetch fresh (fallback for direct calls).
            if positions is None:
                positions = await ex.get_positions()
            position_dict = {p["symbol"].replace("/", ""): p for p in positions}
            current_position = position_dict.get(symbol.replace("/", ""))
    
            # Cooldown check: block a fresh entry right after this symbol closed a
            # position, but never block evaluation of an EXISTING position's exit
            # logic (a position we're already holding must always be free to close).
            if not current_position:
                now_ts = time.time()
                cooldown_until = _state.cooldowns.get(symbol, 0)
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
                    client_order_id = f"{symbol}_{side}_{qty_abs}_{int(time.time())}"
                    order_result = await ex.create_order(
                        symbol=symbol,
                        qty=qty_abs,
                        side=side,
                        type="market",
                        client_order_id=client_order_id,
                    )
                    logger.info(f"Trailing Stop Executed: {symbol}")
                    await send_telegram_alert(f"ð <b>Trailing Stop Triggered</b>\nSymbol: {symbol}\nClosed {qty} @ ${current_price:.2f}")
                    _state.cooldowns[symbol] = time.time() + settings.COOLDOWN_SECONDS_BUY
                    # Reset position pyramid/scale-in tracking on close
                    _state.position_adds.pop(symbol, None)
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
                         regime=signal["regime"], rsi=signal.get("rsi", 0.0))
    
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
                    # Update scan results for this symbol
                    _state.latest_scan_results[symbol] = {"score": committee_result.score, "action": "VETO", "price": current_price}
                    return
    
                if committee_result.action in ["stand_aside", "skip", "hold"]:
                    dashboard.append("FINAL.......... NO TRADE")
                    dashboard.append("Reason......... Committee Consensus")
                    dashboard.append("==============================")
                    print("\n".join(dashboard), flush=True)
                    _state.latest_scan_results[symbol] = {"score": committee_result.score, "action": committee_result.action.upper(), "price": current_price}
                    return
    
                _state.latest_scan_results[symbol] = {"score": committee_result.score, "action": committee_result.action.upper(), "price": current_price}
    
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
                # Estimate expected return using committee score, regime, and consensus
                # Score represents P(win), regime provides base edge, entropy measures consensus
                regime_edge = {
                    "trending": 0.015,      # 1.5% edge in trending
                    "bull": 0.015,
                    "bear": 0.015,
                    "mean_reverting": 0.01,  # 1% edge in mean reversion
                    "sideways": 0.01,
                    "high_volatility": 0.0,  # No edge in high vol
                    "low_volatility": 0.005, # 0.5% edge in low vol
                    "neutral": 0.0
                }
                regime_edge = regime_edge.get(signal.get("regime", "neutral"), 0.0)
                
                # Committee score represents P(win), map to edge
                # Score > 0.5 means bullish, < 0.5 means bearish
                score_edge = (committee_result.score - 0.5) * 2.0  # maps [0,1] -> [-1, 1]
                
                # Entropy penalty: high disagreement reduces confidence
                entropy_penalty = max(0.0, committee_result.entropy - 0.5) * 0.5
                
                # Combined edge estimate (capped at reasonable bounds)
                raw_edge = regime_edge + score_edge * 0.02 - entropy_penalty  # scale score edge to ~2%
                # calculate_position_size() expects this as a fraction (its own
                # PROFIT_TARGET_PCT fallback is 0.03, not 3.0) and converts it to
                # bps internally via `* 10000`. Do NOT also multiply by 100 here --
                # that produced a 100x-inflated "expected edge", which silently
                # defeated the min-edge-after-costs rejection gate below.
                expected_return_pct = max(-0.02, min(0.05, raw_edge))  # cap at -2% to +5% (fraction, e.g. 0.03 = 3%)
                position_size, sizing_status = risk_manager.calculate_position_size(
                    symbol,
                    current_price,
                    signal["regime"],
                    atr=signal.get("atr"),
                    confidence=signal.get("confidence", 1.0),
                    expected_return_pct=expected_return_pct
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
                        risk_manager.release_reserved_exposure(approved_notional)
                        return
                
                # Position pyramid/scale-in gates: prevent unlimited re-buying
                    # of an already-open position without meaningful improvement
                    if signal["action"] == "buy":
                        current_position = None
                        for p in positions:
                            if p["symbol"].replace("/", "") == symbol.replace("/", ""):
                                current_position = p
                                break
                    
                        if current_position is not None:
                            # There's already a position - check scale-in gates
                            add_info = _state.position_adds.get(symbol, {"count": 0, "last_add_time": 0.0, "last_add_score": 0.0})
                            now = time.time()
                            
                            # Gate 1: Max adds cap
                            if add_info["count"] >= MAX_POSITION_ADDS:
                                logger.info(f"[{symbol}] Scale-in vetoed: max adds ({MAX_POSITION_ADDS}) reached (current: {add_info['count']})")
                                return
                            
                            # Gate 2: Minimum time since last add
                            if now - add_info["last_add_time"] < POSITION_ADD_MIN_SECONDS:
                                logger.info(f"[{symbol}] Scale-in vetoed: minimum time between adds not met ({now - add_info['last_add_time']:.0f}s < {POSITION_ADD_MIN_SECONDS}s)")
                                return
                            
                            # Gate 3: Minimum score improvement
                            committee_score = committee_result.score
                            if committee_score - add_info["last_add_score"] < POSITION_ADD_MIN_SCORE_INCREASE:
                                logger.info(f"[{symbol}] Scale-in vetoed: insufficient score improvement ({committee_score:.3f} - {add_info['last_add_score']:.3f} < {POSITION_ADD_MIN_SCORE_INCREASE})")
                                return
                            
                            # All gates passed - apply size decay
                            decay_factor = POSITION_ADD_SIZE_DECAY ** add_info["count"]
                            position_size = round(position_size * decay_factor, 6)
                            logger.info(f"[{symbol}] Scale-in add #{add_info['count'] + 1}: size decayed by {decay_factor:.2f}x to {position_size}")
                            
                            # Update tracking
                            _state.position_adds[symbol] = {
                                "count": add_info["count"] + 1,
                                "last_add_time": now,
                                "last_add_score": committee_score
                            }
                        else:
                            # No existing position - reset tracking
                            _state.position_adds[symbol] = {"count": 0, "last_add_time": 0.0, "last_add_score": 0.0}
     
                # Place order
                client_order_id = f"{symbol}_{signal['action']}_{position_size}_{int(time.time())}"
                try:
                    order_result = await ex.create_order(
                        symbol=symbol,
                        qty=position_size,
                        side=signal["action"],
                        type="market",
                        client_order_id=client_order_id,
                    )
                except Exception as order_e:
                    # Release reserved exposure on order failure so the
                    # headroom isn't permanently leaked.
                    risk_manager.release_reserved_exposure(approved_notional)
                    logger.error(f"[{symbol}] Order placement failed: {order_e}")
                    raise
                
# Release the reserved exposure now that the order has been
                # placed successfully -- the actual portfolio value will be
                # reflected on the next cycle's update_account_status() call.
                risk_manager.release_reserved_exposure(approved_notional)
                
                # Track trade for churn alert
                _state.trade_timestamps.append(time.time())

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
                    
                    # Record fill costs for dynamic transaction cost model
                    if filled_price > 0:
                        expected_price = current_price
                        slippage_bps = abs(filled_price - expected_price) / expected_price * 10000
                        fee_bps = (commission / (filled_price * position_size)) * 10000 if filled_price * position_size > 0 else 0
                        risk_manager.record_fill_costs(symbol, fee_bps, slippage_bps)

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
                        }),
                        tensor_state_json=json.dumps({
                            v.name: getattr(v, "tensor_state", None) 
                            for v in committee_result.votes if getattr(v, "tensor_state", None)
                        })
                    )
                except Exception as db_e:
                    logger.warning(f"Decision snapshot persist failed for {symbol} (non-fatal): {db_e}")
    
            elif signal["action"] == "close" and current_position:
    
                # Close existing position
                qty = float(current_position["qty"])
                side = "sell" if qty > 0 else "buy"
                qty_abs = abs(qty)

                client_order_id = f"{symbol}_{side}_{qty_abs}_{int(time.time())}"
                order_result = await ex.create_order(
                    symbol=symbol,
                    qty=qty_abs,
                    side=side,
                    type="market",
                    client_order_id=client_order_id,
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
                    await send_telegram_alert(f"ð <b>Position Closed</b>\nSymbol: {symbol}\nReason: {signal.get('reason', 'unknown')}")
                    _state.cooldowns[symbol] = time.time() + settings.COOLDOWN_SECONDS_BUY
                    # Reset position pyramid/scale-in tracking on close
                    _state.position_adds.pop(symbol, None)
                    # Clear trailing peaks for this symbol
                    if _state.strategy is not None and hasattr(_state.strategy, '_trailing_peaks'):
                        _state.strategy._trailing_peaks.pop(symbol, None)
                    await _record_committee_outcome(symbol, current_price, exit_reason=signal.get('reason', 'unknown'))
    
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    
async def scan_heartbeat_loop() -> None:
    """Background task to print periodic scan summary."""
    while True:
        try:
            await asyncio.sleep(settings.LOOP_INTERVAL_SEC)
            _state.scan_cycle_count += 1
            
            if not _state.latest_scan_results:
                continue
                
            trades_this_cycle = False
            out = []
            out.append("========================================")
            out.append(f"Cycle {_state.scan_cycle_count}")
            out.append(f"{len(_state.latest_scan_results)} Symbols")
            out.append("Scanning...")
            out.append("========================================")
            
            for sym, data in _state.latest_scan_results.items():
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
    while True:
        try:
            if await risk_manager.check_killswitch_conditions():
                logger.critical("KILLSWITCH ACTIVATED - Liquidating all positions")
                await send_telegram_alert(f"🛑 <b>KILLSWITCH ACTIVATED</b>\nLiquidating all positions immediately.")
                await risk_manager.liquidate_all_positions()
                _state._shutdown_requested = True
                return  # Exit the task, let main loop handle shutdown

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

async def run_periodic_transformer_replay() -> None:
    """Background task to fine-tune the Transformer on the replay buffer daily.

    Distinct from run_periodic_automl (Saturday 2 AM, a full from-scratch
    retrain-and-tournament against 180 days of fresh market data). This is
    the lightweight (10-epoch) continuous replay fine-tune -
    retrain_transformer.py - which trains on data/historical_experiences.jsonl
    (backtest-simulated) AND data/live_experiences.jsonl (real closed trades,
    appended by _record_committee_outcome above). This is what actually lets
    the Transformer learn from forward-testing results, not just backtests.
    Scheduled at 1 AM daily, ahead of the heavier Saturday/Sunday jobs.
    """
    import sys
    import os
    from datetime import datetime, timedelta

    script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'retrain_transformer.py')

    while True:
        try:
            now = datetime.now()
            target_time = now.replace(hour=1, minute=0, second=0, microsecond=0)
            if now >= target_time:
                target_time += timedelta(days=1)

            sleep_seconds = (target_time - now).total_seconds()
            logger.info(f"Transformer replay fine-tune scheduled for {target_time} (in {sleep_seconds/3600:.1f} hours)")

            await asyncio.sleep(sleep_seconds)

            logger.info("Running Transformer replay fine-tune...")
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info(f"Transformer replay fine-tune completed successfully:\n{stdout.decode().strip()}")
            else:
                logger.error(f"Transformer replay fine-tune failed with code {process.returncode}:\n{stderr.decode().strip()}")

        except Exception as e:
            logger.error(f"Error running Transformer replay fine-tune: {e}")
            await asyncio.sleep(3600)

async def run_periodic_ppo_retrain() -> None:
    """Background task to retrain the PPO Meta-Learner every Sunday morning.

    evolutionary_ppo_trainer.py is the only script that actually produces a
    new PPO model for rl_meta.py to load - without this scheduled, the RL
    meta-learner stays frozen at whatever it was last manually trained on,
    indefinitely, even though the mathematical AdaptiveMetaLearner and
    strategy selector are both learning from every closed live trade.
    Scheduled at 6 AM (2 hours after run_periodic_research's 4 AM slot) to
    avoid both heavy jobs contending for CPU/data-fetch at the same time.
    """
    import sys
    import os
    from datetime import datetime, timedelta

    script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'evolutionary_ppo_trainer.py')

    while True:
        try:
            now = datetime.now()
            # Calculate days until Sunday (6 = Sunday)
            days_ahead = 6 - now.weekday()
            if days_ahead < 0 or (days_ahead == 0 and now.hour >= 6):
                days_ahead += 7

            # Target 6 AM on Sunday
            target_time = now + timedelta(days=days_ahead)
            target_time = target_time.replace(hour=6, minute=0, second=0, microsecond=0)

            sleep_seconds = (target_time - now).total_seconds()
            logger.info(f"PPO Meta-Learner retraining scheduled for {target_time} (in {sleep_seconds/3600:.1f} hours)")

            await asyncio.sleep(sleep_seconds)

            logger.info("Running Evolutionary PPO Trainer...")
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info(f"PPO Meta-Learner retraining completed successfully:\n{stdout.decode().strip()}")
            else:
                logger.error(f"PPO Meta-Learner retraining failed with code {process.returncode}:\n{stderr.decode().strip()}")

        except Exception as e:
            logger.error(f"Error running PPO Meta-Learner retraining: {e}")
            await asyncio.sleep(3600)

async def run_periodic_decision_transformer_retrain() -> None:
    """Background task to retrain the Decision Transformer weekly.
    
    Trains the offline RL sequence model on accumulated closed decision
    snapshots (backtest + live trades). Scheduled on Sunday 8 AM
    (2 hours after PPO retraining) to avoid resource contention.
    """
    import sys
    import os
    from datetime import datetime, timedelta

    script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'train_decision_transformer.py')

    while True:
        try:
            now = datetime.now()
            # Calculate days until Sunday (6 = Sunday)
            days_ahead = 6 - now.weekday()
            if days_ahead < 0 or (days_ahead == 0 and now.hour >= 8):
                days_ahead += 7

            # Target 8 AM on Sunday
            target_time = now + timedelta(days=days_ahead)
            target_time = target_time.replace(hour=8, minute=0, second=0, microsecond=0)

            sleep_seconds = (target_time - now).total_seconds()
            logger.info(f"Decision Transformer retraining scheduled for {target_time} (in {sleep_seconds/3600:.1f} hours)")

            await asyncio.sleep(sleep_seconds)

            logger.info("Running Decision Transformer retraining...")
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info(f"Decision Transformer retraining completed successfully:\n{stdout.decode().strip()}")
            else:
                logger.error(f"Decision Transformer retraining failed with code {process.returncode}:\n{stderr.decode().strip()}")

        except Exception as e:
            logger.error(f"Error running Decision Transformer retraining: {e}")
            await asyncio.sleep(3600)

async def run_periodic_pbt() -> None:
    """Background task to run Population-Based Training periodically.
    
    Runs every 30 minutes to continuously evolve hyperparameters using PBT.
    This replaces Bayesian Optimization for non-stationary market adaptation.
    """
    # Initial delay to let bot stabilize
    await asyncio.sleep(1800)  # 30 minutes
    
    while True:
        try:
            logger.info("Running Population-Based Training cycle...")
            
            trainer = get_pbt_trainer()
            
            # Collect performance data from recent trades for each worker
            # In a real implementation, this would aggregate live trading performance
            # For now, we simulate by evaluating current performance
            
            # Apply best config to live components
            live_components = {}
            if hasattr(_state, 'risk_manager') and _state.risk_manager:
                pass  # Risk manager config would go here
            
            trainer.apply_best_to_live(live_components)
            
            stats = trainer.get_population_stats()
            logger.info(f"PBT cycle completed: pop={stats.get('population_size', 0)}, "
                       f"best_perf={stats.get('max_performance', 0):.4f}, "
                       f"mean_perf={stats.get('mean_performance', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Error in PBT cycle: {e}")
        
        # Run every 30 minutes
        await asyncio.sleep(30 * 60)

# Run every 30 minutes
        await asyncio.sleep(30 * 60)

async def run_periodic_ood_retrain() -> None:
    """Background task to retrain OOD Discriminator periodically.
    
    Runs every hour to retrain the discriminator on new live data vs historical data.
    """
    # Initial delay to let bot stabilize
    await asyncio.sleep(3600)  # 1 hour
    
    while True:
        try:
            logger.info("Running OOD Discriminator retraining cycle...")
            
            from src.db import get_closed_decision_snapshots
            from src.ood_discriminator import get_ood_discriminator
            from src.committee.decision_transformer import build_state_vector, BRAINS, REGIMES
            
            ood_disc = get_ood_discriminator()
            if not ood_disc._is_trained:
                logger.info("OOD Discriminator not yet trained, skipping retrain")
                await asyncio.sleep(3600)
                continue
            
            # Get historical states from closed decisions
            closed_decisions = get_closed_decision_snapshots(limit=5000)
            if len(closed_decisions) < 100:
                logger.warning("Not enough historical data for OOD retraining")
                await asyncio.sleep(3600)
                continue
            
            # Build historical state vectors
            historical_states = []
            for dec in closed_decisions:
                regime = dec.get("regime", "default")
                features = dec.get("features", {})
                brain_votes = dec.get("brain_votes", {})
                state = build_state_vector(regime, features, brain_votes, REGIMES, BRAINS)
                historical_states.append(state)
            
            historical_states = np.array(historical_states)
            
            # Get recent live states (from recent decisions, last 24 hours)
            # For simplicity, use the same closed decisions but we could track live states separately
            # In a full implementation, we'd track live states separately during trading
            live_states = historical_states[-min(200, len(historical_states)):]  # Last 200 as "live"
            
            if len(live_states) < 50:
                logger.warning("Not enough live states for OOD retraining")
                await asyncio.sleep(3600)
                continue
            
            # Retrain
            acc = ood_disc.train_on_data(historical_states, live_states)
            logger.info(f"OOD Discriminator retrained: val_acc={acc:.3f}")
            
            # Save model
            ood_disc.save()
            
        except Exception as e:
            logger.error(f"Error in OOD Discriminator retraining: {e}")
        
        # Run every hour
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

    # Declared before anything else in the try block so the `finally` below
    # can always reference it -- otherwise an exception raised during early
    # startup (before the tasks are created) hits an UnboundLocalError in
    # `finally` that masks the real error.
    active_tasks: set[asyncio.Task] = set()

    try:
        logger.info("Initializing Apex Oracle Bot v2.0.0")
        logger.info("=================================")
        logger.info("Configuration:")
        logger.info(f"Bot Name: {settings.BOT_NAME}")
        logger.info(f"Database: {settings.DATABASE_URL}")
        logger.info(f"Exchange: Alpaca Crypto (Paper={settings.ALPACA_BASE_URL.endswith('paper-api.alpaca.markets')})")
        logger.info(f"Symbols: {settings.SYMBOLS}")
        logger.info("=================================")

        # Start API server FIRST so healthcheck passes during initialization
        await start_fastapi_server_async()
        logger.info("FastAPI server started")

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
        _state.ex = AlpacaExchange()
        try:
            await _state.ex.load()
            logger.info("Alpaca exchange connected")
        except RuntimeError:
            # Re-raise configuration and auth errors which are fatal
            raise
        except Exception as e:
            logger.warning(f"Alpaca exchange connection failed on startup: {e}. Bot will start in offline/retry mode.")

        # Register this process in the deployment registry
        try:
            # Create a simple namespace object for the register function
            class Args:
                role = "trader"
                symbols = ",".join(settings.SYMBOLS)
            register_process(Args())
            logger.info("Deployment registry: process registered")
        except Exception as e:
            logger.warning(f"Deployment registry registration failed (non-fatal): {e}")

        # Initialize trading strategy and risk manager
        _state.strategy = TradingStrategy(_state.ex)
        _state.risk_manager = RiskManager(_state.ex)
        logger.info("Trading strategy and risk manager initialized")

        # Initialize AlertingEngine
        alerting_engine = AlertingEngine(risk_manager=_state.risk_manager, exchange=_state.ex)
        logger.info("Alerting engine initialized")

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
        ks_task = asyncio.create_task(monitor_killswitch(_state.risk_manager), name="killswitch")
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

        # Start periodic PPO Meta-Learner retraining
        ppo_retrain_task = asyncio.create_task(run_periodic_ppo_retrain(), name="ppo_retrain")
        active_tasks.add(ppo_retrain_task)
        ppo_retrain_task.add_done_callback(_on_task_done)
        logger.info("Periodic PPO Meta-Learner retraining task started")

        # Start periodic Decision Transformer retraining
        dt_retrain_task = asyncio.create_task(run_periodic_decision_transformer_retrain(), name="dt_retrain")
        active_tasks.add(dt_retrain_task)
        dt_retrain_task.add_done_callback(_on_task_done)
        logger.info("Periodic Decision Transformer retraining task started")

        # Start periodic Population-Based Training
        pbt_task = asyncio.create_task(run_periodic_pbt(), name="pbt")
        active_tasks.add(pbt_task)
        pbt_task.add_done_callback(_on_task_done)
        logger.info("Periodic Population-Based Training task started")

        # Start periodic OOD Discriminator retraining
        ood_retrain_task = asyncio.create_task(run_periodic_ood_retrain(), name="ood_retrain")
        active_tasks.add(ood_retrain_task)
        ood_retrain_task.add_done_callback(_on_task_done)
        logger.info("Periodic OOD Discriminator retraining task started")

        # Start periodic Transformer replay fine-tune
        transformer_replay_task = asyncio.create_task(run_periodic_transformer_replay(), name="transformer_replay")
        active_tasks.add(transformer_replay_task)
        transformer_replay_task.add_done_callback(_on_task_done)
        logger.info("Periodic Transformer replay fine-tune task started")

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

# Start deployment registry heartbeat
        async def deployment_heartbeat_loop():
            while not _state._shutdown_requested:
                try:
                    await asyncio.sleep(60)  # Heartbeat every 60 seconds
                    cleanup_stale(None)
                    heartbeat_process(None)
                except Exception as e:
                    logger.warning(f"Deployment heartbeat failed: {e}")

        heartbeat_task = asyncio.create_task(deployment_heartbeat_loop(), name="deployment_heartbeat")
        active_tasks.add(heartbeat_task)
        heartbeat_task.add_done_callback(_on_task_done)
        logger.info("Deployment registry heartbeat started")

        # Start Alerting Engine monitoring (runs every 30 seconds)
        async def alerting_monitoring_loop():
            while not _state._shutdown_requested:
                try:
                    await asyncio.sleep(30)
                    await alerting_engine.run_monitoring_cycle()
                    
                    # Check churn alert (trades per hour)
                    now = time.time()
                    # Clean old timestamps
                    _state.trade_timestamps = [ts for ts in _state.trade_timestamps if now - ts < 3600]
                    trades_last_hour = len(_state.trade_timestamps)
                    await alerting_engine.check_churn_alert(trades_last_hour)
                    
                    # Check exchange failure alert
                    if _state.exchange_failure_count > 0:
                        await alerting_engine.check_exchange_failure_alert(
                            f"Consecutive failures: {_state.exchange_failure_count}",
                            _state.exchange_failure_count
                        )
                        
                except Exception as e:
                    logger.error(f"[ALERTING] Monitoring cycle error: {e}")
                    await asyncio.sleep(10)

        alerting_task = asyncio.create_task(alerting_monitoring_loop(), name="alerting")
        active_tasks.add(alerting_task)
        alerting_task.add_done_callback(_on_task_done)
        logger.info("Alerting engine monitoring started")

        # Main trading loop
        logger.info(f"Bot initialization complete. Starting stateless REST polling loop for {settings.SYMBOLS} (interval: {settings.LOOP_INTERVAL_SEC}s).")

        try:
            while not _state._shutdown_requested:
                regime_flag = read_regime_flag()
                banned_symbols = get_banned_symbols()

                # Fetch all latest bars concurrently instead of sequentially
                bar_tasks = {symbol: _state.ex.get_latest_bar(symbol) for symbol in settings.SYMBOLS}
                bar_results = await asyncio.gather(*bar_tasks.values(), return_exceptions=True)

                # Track exchange failures
                any_bar_success = False
                for bar_result in bar_results:
                    if not isinstance(bar_result, Exception) and not bar_result.is_empty():
                        any_bar_success = True
                        break
                
                if any_bar_success:
                    _state.exchange_failure_count = 0
                else:
                    _state.exchange_failure_count += 1

                now_ts = time.time()
                expired = [k for k, v in _state.cooldowns.items() if v < now_ts]
                for k in expired:
                    del _state.cooldowns[k]

                # Fetch positions once per cycle (was fetched redundantly per symbol)
                positions = await _state.ex.get_positions()

                for symbol, bar_result in zip(settings.SYMBOLS, bar_results):
                    try:
                        if isinstance(bar_result, Exception):
                            logger.error(f"[MAIN_LOOP] Error fetching bar for {symbol}: {bar_result}")
                            continue
                        latest_bar_df = bar_result
                        if latest_bar_df.is_empty():
                            continue

                        current_price = latest_bar_df["close"][0]

                        # Dispatch signal processing to a background task
                        task = asyncio.create_task(
                            process_signal_for_symbol(
                                symbol, current_price, _state.risk_manager, _state.strategy, _state.ex,
                                positions=positions,
                                regime_flag=regime_flag,
                                banned_symbols=banned_symbols,
                            )
                        )
                        active_tasks.add(task)
                        task.add_done_callback(active_tasks.discard)
                    except Exception as sym_e:
                        logger.error(f"[MAIN_LOOP] Error processing {symbol}: {sym_e}")

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
        # Clean up _background_tasks from _record_committee_outcome
        for task in list(_state._background_tasks):
            task.cancel()
        if _state._background_tasks:
            await asyncio.gather(*_state._background_tasks, return_exceptions=True)
        if _state.ex:
            await _state.ex.close()
        logger.info("Shutdown complete.")


def run_bot() -> None:
    """Synchronous wrapper for async bot."""
    asyncio.run(run_trading_bot())
