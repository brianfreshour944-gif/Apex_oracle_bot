import asyncio
import hashlib
import json
import logging
import math
import os as _os
import sys
import threading
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
from tenacity import RetryError

from scripts.deployment_registry import cleanup_stale, heartbeat_process, register_process
from src import model_store
from src.alerting import get_alerting_engine
from src.api import start_fastapi_server_async
from src.committee.transformer_brain import _model_inference_lock
from src.config import (
    MAX_POSITION_ADDS,
    POSITION_ADD_MIN_SCORE_INCREASE,
    POSITION_ADD_MIN_SECONDS,
    POSITION_ADD_SIZE_DECAY,
    settings,
)
from src.db import get_recent_realized_pnl, init_db
from src.exchange import AlpacaExchange
from src.logging_config import (
    configure_structlog,
    get_logger,
    log_training_job_result,
    log_training_job_start,
)
from src.persistent_state import PersistentBotState
from src.population_trainer import get_pbt_trainer
from src.risk import RiskManager
from src.strategies import TradingStrategy
from src.telegram_alerts import send_telegram_alert


def _describe_exception(e: Exception) -> str:
    """Unwrap tenacity.RetryError to show the actual underlying failure.

    str(RetryError) is just "RetryError[<Future at 0x... state=finished
    raised APIError>]" -- it never shows the real error message, which is
    exactly the information needed to diagnose why an order was rejected
    (insufficient buying power, invalid qty, symbol not tradable, etc).
    """
    if isinstance(e, RetryError) and e.last_attempt is not None:
        try:
            inner = e.last_attempt.exception()
            if inner is not None:
                return f"{inner!r} (after retries exhausted)"
        except Exception:
            pass
    return str(e)


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

# Real closed trades for the transformer's replay retrain. Overridable so the
# test suite (tests/conftest.py) can point it at a temp file instead of the
# bot's real buffer.
LIVE_BUFFER_PATH = _os.environ.get("APEX_LIVE_BUFFER_PATH") or _os.path.join(
    _os.path.dirname(__file__), "..", "data", "live_experiences.jsonl"
)


async def _record_committee_outcome(
    symbol: str,
    exit_price: float,
    exit_reason: str | None = None,
    *,
    entry_price: float | None = None,
    qty: float | None = None,
    commission: float = 0.0,
) -> None:
    """On position exit, close the open decision snapshot and update the learner.

    ``entry_price``/``qty``: when supplied (callers have the exchange position
    in hand at close time), they override the snapshot values. The snapshot
    records the FIRST entry only, so for scale-in positions its qty/entry
    understate the true position -- the exchange's avg_entry_price and total
    qty are authoritative (audit finding F-A).

    ``commission``: round-trip commission actually charged, subtracted from
    realized PnL (audit finding F-B). Callers should also pass the real fill
    price as ``exit_price`` (not the pre-order signal price) so recorded PnL
    includes slippage.

    Fully fail-safe: realized-PnL bookkeeping for the adaptive layer must never
    interfere with trading. risk.py stays authoritative for the exit itself.
    """
    try:
        if not math.isfinite(exit_price) or exit_price <= 0.0:
            logger.warning(
                f"_record_committee_outcome: invalid exit_price {exit_price!r} for "
                f"{symbol} — skipping outcome recording to avoid corrupting "
                f"adaptive learner training data."
            )
            return

        from datetime import datetime

        from src.committee.committee import get_meta_learner
        from src.db import close_decision_snapshot, get_open_snapshot
        from src.metrics import alert_weight_change, update_adaptive_metrics

        snap = await asyncio.to_thread(get_open_snapshot, symbol)
        if not snap:
            return

        entry_price = float(entry_price) if entry_price is not None and entry_price > 0 \
            else float(snap.get("entry_price", 0.0))
        qty = abs(float(qty)) if qty is not None and qty != 0 else float(snap.get("qty", 0.0))
        action = snap.get("final_action", "buy")
        if entry_price <= 0 or qty == 0:
            return

        if action == "buy":
            realized_pnl = (exit_price - entry_price) * qty
        else:  # sell / short
            realized_pnl = (entry_price - exit_price) * qty

        # Fees are real money. ``commission`` is the fee on the EXIT fill the
        # caller just observed (exchange.py estimates it when Alpaca reports
        # none). A round trip also pays a fee on the ENTRY leg, which the exit
        # caller cannot see -- read it back from the orders ledger so the
        # snapshot's realized PnL is not entry-fee-blind. Both are estimates
        # when the exchange reports no commission (flagged commission_estimated
        # on the order record); we still subtract them so near-zero trades
        # don't get inflated win labels (audit finding F-B).
        exit_fee = abs(float(commission))
        entry_fee = 0.0
        try:
            from src.db import get_entry_fee_estimate
            # Match the entry fills by the snapshot's decision_id and SUM them
            # (a scale-in folds several buys into this one snapshot, each with
            # its own fee). Only fall back to the symbol/price heuristic when
            # there is no decision_id to key on.
            entry_fee = abs(float(await asyncio.to_thread(
                get_entry_fee_estimate,
                symbol.replace("/", ""),
                entry_price,
                snap.get("decision_id"),
            )))
        except Exception as fee_err:
            logger.debug(f"entry-fee lookup failed for {symbol} (non-fatal): {fee_err}")
        realized_pnl -= exit_fee + entry_fee

        # return_pct must be computed from the NET (post-commission) PnL, not
        # the gross price delta -- otherwise a trade that's a real loser after
        # fees (e.g. a $10 gross move eaten by a $15 commission, net -$5) can
        # report a POSITIVE return_pct, since the commission subtraction above
        # never touches it. This field feeds performance_tracker.py's
        # Sharpe/win-rate/decay-alert computations, strategy_selector.py's
        # adaptive-learner reward signal, AND track_record_status.py's
        # "positive expectancy" foundation-freeze gate -- all three would
        # silently see inflated, fee-blind performance without this fix.
        # Matches the net-pnl/notional convention already used correctly in
        # reconcile_open_snapshots() elsewhere in this file. Found via the
        # 2026-09-21 financial-correctness audit.
        notional = entry_price * qty
        return_pct = (realized_pnl / notional * 100.0) if notional > 0 else 0.0

        holding_sec = 0.0
        created = snap.get("created_at")
        if created:
            try:
                started = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                holding_sec = max(0.0, (datetime.now(UTC) - started).total_seconds())
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
        if _state.strategy is not None and hasattr(_state.strategy, '_trailing_troughs') and symbol in _state.strategy._trailing_troughs:
            trough = _state.strategy._trailing_troughs[symbol]
            if action == "buy":
                max_adv_pct = (entry_price - trough) / entry_price * 100.0 if trough < entry_price else 0.0
            else:
                max_adv_pct = (trough - entry_price) / entry_price * 100.0 if trough > entry_price else 0.0

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

        # Clear peak/trough price tracking for this symbol on any position close
        # to prevent stale values from affecting future positions. Locked for
        # consistency with the other peak_prices readers/writers in risk.py --
        # found unlocked via an external concurrency audit, 2026-09-22.
        if _state.risk_manager is not None:
            with _state.risk_manager._peak_prices_lock:
                _state.risk_manager.peak_prices.pop(symbol, None)
        if _state.strategy is not None and hasattr(_state.strategy, '_trailing_peaks'):
            _state.strategy._trailing_peaks.pop(symbol, None)
        if _state.strategy is not None and hasattr(_state.strategy, '_trailing_troughs'):
            _state.strategy._trailing_troughs.pop(symbol, None)

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
                live_buffer_path = LIVE_BUFFER_PATH
                os.makedirs(os.path.dirname(live_buffer_path), exist_ok=True)
                label = 1.0 if realized_pnl > 0 else 0.0
                entry = {"tensor": tensor_state, "label": label}
                # Entry time lets retrain_transformer.py order live trades
                # chronologically with the historical buffer.
                created = snap.get("created_at")
                if created is not None:
                    entry["entry_time"] = created.isoformat() if hasattr(created, "isoformat") else str(created)
                record = json.dumps(entry)

                def _append_live_experience():
                    with open(live_buffer_path, "a", encoding="utf-8") as f:
                        f.write(record + "\n")

                await asyncio.to_thread(_append_live_experience)
                logger.info(f"[LIVE_BUFFER] appended 1 record for {symbol} (label={label})")
            else:
                logger.debug(f"[LIVE_BUFFER] no tensor_state for {symbol}; trade not written to replay buffer")
        except Exception as e:
            logger.warning(f"[LIVE_BUFFER] failed to append live trade (non-fatal): {e}")

# Online Transformer gradient step: one step on the just-closed trade's
        # tensor state. This provides continuous learning between daily full
        # retrain_transformer.py runs. Fail-safe: runs in background thread,
        # never blocks trading, errors swallowed.
        try:
            tensor_state = snap.get("tensor_state")
            if tensor_state is not None:
                 def _online_transformer_step():
                     try:
                         import numpy as np
                         import torch

                         from src.committee.transformer_brain import get_ml_predictor
                         
                         predictor = get_ml_predictor()
                         if predictor is None:
                             return
                         
                         model = predictor["model"]
                         scaler = predictor["scaler"]
                         device = predictor["device"]

                         # tensor_state is ALREADY scaled: transformer_brain.py
                         # stores data_scaled.tolist(), the exact model input at
                         # prediction time, and retrain_transformer.py trains on
                         # it as-is. This used to call scaler.transform() on it
                         # again, which on a real stored record pushed values
                         # from about [-3, 4] out to about -285, so every
                         # online step trained the live model on inputs it
                         # never sees when predicting.
                         data = np.array(tensor_state, dtype=np.float32)
                         n_features = getattr(scaler, "n_features_in_", None)
                         if data.ndim == 2 and len(data) > 0 and (n_features is None or data.shape[1] == n_features):
                             data_scaled = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                             
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
                     except Exception as step_e:
                         logger.debug(f"Online Transformer gradient step failed (non-fatal): {step_e}")
                  

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

        # ─── Trigger Causal Attribution ───
        async def _run_attribution():
            try:
                from src.attribution import analyze_closed_trade
                res = await analyze_closed_trade(snap["decision_id"])
                if res:
                    logger.info(f"🧠 Causal Attribution for {symbol} ({action}):")
                    logger.info(f"   Success Factors: {res.get('success_factors')}")
                    logger.info(f"   Key Signals: {res.get('key_signals')}")
                    logger.info(f"   MVP Member: {res.get('mvp_member')}")
                    logger.info(f"   Robustness: {res.get('robustness_score')}")
                    logger.info(f"   Lessons: {res.get('lessons_learned')}")
                    
                    msg = (
                        f"🧠 <b>Trade Attribution: {symbol} ({action})</b>\n"
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
        self.ex: AlpacaExchange | None = None
        self.strategy: TradingStrategy | None = None
        self.risk_manager: RiskManager | None = None
        self.latest_scan_results: dict[str, dict] = {}
        self.scan_cycle_count: int = 0
        self.cooldowns: dict[str, float] = {}
        self.position_adds: dict[str, dict[str, Any]] = {}
        self._symbol_locks: dict[str, asyncio.Lock] = {}
        self._background_tasks: set[asyncio.Task] = set()
        self._shutdown_requested: bool = False
        # CR-12: set when startup/retry reconciliation couldn't fetch positions
        # at all -- blocks new entries (existing positions still exit/manage)
        # until a retry succeeds, rather than trading on an unknown state.
        self.reconciliation_incomplete: bool = False
        self._reconciliation_retry_active: bool = False
        # CR-2: symbols with a stale/unresolved order still open on the
        # exchange from before a crash (or still resolving normally) -- new
        # entries for these are blocked until the order fills or cancels.
        # Refreshed every main-loop cycle, not just at startup.
        self.symbols_with_unresolved_orders: set[str] = set()
        self._regime_flag_cache: dict[str, Any] = {}
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
        # threading.Lock, not asyncio.Lock: _online_transformer_step() (bot.py,
        # inside the online-learning gradient-step closure) runs via
        # asyncio.to_thread on a real OS thread, and uses plain sync `with`
        # around this lock, not `async with`. asyncio.Lock doesn't support the
        # sync context-manager protocol at all -- `with asyncio.Lock():` raises
        # `TypeError: 'Lock' object does not support the context manager
        # protocol` immediately, every call, which was silently swallowed by
        # the broad `except Exception` around the whole gradient step. Online
        # transformer learning has been a complete no-op, not merely
        # under-synchronized. Found via an external concurrency audit,
        # confirmed by reproducing the TypeError directly, 2026-09-22.
        self._transformer_online_lock: threading.Lock = threading.Lock()

        # Alerting metrics
        self.trade_timestamps: list[float] = []
        self.exchange_failure_count: int = 0

        # Per-symbol wall-clock time of the last successfully-placed order.
        # In-memory only (deliberately NOT persisted): after a restart the
        # positions are re-fetched fresh at startup, so the staleness window
        # this guards against (ENTRY_RACE_GUARD_SECONDS) resets naturally.
        # See the ENTRY_RACE_GUARD check in process_signal_for_symbol.
        self.last_fill_times: dict[str, float] = {}
        # Protective stop_limit order ids (resting exchange-side sells) keyed
        # by slash-less symbol. Armed on new entries when
        # PROTECTIVE_STOPS_ENABLED; every exit path cancels the entry here
        # first, or a dangling sell would linger after the position is gone.
        self.protective_stops: dict[str, str] = {}

    def get_regime_flag(self) -> dict[str, Any]:
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
            with open(_REGIME_FLAG_PATH) as f:
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
            with open(_BANNED_SYMBOLS_PATH) as f:
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
        self._transformer_online_updates = 0
        self._transformer_online_lr_schedule = "cosine"
        self._transformer_online_lr_base = 1e-5
        self._transformer_online_lr_min = 1e-6
        self._transformer_online_warmup_steps = 100
        self._transformer_online_total_steps = 10000
        self._transformer_online_lock = threading.Lock()

    def cleanup_stale_state(self, max_age_seconds: float = 3600) -> dict[str, int]:
        """Clean up stale state entries to prevent memory leaks.
        
        Args:
            max_age_seconds: Maximum age of entries before cleanup (default 1 hour)
            
        Returns:
            Dict with counts of cleaned entries per category
        """
        now = time.time()
        cleaned = {
            "cooldowns": 0,
            "position_adds": 0,
            "symbol_locks": 0,
            "scan_results": 0,
            "trade_timestamps": 0,
        }
        
        # Clean cooldowns (expired entries)
        expired_cooldowns = [k for k, v in self.cooldowns.items() if v < now]
        for k in expired_cooldowns:
            del self.cooldowns[k]
        cleaned["cooldowns"] = len(expired_cooldowns)
        
        # Clean position_adds entries stale for longer than max_age_seconds
        # (no new scale-in add in that window strongly implies the position
        # was closed through a path that didn't clear this entry -- e.g. a
        # manual close on the exchange, or a reconciliation-driven close --
        # since the explicit clear_position_adds()-on-close path is the ONLY
        # other cleanup this dict gets otherwise. Previously unbounded:
        # confirmed as a real gap 2026-09-21 cross-checking an external audit
        # against this code -- cooldowns/trade_timestamps already had TTL
        # cleanup below, position_adds did not.
        stale_adds = [
            k for k, v in self.position_adds.items()
            if now - v.get("last_add_time", 0) > max_age_seconds
        ]
        for k in stale_adds:
            del self.position_adds[k]
        cleaned["position_adds"] = len(stale_adds)

        # Clean symbol locks for symbols not in active trading
        # We keep locks for symbols in SYMBOLS config to avoid recreating
        active_symbols = set(settings.SYMBOLS)
        stale_locks = [k for k in self._symbol_locks.keys() if k not in active_symbols]
        for k in stale_locks:
            del self._symbol_locks[k]
        cleaned["symbol_locks"] = len(stale_locks)
        
        # Clean scan results for symbols not in active trading
        stale_results = [k for k in self.latest_scan_results.keys() if k not in active_symbols]
        for k in stale_results:
            del self.latest_scan_results[k]
        cleaned["scan_results"] = len(stale_results)
        
        # Clean trade timestamps older than max_age_seconds
        old_count = len(self.trade_timestamps)
        self.trade_timestamps = [ts for ts in self.trade_timestamps if now - ts < max_age_seconds]
        cleaned["trade_timestamps"] = old_count - len(self.trade_timestamps)
        
        if any(v > 0 for v in cleaned.values()):
            logger.debug(f"Cleaned stale state: {cleaned}")
        
        return cleaned


# Global state instance (single instance for the process)
_state = BotState()

# ADVERSARIAL_SCORE_FLOOR (the HIGH-disagreement veto floor) now lives in
# src/trade_decision.py with the rest of the entry rules shared with the
# backtester; re-exported here for existing references.
from src.trade_decision import (  # noqa: E402
    ADVERSARIAL_SCORE_FLOOR,
    adversarial_veto_reasons,
    apply_entry_size_multipliers,
    estimate_expected_return,
)

# Paths for config files
_REGIME_FLAG_PATH = "data/regime_flag.txt"
_BANNED_SYMBOLS_PATH = _os.path.join(_os.path.dirname(__file__), '..', 'data', 'banned_symbols.json')


def read_regime_flag():
    """Read the regime flag file, cached and only re-read when the file's mtime changes."""
    return _state.get_regime_flag()


def _stop_key(symbol: str) -> str:
    """Protective-stop bookkeeping is keyed by slash-less symbol (BTCUSD)."""
    return symbol.replace("/", "")


def _protstop_client_order_id(symbol: str, decision_id: str | None) -> str | None:
    """Deterministic client_order_id for a symbol's protective stop.

    Derived from the decision_id (not a timestamp) so the resting stop's id can
    be recomputed from the open snapshot after a restart -- reconcile then
    matches its fill by client_order_id instead of guessing by recency. Returns
    None when there is no decision_id, letting the caller fall back to a
    timestamped id.

    The decision_id is hashed rather than embedded: Alpaca caps
    client_order_id at 48 characters, and a raw decision_id plus the symbol
    prefix overflows that, which would make the exchange reject the stop.
    """
    if not decision_id:
        return None
    digest = hashlib.sha1(str(decision_id).encode("utf-8")).hexdigest()[:16]
    return f"{_stop_key(symbol)}_ps_{digest}"


async def _arm_protective_stop(
    ex, symbol: str, qty: float, stop_price: float, decision_id: str | None = None
) -> None:
    """Place an exchange-side protective stop_limit sell for a fresh long.

    Fully fail-safe: a rejected stop (bad distance, exchange hiccup) must not
    undo a fill that already happened -- it logs and returns, leaving the
    position to the bot's normal polled exits.

    No-op unless PROTECTIVE_STOPS_ENABLED, and requires an exchange adapter
    that implements submit_protective_stop (older fakes in tests may not).

    ``decision_id`` links the resting stop back to the entry's decision
    snapshot so its fill can later be attributed to the right position.
    """
    if not getattr(settings, "PROTECTIVE_STOPS_ENABLED", False):
        return
    submit = getattr(ex, "submit_protective_stop", None)
    if submit is None:
        return
    key = _stop_key(symbol)
    # Replace any stale stop for this symbol before arming a new one, so a
    # scale-in or a re-entry never leaves two resting sells for one position.
    await _cancel_protective_stop(ex, symbol)
    # Derive the client_order_id from the decision_id when we have one: it makes
    # the resting stop's id recoverable from the snapshot after a restart, so
    # reconcile can match its fill by client_order_id (see _protstop_client_order_id)
    # and the same decision re-arming the stop stays idempotent.
    client_order_id = _protstop_client_order_id(symbol, decision_id) or f"{key}_protstop_{int(time.time())}"
    try:
        limit_price = stop_price * (1.0 - float(settings.PROTECTIVE_STOP_LIMIT_OFFSET_PCT))
        info = await submit(
            symbol=symbol,
            qty=qty,
            stop_price=stop_price,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )
        if info and info.get("id"):
            _state.protective_stops[key] = str(info["id"])
            # Persist the resting stop in the order ledger (same schema the
            # entry/exit orders use) so a crash/restart can recover its real
            # fill by client_order_id instead of guessing from the symbol's
            # most-recent order. Fail-safe: a ledger write never blocks arming.
            # Persist the id the exchange actually RESTED (info's
            # client_order_id): on the duplicate-recovery path the adapter may
            # have re-armed under a fresh nonce, so recording our requested id
            # would put a client_order_id in the ledger that no live order has.
            # The decision_id is the reconcile key (tier 1) and is unaffected.
            await asyncio.to_thread(
                _persist_order_record,
                info,
                symbol,
                "sell",
                info.get("client_order_id") or client_order_id,
                decision_id=decision_id,
                order_type="stop_limit",
                time_in_force="gtc",
            )
    except Exception as stop_err:
        logger.warning(
            f"[PROTECTIVE_STOP] failed to arm for {symbol} qty={qty} "
            f"stop={stop_price:.2f} (position still managed by polled exits): {stop_err!r}"
        )


async def _cancel_protective_stop(ex, symbol: str) -> None:
    """Cancel and forget the resting protective stop for a symbol, if any.

    Called on EVERY exit path before/around the close order. Fail-safe: a
    failed cancel must not block the exit itself (the position is the real
    risk); it logs and clears local bookkeeping so we don't retry forever.
    """
    key = _stop_key(symbol)
    order_id = _state.protective_stops.pop(key, None)
    if not order_id:
        return
    cancel = getattr(ex, "cancel_order", None)
    if cancel is None:
        return
    try:
        await cancel(order_id)
    except Exception as cancel_err:
        logger.warning(
            f"[PROTECTIVE_STOP] failed to cancel {order_id} for {symbol} "
            f"before exit (manual check advised): {cancel_err!r}"
        )


# ── Crash-recovery state persistence (audit F1-F3) ───────────────────────────
# One debounced writer for the whole process. The main loop calls
# flush_crash_recovery_state() once per cycle; mutations elsewhere just set
# the dirty flag. Never throws -- persistence problems must not block trading.
_crash_state_writer = PersistentBotState(flush_interval=5.0)


async def flush_crash_recovery_state(force: bool = False) -> None:
    """Persist peak_prices / cooldowns / position_adds if (or when) dirty.

    The snapshot dict construction is pure in-memory dict copy (microseconds),
    but the actual JSON write is offloaded to a worker thread so the event
    loop is never blocked by disk I/O.
    """
    try:
        snapshot = {
            "peak_prices": _state.risk_manager.persist_peak_prices() if _state.risk_manager else {},
            "trailing_peaks": dict(getattr(_state.strategy, "_trailing_peaks", {}) or {}) if _state.strategy else {},
            "trailing_troughs": dict(getattr(_state.strategy, "_trailing_troughs", {}) or {}) if _state.strategy else {},
            "cooldowns": dict(_state.cooldowns),
            "position_adds": dict(_state.position_adds),
            "risk_peak_equity": _state.risk_manager.peak_equity if _state.risk_manager else 0.0,
            "risk_daily_pnl": _state.risk_manager.daily_pnl if _state.risk_manager else 0.0,
            "risk_start_of_day_equity": _state.risk_manager.start_of_day_equity if _state.risk_manager else 0.0,
            # UTC day the baseline above belongs to (last_check_time advances
            # at each day reset) -- restore discards a baseline from another day.
            "risk_start_of_day_date": _state.risk_manager.last_check_time.date().isoformat() if _state.risk_manager else "",
            # First max-drawdown breach time, so a restart doesn't restart the
            # flat-book cooldown clock (frequent redeploys would block forever).
            "risk_drawdown_tripped_at": (
                _state.risk_manager._drawdown_tripped_at.isoformat()
                if _state.risk_manager and _state.risk_manager._drawdown_tripped_at else ""
            ),
        }
        if _state.risk_manager is not None:
            _state.risk_manager.consume_peaks_dirty()
        if force or _crash_state_writer._dirty or (time.monotonic() - _crash_state_writer._last_flush >= _crash_state_writer.flush_interval):
            await asyncio.to_thread(_crash_state_writer.flush, snapshot)
    except Exception as e:
        logger.debug(f"Crash-recovery state flush skipped (non-fatal): {e}")


def apply_crash_recovery_state(recovery: dict[str, Any] | None) -> dict[str, int]:
    """Apply a persisted crash-recovery snapshot onto the live bot state.

    Restores peak_prices / trailing peaks / cooldowns / position_adds and the
    risk-manager equity/PNL baselines so trailing stops, entry cooldowns, the
    scale-in cap, and the drawdown killswitch all survive a restart instead of
    silently resetting. Fail-safe: never raises, returns per-key restore counts.
    """
    counts = {
        "peak_prices": 0,
        "trailing_peaks": 0,
        "trailing_troughs": 0,
        "cooldowns": 0,
        "position_adds": 0,
    }
    if not recovery:
        return counts
    try:
        if _state.risk_manager is not None:
            if "peak_prices" in recovery:
                _state.risk_manager.peak_prices.update(recovery["peak_prices"])
                counts["peak_prices"] = len(recovery["peak_prices"])
            # peak equity -- prevents drawdown reset to 0% after crash
            # (without this, a 5% drop + crash + restart = killswitch thinks
            #  equity is at peak and won't trip at 10% drawdown)
            if recovery.get("risk_peak_equity", 0) > 0:
                _state.risk_manager.peak_equity = recovery["risk_peak_equity"]
            if recovery.get("risk_drawdown_tripped_at"):
                try:
                    _state.risk_manager._drawdown_tripped_at = datetime.fromisoformat(
                        recovery["risk_drawdown_tripped_at"]
                    )
                except (TypeError, ValueError):
                    pass  # malformed -> breach re-stamps on the next status update
            # Daily-loss baseline: only valid for the UTC day it was taken.
            # RiskManager.last_check_time starts at "now", so a restored
            # baseline from an earlier day would never hit the day reset and
            # multi-day losses would count as today's (tripped the daily-loss
            # killswitch 1s after a 2026-10-04 restart with no trades). Undated
            # legacy snapshots can't prove they're from today, so skip them;
            # the first status update re-baselines to current equity.
            today = datetime.now(UTC).date().isoformat()
            if recovery.get("risk_start_of_day_date") == today:
                if "risk_daily_pnl" in recovery:
                    _state.risk_manager.daily_pnl = recovery["risk_daily_pnl"]
                if recovery.get("risk_start_of_day_equity", 0) > 0:
                    _state.risk_manager.start_of_day_equity = recovery["risk_start_of_day_equity"]
        if _state.strategy is not None:
            if "trailing_peaks" in recovery:
                _state.strategy._trailing_peaks.update(recovery["trailing_peaks"])
                counts["trailing_peaks"] = len(recovery["trailing_peaks"])
            if "trailing_troughs" in recovery:
                _state.strategy._trailing_troughs.update(recovery["trailing_troughs"])
                counts["trailing_troughs"] = len(recovery["trailing_troughs"])
        if "cooldowns" in recovery:
            _state.cooldowns.update(recovery["cooldowns"])
            counts["cooldowns"] = len(recovery["cooldowns"])
        if "position_adds" in recovery:
            _state.position_adds.update(recovery["position_adds"])
            counts["position_adds"] = len(recovery["position_adds"])
    except Exception as restore_err:
        logger.warning(f"Could not restore crash-recovery state (non-fatal): {restore_err}")
    return counts


async def crash_state_flush_heartbeat_loop() -> None:
    """Background heartbeat that actually delivers on PersistentBotState's
    documented "bounds the worst-case loss to flush_interval seconds"
    promise (CR-5b).

    Before this, flush_crash_recovery_state() was only ever called once per
    main-loop cycle (LOOP_INTERVAL_SEC, default 60s) -- so a peak/cooldown/
    position-add update that happens mid-cycle (trailing peaks in particular
    update on every price tick, with no mark_dirty() hook at all, by design)
    wasn't persisted until the cycle boundary. A hard kill in that window
    lost up to ~60s of state, not the 5s _crash_state_writer.flush_interval
    implies. Reproduced via simulation (peak 120->130, hard kill before the
    next flush, restored=120) verifying an external crash-recovery audit,
    confirmed by tracing flush_crash_recovery_state()'s only caller being
    the once-per-cycle main loop, 2026-09-22.
    """
    while not _state._shutdown_requested:
        await asyncio.sleep(_crash_state_writer.flush_interval)
        try:
            await flush_crash_recovery_state()
        except Exception as e:
            logger.debug(f"Crash-state flush heartbeat tick failed (non-fatal): {e}")


async def _close_orphan_position(exchange: AlpacaExchange, symbol_clean: str, positions: list[dict[str, Any]]) -> None:
    """Close an exchange position the bot has no DB snapshot for.

    Called during startup reconciliation to free position slots consumed by
    crash-gap orphan positions. Locates the position, flips the qty sign, and
    submits a market order to flatten it.
    """
    try:
        position = next((p for p in positions if p["symbol"].replace("/", "") == symbol_clean), None)
        if position is None:
            return

        qty = abs(float(position.get("qty", 0)))
        if qty <= 0:
            return

        side = "sell" if float(position.get("qty", 0)) > 0 else "buy"
        raw_symbol = position["symbol"]
        logger.warning(f"[RECONCILE] Closing orphan {raw_symbol} position: qty={qty}, side={side}")

        await exchange.create_order(raw_symbol, qty=qty, side=side, bypass_circuit_breaker=True)
        logger.warning(f"[RECONCILE] Orphan position {symbol_clean} closed (qty={qty})")
    except Exception as e:
        logger.error(f"[RECONCILE] Failed to close orphan position {symbol_clean}: {e}")


async def reconcile_open_snapshots(exchange: AlpacaExchange) -> None:
    """Startup reconciliation pass (audit F4).

    After a crash or an out-of-bot close (manual close on the exchange while
    the process was down), decision snapshots can be left status='open'
    forever. Compare every open snapshot against the exchange's actual
    positions:
      - snapshot for a symbol with NO exchange position -> the position was
        closed outside the bot; close the snapshot with the last known price
        so the DB doesn't accumulate ghosts and the adaptive meta-learner
        still sees the outcome.
      - exchange position with NO open snapshot (restart gap) -> warn.
    Fully fail-safe: any error is logged and skipped.
    """
    from src.db import close_decision_snapshot, get_all_open_snapshots
    try:
        open_snaps = await asyncio.to_thread(get_all_open_snapshots)
    except Exception as e:
        # Same fail-safe treatment as a positions-fetch failure (CR-12) --
        # without knowing what's locally recorded as open, this function
        # can't tell a genuine orphan from a crash-gap position, or a
        # genuine ghost from a real one. get_all_open_snapshots() used to
        # swallow this and return [], making a DB read failure
        # indistinguishable from "genuinely nothing is open" -- reconcile
        # silently skipped both the ghost-close and orphan-check passes
        # with no signal anything was wrong. Found via an external
        # crash-recovery audit, 2026-09-22.
        logger.critical(f"[RECONCILE] Could not read open snapshots from the DB ({e}). "
                        f"Blocking new entries until a reconciliation retry succeeds.")
        _state.reconciliation_incomplete = True
        try:
            await get_alerting_engine().alert_system_health(
                "startup_reconciliation", "down", {"error": str(e), "stage": "get_all_open_snapshots"}
            )
        except Exception as alert_err:
            logger.debug(f"Reconciliation-failure alert skipped (non-fatal): {alert_err}")
        if not _state._reconciliation_retry_active:
            _state._reconciliation_retry_active = True
            task = asyncio.create_task(_retry_reconciliation_until_success(exchange))
            _state._background_tasks.add(task)
            task.add_done_callback(_state._background_tasks.discard)
        return

    positions_fetch_failed = False
    try:
        positions = await exchange.get_positions()
        _state.reconciliation_incomplete = False
    except Exception as e:
        # Fail-safe UNKNOWN state, not flat. Substituting positions=[] here
        # used to fall through into the ghost-close loop below with an empty
        # held_symbols set, which matches nothing -- every open snapshot got
        # closed as a "ghost" while the exchange might still hold every one
        # of those positions for real. Confirmed via simulation 2026-09-21
        # verifying an external crash-recovery audit: 1 open snapshot -> 0
        # after a single fetch failure. Skip the position-vs-snapshot
        # reconciliation entirely and leave snapshots untouched. The
        # stale-order check further down is independent of get_positions()
        # and still runs.
        logger.critical(
            f"[RECONCILE] Position-vs-snapshot reconciliation ABORTED: could not fetch "
            f"positions ({e}). Leaving all {len(open_snaps)} open snapshot(s) untouched "
            f"rather than treating an unknown exchange state as flat. Blocking new "
            f"entries until a reconciliation retry succeeds."
        )
        positions_fetch_failed = True
        positions = []
        _state.reconciliation_incomplete = True
        try:
            await get_alerting_engine().alert_system_health(
                "startup_reconciliation", "down", {"error": str(e), "open_snapshots": len(open_snaps)}
            )
        except Exception as alert_err:
            logger.debug(f"Reconciliation-failure alert skipped (non-fatal): {alert_err}")
        # Guarded so a retry attempt's own failure (which re-enters this same
        # except block) can't spawn a second, nested retry loop on top of the
        # one already running.
        if not _state._reconciliation_retry_active:
            _state._reconciliation_retry_active = True
            task = asyncio.create_task(_retry_reconciliation_until_success(exchange))
            _state._background_tasks.add(task)
            task.add_done_callback(_state._background_tasks.discard)

    held_symbols = {p["symbol"].replace("/", "") for p in positions} if positions else set()

    if positions_fetch_failed:
        # Can't tell a genuine ghost from a real position we just can't see --
        # skip both position-comparison passes below entirely.
        pass
    else:
        # Inverse check: exchange positions the bot has no snapshot for.
        # This handles crash-gap orphan positions that block the position slot limit.
        snap_symbols = {s["symbol"].replace("/", "") for s in open_snaps} if open_snaps else set()
        for held in sorted(held_symbols):
            if held not in snap_symbols:
                held_pos = next((p for p in positions if p["symbol"].replace("/", "") == held), None)
                held_qty = float(held_pos.get("qty", 0)) if held_pos else 0.0
                # CR-1: before liquidating, check the order ledger for a
                # recent order on this symbol -- a fill that landed but
                # crashed before save_decision_snapshot() completed looks
                # identical to a genuinely pre-existing/manual position from
                # here. Re-attach a snapshot instead of force-closing a
                # position the bot itself just opened. Found via an external
                # crash-recovery audit, confirmed by reproduction, 2026-09-22.
                recent_order = await asyncio.to_thread(_find_recent_order_for_symbol, held)
                if recent_order is not None:
                    logger.warning(
                        f"[RECONCILE] Exchange position in {held} (qty={held_qty}) has no "
                        f"snapshot, but order ledger shows a recent {recent_order['side']} "
                        f"order {recent_order['order_id']} for this symbol -- re-attaching "
                        f"a snapshot instead of closing (crash-gap fill, not a true orphan)."
                    )
                    await asyncio.to_thread(
                        _reattach_snapshot_from_order, recent_order, held_qty
                    )
                    continue
                logger.critical(
                    f"[RECONCILE] Exchange reports an open position in {held} with no open "
                    f"decision snapshot and no recent order record (opened before/outside "
                    f"this process). Attempting to close to free position slot."
                )
                try:
                    await get_alerting_engine().alert_position_desync(held, expected_qty=0.0, actual_qty=held_qty)
                except Exception as alert_err:
                    logger.debug(f"Position-desync alert skipped (non-fatal): {alert_err}")
                await _close_orphan_position(exchange, held, positions)

        # Ledger refresh for PROTECTIVE-STOP fills, run BEFORE the ghost-close
        # loop below. _arm_protective_stop writes the resting stop's ledger row
        # once, at arm time, while the stop is still ``new`` (status="new",
        # filled_avg_price=0). A stop that later TRIGGERS fills server-side with
        # no bot code in the path, so its ledger row keeps the stale "new"/0
        # fill -- the ghost-close loop's exit lookup then sees filled_avg_price
        # 0, drops it, and falls back to the last bar (a fabricated exit price
        # for the learner, the exact class of bug CR-6 fixed for the normal exit
        # paths). Refresh every stale stop from the exchange first so its real
        # fill is on record before the ghost-close loop reads the ledger.
        #
        # Candidates come from the ORDERS LEDGER, not _state.protective_stops:
        # that registry is in-memory and EMPTY after a restart, and reconcile
        # runs at startup -- before any cycle repopulates it -- so keying off it
        # would skip exactly the restart case this pass exists for. This also
        # covers a stop whose position is no longer held (it filled and
        # flattened the position). Fail-safe: any error is logged and skipped.
        try:
            from src.db import get_open_snapshot, get_stale_protective_stops
            from src.exchange import LIVE_ORDER_STATUSES
            stale_stops = await asyncio.to_thread(get_stale_protective_stops)
            for rec in stale_stops:
                order_id = rec.get("order_id")
                o_sym = str(rec.get("symbol", "")).replace("/", "")
                if not order_id:
                    continue
                try:
                    fresh = await exchange.get_order(str(order_id))
                except Exception as order_err:
                    logger.debug(f"[RECONCILE] Could not refresh protective stop {order_id} (non-fatal): {order_err}")
                    continue
                if not isinstance(fresh, dict):
                    continue
                status = str(fresh.get("status", "")).lower()
                if status in LIVE_ORDER_STATUSES or float(fresh.get("filled_qty", 0) or 0) <= 0:
                    continue
                # It filled while we were down/polling: persist the real fill so
                # the ghost-close loop (and a later restart) matches it by
                # decision_id. Alpaca reports no commission, so fill in the
                # estimate (same as the entry/exit paths) so the stop leg's fee
                # is on record too.
                fresh = AlpacaExchange._apply_estimated_commission(fresh)
                # Prefer the decision_id already on the ledger row (the arm-time
                # key); only if it is missing, look one up from the open snapshot
                # for the stored symbol.
                decision_id = rec.get("decision_id")
                if not decision_id:
                    try:
                        snap = await asyncio.to_thread(get_open_snapshot, rec.get("symbol") or o_sym)
                        decision_id = snap.get("decision_id") if snap else None
                    except Exception as snap_err:
                        logger.debug(f"[RECONCILE] snapshot lookup for stop refresh failed (non-fatal): {snap_err}")
                await asyncio.to_thread(
                    _persist_order_record,
                    fresh,
                    o_sym,
                    "sell",
                    fresh.get("client_order_id") or rec.get("client_order_id"),
                    decision_id,
                    "stop_limit",
                    "gtc",
                )
                logger.warning(
                    f"[RECONCILE] Protective stop {order_id} for {o_sym} filled "
                    f"({status}) -- ledger updated with its real fill price."
                )
        except Exception as e:
            logger.debug(f"[RECONCILE] Protective-stop ledger refresh skipped (non-fatal): {e}")

        # NOTE: no early `return` here when open_snaps is empty -- the loop below
        # is already a no-op on an empty list, and an early return would skip the
        # stale-open-order check further down too. That's not hypothetical: found
        # via simulation 2026-09-21 that a crash occurring during/before order
        # recording (i.e. before save_decision_snapshot() ever ran) leaves ZERO
        # open snapshots in the DB -- exactly the scenario the stale-order check
        # exists to catch -- while a real order can still be sitting on the
        # exchange. An earlier version of this function returned here, silently
        # disabling that check in precisely the case it was built for.
        for snap in open_snaps:
            sym = snap["symbol"]
            sym_clean = sym.replace("/", "")
            if sym_clean in held_symbols:
                continue  # genuinely open position -> snapshot is correct
            exit_price = 0.0
            exit_price_source = "estimated_last_bar"
            action = snap.get("final_action", "buy")
            # CR-6: prefer the actual fill price from the order ledger over
            # the last bar -- the last bar is the CURRENT price at
            # reconciliation time, not the price the position was actually
            # closed at, so it silently mislabels the adaptive learner's
            # training sample. Found via an external crash-recovery audit,
            # confirmed by reproduction (estimated $16 pnl vs true $100 pnl
            # in the audit's own scenario), 2026-09-22.
            # Use the fill that actually CLOSED the position, not simply the
            # symbol's most-recent order: a re-entry or a resting protective
            # stop can leave a newer row, and picking it would attribute the
            # wrong exit price to this snapshot. Prefer the order recorded for
            # this snapshot's decision_id, else the latest fill on the closing
            # side (a long is closed by a sell, a short by a buy).
            exit_side = "sell" if action == "buy" else "buy"
            # Reconcile by client_order_id as well as decision_id: the
            # protective stop's id is derived from the snapshot's decision_id
            # (_protstop_client_order_id), so even a stop fill whose ledger row
            # lost its decision_id is still matched to the right snapshot rather
            # than falling back to the newest unrelated sell.
            protstop_coid = _protstop_client_order_id(sym, snap["decision_id"])
            exit_order = await asyncio.to_thread(
                _find_recent_exit_fill,
                sym_clean,
                snap["decision_id"],
                exit_side,
                3600.0,
                protstop_coid,
            )
            if exit_order is not None:
                exit_price = float(exit_order["filled_avg_price"])
                exit_price_source = "actual_fill"
            else:
                try:
                    bars = await exchange.get_latest_bar(sym)
                    if not bars.is_empty():
                        exit_price = float(bars["close"][0])
                except Exception as e:
                    logger.debug(f"[RECONCILE] No price available for {sym}: {e}")
            entry_price = float(snap.get("entry_price", 0.0))
            qty = float(snap.get("qty", 0.0))
            if exit_price > 0 and entry_price > 0 and qty != 0:
                pnl = (exit_price - entry_price) * qty if action == "buy" else (entry_price - exit_price) * qty
            else:
                pnl = 0.0
            closed = await asyncio.to_thread(
                close_decision_snapshot,
                snap["decision_id"],
                realized_pnl=pnl,
                return_pct=(pnl / (entry_price * qty) * 100.0) if (entry_price > 0 and qty != 0 and pnl is not None) else 0.0,
                exit_reason=f"reconciled_after_restart:{exit_price_source}",
            )
            logger.warning(
                f"[RECONCILE] Closed ghost snapshot {snap['decision_id']} for {sym} "
                f"(no exchange position; exit_price={exit_price} [{exit_price_source}], "
                f"pnl={pnl:.2f}, closed={closed})"
            )
            try:
                await get_alerting_engine().alert_position_desync(sym, expected_qty=qty, actual_qty=0.0)
            except Exception as alert_err:
                logger.debug(f"Position-desync alert skipped (non-fatal): {alert_err}")

    # Check for stale open orders from a crashed cycle. If the bot submitted an
    # order to the exchange but crashed before recording the response, the order
    # may still be open on the exchange. We cannot cancel it safely (partial
    # fills would leave the bot with an untracked position), so we log a warning
    # so the operator can review -- and, critically, block new entries for that
    # symbol until it resolves (fills or gets cancelled). Without this, a fresh
    # signal for the same symbol re-fetches positions (still empty while the
    # stale order is unfilled), finds no snapshot, and submits a SECOND buy --
    # 2x intended exposure from one signal. Found via an external crash-recovery
    # audit, confirmed by reproduction, 2026-09-22. _state.symbols_with_unresolved_orders
    # is refreshed every cycle in the main loop (not just at startup), so a
    # symbol unblocks itself once the order actually resolves.
    try:
        open_orders = await exchange.get_orders(status="new", limit=100)
        if open_orders:
            unresolved = set()
            for order in open_orders:
                o_sym = order.get("symbol", "?").replace("/", "")
                unresolved.add(o_sym)
                logger.warning(
                    f"[RECONCILE] Stale open order {order.get('id', '?')} for {o_sym} "
                    f"(status={order.get('status')}, qty={order.get('qty')}, "
                    f"filled_qty={order.get('filled_qty')}) — submitted before crash. "
                    f"Blocking new entries for {o_sym} until this order resolves."
                )
            _state.symbols_with_unresolved_orders |= unresolved
    except Exception as e:
        logger.debug(f"[RECONCILE] Could not fetch open orders (non-fatal): {e}")

    # Protective-stop reconciliation. A restart loses _state.protective_stops,
    # so two things must be fixed up or the bot silently loses its tail-risk
    # protection (and can leak resting sells):
    #   1. Re-adopt resting stop_limit sells for still-held symbols (the
    #      process may have restarted between arming and the position's exit)
    #      so a later exit can cancel them -- otherwise they linger forever.
    #   2. Cancel a stop_limit sell that has NO matching position (its
    #      position was closed while we were down) -- a dangling sell.
    # Only active when PROTECTIVE_STOPS_ENABLED; fail-safe on any error.
    if getattr(settings, "PROTECTIVE_STOPS_ENABLED", False) and not positions_fetch_failed:
        try:
            open_orders = await exchange.get_orders(status="new", limit=100)
            for order in open_orders:
                o_sym = str(order.get("symbol", "?")).replace("/", "")
                if order.get("type") != "stop_limit" or order.get("side") != "sell":
                    continue
                if o_sym in held_symbols:
                    _state.protective_stops[o_sym] = str(order["id"])
                    logger.info(
                        f"[RECONCILE] Re-adopted resting protective stop "
                        f"{order['id']} for held position {o_sym}"
                    )
                else:
                    logger.warning(
                        f"[RECONCILE] Dangling protective stop {order['id']} for "
                        f"{o_sym} with no open position -- cancelling."
                    )
                    try:
                        await exchange.cancel_order(str(order["id"]))
                    except Exception as cancel_err:
                        logger.warning(
                            f"[RECONCILE] Could not cancel dangling stop "
                            f"{order['id']} for {o_sym}: {cancel_err!r}"
                        )
        except Exception as e:
            logger.debug(f"[RECONCILE] Protective-stop reconciliation skipped (non-fatal): {e}")


def _parse_order_timestamp(value: Any) -> datetime | None:
    """Best-effort parse of an order timestamp (ISO str or datetime) for the
    ledger's ``filled_at`` column. Returns None on anything unparseable so a
    malformed exchange field never blocks the ledger write."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _persist_order_record(
    order_result: dict,
    symbol: str,
    side: str,
    client_order_id: str | None,
    decision_id: str | None = None,
    order_type: str = "market",
    time_in_force: str = "ioc",
) -> None:
    """Fire-and-forget write to the order ledger (CR-3). save_order_record()
    was defined but had zero callers -- after a crash, the bot had no local
    record of order ids/client_order_ids/fills to reconcile against, only
    whatever it could re-derive from exchange positions. Also feeds CR-1
    (re-attaching a crash-gap fill instead of liquidating it as an orphan)
    and CR-6 (using the real fill price instead of estimating from a bar).
    Found via an external crash-recovery audit, confirmed by AST scan
    showing zero call sites outside db.py, 2026-09-22.

    ``order_type``/``time_in_force`` default to the market/ioc entry/exit
    convention; the protective stop_limit path passes stop_limit/gtc so the
    ledger (and a later reconcile) can tell the resting sell apart.
    """
    from src.db import save_order_record
    try:
        save_order_record(
            order_id=str(order_result.get("id", client_order_id or "unknown")),
            decision_id=decision_id,
            symbol=symbol,
            side=side,
            qty=float(order_result.get("qty", 0.0) or 0.0),
            filled_qty=float(order_result.get("filled_qty", 0.0) or 0.0),
            filled_avg_price=float(order_result.get("filled_avg_price", 0.0) or 0.0),
            commission=float(order_result.get("commission", 0.0) or 0.0),
            status=str(order_result.get("status", "unknown")),
            type=order_type,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            filled_at=_parse_order_timestamp(order_result.get("filled_at")),
        )
    except Exception as e:
        logger.debug(f"Order-ledger write failed for {symbol} (non-fatal): {e}")


async def _snapshot_decision_id(symbol: str) -> str | None:
    """Return the open decision snapshot's id for *symbol*, or None.

    Exit orders (trailing stop, strategy close, killswitch flatten) are the
    only orders besides the protective stop that should carry the snapshot's
    decision_id: without it the ledger row can't be tied to the trade it
    closed, so a restart/reconcile has to guess by recency (the bug that fed
    the adaptive learner a wrong exit price). Fail-safe: any lookup error
    returns None, which the reconcile fallback handles.
    """
    try:
        from src.db import get_open_snapshot
        snap = await asyncio.to_thread(get_open_snapshot, symbol)
        return snap.get("decision_id") if snap else None
    except Exception as e:
        logger.debug(f"[ORDER] snapshot lookup failed for {symbol} (non-fatal): {e}")
        return None


async def _persist_exit_order(
    order_result: dict,
    symbol: str,
    side: str,
    client_order_id: str | None,
    decision_id: str | None = None,
) -> None:
    """Record an exit order in the ledger with the snapshot's decision_id and
    the exchange's filled_at, so reconcile can match the real exit later."""
    await asyncio.to_thread(
        _persist_order_record, order_result, symbol, side, client_order_id, decision_id
    )


def _find_recent_order_for_symbol(symbol_clean: str, max_age_sec: float = 3600.0) -> dict | None:
    """Look up the most recent order-ledger record for a symbol (CR-1/CR-6).

    Used by reconcile_open_snapshots() to tell a genuine crash-gap fill (the
    bot's own order landed but the process died before save_decision_snapshot()
    ran) apart from a truly pre-existing/manually-opened position. Bounded to
    the last hour so a stale ledger entry from long ago can't misattribute an
    unrelated position.

    Protective-stop rows (type=stop_limit) are skipped: a resting sell is not
    the entry that opened the position, and since the stop ledger write was
    added it can be the symbol's most-recent row -- returning it would reattach
    a snapshot with side="sell" and a zero entry price.
    """
    from src.db import get_recent_order_records
    try:
        candidates = get_recent_order_records(symbol_clean, max_age_sec=max_age_sec)
        entries = [c for c in candidates if str(c.get("type", "")).lower() != "stop_limit"]
        return entries[0] if entries else None
    except Exception as e:
        logger.debug(f"[RECONCILE] Order-ledger lookup failed for {symbol_clean} (non-fatal): {e}")
        return None


def _record_filled_at_ms(record: dict) -> float:
    """Sort key for a ledger record's real exit time: ``filled_at`` when the
    exchange supplied it, else the submission time. Returns -inf when neither
    is present so an un-timestamped record sorts as oldest, never newest."""
    ts = _parse_order_timestamp(record.get("filled_at"))
    if ts is None:
        ts = _parse_order_timestamp(record.get("submitted_at"))
    if ts is None:
        return float("-inf")
    try:
        return ts.timestamp()
    except Exception:
        return float("-inf")


def _select_exit_fill(
    candidates: list[dict],
    decision_id: str | None = None,
    exit_side: str = "sell",
    client_order_id: str | None = None,
) -> dict | None:
    """Pick the exit fill that actually closed a position from the ledger.

    The symbol's most-recent order is NOT necessarily the one that closed the
    position: a re-entry after the exit, or a protective stop resting while a
    later market sell fires, can both leave a newer ledger row. Choosing wrong
    feeds the adaptive learner a fabricated exit price (the pre-fix bug: one
    ghost-close used the last bar, the other trusted ``records[0]``).

    Selection order, most-specific first:
      1. ``decision_id`` match -- the exact order recorded for this snapshot.
      2. ``client_order_id`` match -- used when the snapshot's decision_id has
         no ledger row (the record was lost) but the exit order's id is known.
      3. the LATEST filled order on ``exit_side`` -- a long is closed by a
         sell (the default); latest-by-``filled_at`` breaks ties.
    Returns None when no such fill exists, so the caller falls back to the bar.
    """
    fills = [
        c for c in candidates
        if str(c.get("side", "")).lower() == exit_side
        and float(c.get("filled_avg_price", 0.0) or 0.0) > 0
    ]
    if not fills:
        return None
    if decision_id:
        for c in fills:
            if c.get("decision_id") == decision_id:
                return c
    if client_order_id:
        for c in fills:
            if c.get("client_order_id") == client_order_id:
                return c
    return max(fills, key=_record_filled_at_ms)


def _find_recent_exit_fill(
    symbol_clean: str,
    decision_id: str | None = None,
    exit_side: str = "sell",
    max_age_sec: float = 3600.0,
    client_order_id: str | None = None,
) -> dict | None:
    """Ledger lookup for the exit that closed a symbol's position (CR-6).

    Unlike ``_find_recent_order_for_symbol`` (which returns the single most
    recent record, entry or exit), this returns the fill that closed the
    position, preferring the order recorded for ``decision_id`` (or, failing
    that, one whose ``client_order_id`` matches), and otherwise the latest fill
    on ``exit_side`` (a long is closed by a sell). A filled protective
    stop_limit sell is itself a filled sell, so a stop-triggered exit is
    recovered the same way. Fail-safe: any error returns None so the caller
    falls back to the last bar.
    """
    from src.db import get_recent_order_records
    try:
        candidates = get_recent_order_records(symbol_clean, max_age_sec=max_age_sec)
        return _select_exit_fill(candidates, decision_id, exit_side, client_order_id)
    except Exception as e:
        logger.debug(f"[RECONCILE] Exit-fill lookup failed for {symbol_clean} (non-fatal): {e}")
        return None


def _reattach_snapshot_from_order(order: dict, actual_qty: float) -> None:
    """Recreate a decision snapshot from an order-ledger record (CR-1) so a
    crash-gap position (fill landed, save_decision_snapshot() never ran)
    stays tracked by the adaptive meta-learner instead of being force-closed.
    """
    from src.db import save_decision_snapshot
    try:
        save_decision_snapshot(
            decision_id=order.get("decision_id") or f"reattached_{order['order_id']}",
            symbol=order["symbol"],
            regime="unknown",
            final_action=order.get("side", "buy"),
            confidence=0.0,
            size_multiplier=1.0,
            entry_price=float(order.get("filled_avg_price", 0.0)),
            qty=actual_qty,
            brain_votes={},
        )
    except Exception as e:
        logger.warning(f"[RECONCILE] Could not re-attach snapshot for {order.get('symbol')}: {e}")


async def _retry_reconciliation_until_success(exchange: AlpacaExchange) -> None:
    """Keep retrying startup reconciliation after a positions-fetch failure
    (CR-12) until it succeeds, instead of leaving `_state.reconciliation_incomplete`
    (which blocks new entries) stuck True for the rest of the process lifetime.
    """
    try:
        while _state.reconciliation_incomplete and not _state._shutdown_requested:
            await asyncio.sleep(settings.KILLSWITCH_CHECK_INTERVAL_SEC)
            try:
                await reconcile_open_snapshots(exchange)
            except Exception as e:
                logger.warning(f"[RECONCILE] Retry attempt failed (non-fatal, will retry again): {e}")
    finally:
        _state._reconciliation_retry_active = False


def _prune_stale_restored_state(held_symbols: set[str]) -> tuple[list[str], list[str]]:
    """Drop restored peak_prices/_trailing_peaks/_trailing_troughs/
    position_adds entries for symbols that AREN'T in `held_symbols` (CR-C1).

    The crash-recovery state restore applies the persisted file's peaks
    unconditionally, with no cross-check against real positions. If a
    position was closed while the bot was down (manually, or via some other
    out-of-band path) but the symbol is re-entered later at a different
    price, check_trailing_stop() keys purely on symbol -- it can't tell
    "this is a stale peak from the last trade" from "this is this trade's
    own peak", so the OLD peak silently governs the NEW position. Concrete
    failure traced through the actual trailing-stop math: old peak 120 from
    a closed trade, new entry at 105, price ticks to 116 (a normal, healthy
    move for the NEW position, nowhere near ITS OWN stop) -- but drawdown-
    from-the-STALE-peak is (120-116)/120 = 3.3%, which can exceed the
    trigger distance and close a fresh position within minutes of opening
    it. position_adds has the same hazard (a stale scale-in count/timer
    from the old trade wrongly gating the new one's own scale-ins).

    cooldowns is deliberately NOT pruned here -- it exists specifically to
    block re-entry into a symbol that is NOT currently held, so
    intersecting it with held_symbols would erase every cooldown on every
    restart.

    Returns (dropped_peak_symbols, dropped_position_adds_symbols).

    Found via an external crash-recovery audit, confirmed by tracing
    check_trailing_stop's actual peak-comparison logic, 2026-09-22.
    """
    stale_peaks = []
    if _state.risk_manager is not None:
        stale_peaks = [s for s in _state.risk_manager.peak_prices if s.replace("/", "") not in held_symbols]
        for s in stale_peaks:
            del _state.risk_manager.peak_prices[s]
    if _state.strategy is not None:
        for attr in ("_trailing_peaks", "_trailing_troughs"):
            d = getattr(_state.strategy, attr, None)
            if d:
                for s in [s for s in d if s.replace("/", "") not in held_symbols]:
                    del d[s]
    stale_adds = [s for s in _state.position_adds if s.replace("/", "") not in held_symbols]
    for s in stale_adds:
        del _state.position_adds[s]
    if stale_peaks or stale_adds:
        logger.warning(
            f"Dropped restored peak/trailing/position_adds entries for symbols "
            f"not currently held (stale from a previous, since-closed trade): "
            f"peaks={stale_peaks}, position_adds={stale_adds}"
        )
    return stale_peaks, stale_adds


def get_banned_symbols():
    """Read the banned symbols list generated by the weekly analyzer, cached."""
    return _state.get_banned_symbols()


def _count_same_regime_open_positions(symbol: str, positions: list, strategy: TradingStrategy | None) -> int:
    """Count currently-open positions (excluding `symbol` itself) classified
    in the same regime as `symbol`, using TradingStrategy._regime_cache as a
    cheap proxy for correlated exposure (real return-series correlation
    isn't wired up anywhere live). Alpaca position symbols come back without
    the slash (e.g. "BTCUSD"); the regime cache is keyed with it (e.g.
    "BTC/USD", matching settings.SYMBOLS) -- reverse-map before comparing.
    """
    if strategy is None or not positions:
        return 0
    candidate_cached = strategy._regime_cache.get(symbol)
    candidate_regime = candidate_cached[1].get("regime") if candidate_cached else None
    if not candidate_regime:
        return 0
    symbol_lookup = {s.replace("/", ""): s for s in settings.SYMBOLS}
    candidate_raw = symbol.replace("/", "")
    count = 0
    for p in positions:
        raw_symbol = p.get("symbol", "")
        if not raw_symbol or raw_symbol == candidate_raw:
            continue
        slashed = symbol_lookup.get(raw_symbol)
        if not slashed:
            continue
        other_cached = strategy._regime_cache.get(slashed)
        other_regime = other_cached[1].get("regime") if other_cached else None
        if other_regime == candidate_regime:
            count += 1
    return count


async def process_signal_for_symbol(symbol: str, current_price: float, risk_manager: RiskManager, strategy: TradingStrategy, ex: AlpacaExchange, positions: list | None = None, regime_flag: dict | None = None, banned_symbols: set | None = None) -> None:
    """Processes signal for a single symbol asynchronously."""
    # Get or create lock for this symbol
    lock = _state._symbol_locks.setdefault(symbol, asyncio.Lock())
    async with lock:
        try:
            # --- HARD KILLSWITCH CHECK ---
            # Block ALL new entries when the killswitch is active. Exits (close
            # orders) are still allowed -- the main loop already issued flatten
            # orders, but a trailing-stop / SL / TP exit that arrives after the
            # main-loop flush must also be honoured rather than silently blocked.
            if risk_manager.is_killswitch_active():
                if positions is None:
                    try:
                        positions = await ex.get_positions()
                    except Exception:
                        positions = []
                position_dict = {p["symbol"].replace("/", ""): p for p in positions}
                current_position = position_dict.get(symbol.replace("/", ""))
                if current_position is None:
                    logger.warning(
                        f"[{symbol}] KILLSWITCH active — refusing new entry. "
                        f"Reason: {risk_manager.killswitch_reason}"
                    )
                    return
                # Fall through: an existing position may still need to exit.
            if positions is None:
                # Fallback for direct calls with no pre-fetched positions.
                positions = await ex.get_positions()
            else:
                # Authoritative refresh, same reasoning as the later one
                # before buy-sizing (bot.py, "AUTHORITATIVE POSITION REFRESH"
                # below): the `positions` snapshot passed from the main loop
                # is a cycle-start snapshot shared by every concurrently-
                # evaluated symbol. The trailing-stop check right below reads
                # avg_entry_price/qty from it -- if this symbol's own
                # position changed since cycle start (a fill or manual
                # close), the stop decision uses stale entry price/qty,
                # possibly closing the wrong quantity or missing a stop that
                # should have fired. The buy path already re-fetches for its
                # own sizing/exposure math further down; the trailing-stop
                # check ran on the older snapshot every time before this.
                # Found via an external concurrency audit, confirmed by
                # tracing that no refresh existed between here and the
                # trailing-stop check below, 2026-09-22.
                try:
                    positions = await ex.get_positions()
                except Exception as pos_refresh_err:
                    logger.debug(
                        f"[{symbol}] Pre-trailing-stop position refresh failed: "
                        f"{_describe_exception(pos_refresh_err)} — using cycle-start snapshot"
                    )
            position_dict = {p["symbol"].replace("/", ""): p for p in positions}
            current_position = position_dict.get(symbol.replace("/", ""))

            # Alpaca position dicts carry no entry timestamp, which silently
            # disabled strategies._check_price_based_exits' MAX_HOLD_HOURS
            # check ("created_at" was never in the dict). Attach the entry
            # time persisted on the symbol's open decision snapshot instead.
            # Fail-safe: any error just leaves the max-hold check inert, as
            # before -- it never blocks evaluation of the position.
            if current_position is not None and "created_at" not in current_position:
                try:
                    from src.db import get_open_snapshot
                    snap = await asyncio.to_thread(get_open_snapshot, symbol)
                    if snap and snap.get("created_at"):
                        current_position["created_at"] = snap["created_at"]
                except Exception as snap_err:
                    logger.debug(f"[{symbol}] Could not attach entry time for max-hold check: {snap_err}")

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
                # Regime isn't freshly computed yet at this point in the cycle
                # (that happens below, in generate_trading_signal) -- read the
                # last cached regime for this symbol instead of restructuring
                # the evaluation order. None (no cache entry yet, e.g. first
                # evaluation of a symbol) is treated as neutral/unscaled.
                cached_regime = _state.strategy._regime_cache.get(symbol) if _state.strategy is not None else None
                regime_for_trailing = cached_regime[1].get("regime") if cached_regime else None
                trailing_action = risk_manager.check_trailing_stop(symbol, current_price, avg_entry_price, qty, regime=regime_for_trailing)
                
                if trailing_action == "close":
                    side = "sell" if qty > 0 else "buy"
                    qty_abs = abs(qty)
                    # Cancel the resting protective stop before closing, or it
                    # would linger against a position that no longer exists.
                    await _cancel_protective_stop(ex, symbol)
                    client_order_id = f"{symbol}_{side}_{qty_abs}_{int(time.time())}"
                    order_result = await ex.create_order(
                        symbol=symbol,
                        qty=qty_abs,
                        side=side,
                        type="market",
                        client_order_id=client_order_id,
                        bypass_circuit_breaker=True,
                    )
                    # Wire decision_id + filled_at so a restart/reconcile can
                    # match this exit to its snapshot by correlation, not by
                    # recency (task: exit records must carry decision_id).
                    await _persist_exit_order(
                        order_result, symbol, side, client_order_id,
                        await _snapshot_decision_id(symbol),
                    )
                    logger.info(f"Trailing Stop Executed: {symbol}")
                    await send_telegram_alert(f"🔔 <b>Trailing Stop Triggered</b>\nSymbol: {symbol}\nClosed {qty} @ ${current_price:.2f}")
                    _state.cooldowns[symbol] = time.time() + settings.COOLDOWN_SECONDS_BUY
                    # Reset position pyramid/scale-in tracking on close
                    _state.position_adds.pop(symbol, None)
                    _crash_state_writer.mark_dirty()
                    # Audit F-A/F-B: record against the exchange's actual
                    # avg entry / total qty (scale-ins make the snapshot's
                    # first-entry values wrong), use the real fill price (not
                    # the signal-time price), and subtract the commission.
                    trail_fill = order_result.get("filled_avg_price", 0.0)
                    trail_commission = order_result.get("commission", 0.0)
                    # Record fill costs for the dynamic transaction cost model --
                    # trailing-stop exits previously didn't feed this at all (no
                    # slippage/fee recording of any kind), despite being the
                    # exit path most likely to slip (fires in fast-moving
                    # markets). Symmetric to the buy-path recording. Found via
                    # an external financial-correctness audit, confirmed by
                    # reading the code, 2026-09-22.
                    if trail_fill > 0 and qty_abs > 0:
                        slippage_bps = abs(trail_fill - current_price) / current_price * 10000
                        fee_bps = (trail_commission / (trail_fill * qty_abs)) * 10000 if trail_fill * qty_abs > 0 else 0
                        risk_manager.record_fill_costs(symbol, fee_bps, slippage_bps)
                    await _record_committee_outcome(
                        symbol,
                        trail_fill if trail_fill > 0 else current_price,
                        exit_reason="trailing_stop",
                        entry_price=avg_entry_price,
                        qty=qty_abs,
                        commission=trail_commission,
                    )
                    return  # Skip standard signals
    
            # Generate trading signal
            signal = await strategy.generate_trading_signal(
                symbol,
                current_price,
                current_position
            )
    
            logger.debug(
                "[SCAN] Signal for %s @ $%.2f: %s (regime: %s, RSI: %.2f)",
                symbol, current_price, signal["action"], signal["regime"], signal.get("rsi", 0.0),
                         symbol=symbol, current_price=current_price, action=signal["action"],
                         regime=signal["regime"], rsi=signal.get("rsi", 0.0))
    
            # Bypass committee for hard exits (SL/TP)
            if signal["action"] == "close":
                pass # proceed directly to close logic below
            else:
                # ─── 5-BRAIN ENSEMBLE COMMITTEE EVALUATION ───
                from src.committee.committee import run_committee
                from src.committee.models import (
                    calculate_directional_entropy,
                    disagreement_from_entropy,
                )
                committee_result = await run_committee(symbol, current_price, signal)

                # Brain disagreement comes from the committee vote entropy.
                # strategies.py defaults it to "LOW" as a placeholder -- this
                # override is what makes the HIGH-disagreement adversarial
                # veto below actually reachable.
                #
                # It MUST use the DIRECTIONAL entropy (buy/sell votes only),
                # NOT committee_result.entropy. That value counts HOLD votes as
                # disagreement, while the committee score the veto compares it
                # against excludes HOLD from its numerator but still counts the
                # weight in its denominator -- mixing the two made the veto
                # self-locking (1 directional vote + 3 HOLDs = entropy 0.811
                # "HIGH" with a diluted ~0.23 score -> rejected every symbol
                # every cycle). See calculate_directional_entropy()'s docstring.
                # committee_result.entropy itself is unchanged: the sizing
                # multiplier, Prometheus metrics and the OOD/decision-transformer
                # state vectors still consume it and expect the full range.
                directional_entropy = calculate_directional_entropy(
                    getattr(committee_result, "votes", None) or []
                )
                signal["brain_disagreement"] = disagreement_from_entropy(directional_entropy)
    
                # ─── BUILD REGIME DASHBOARD ───
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
                dashboard.append(
                    f"Disagreement... {signal['brain_disagreement']} "
                    f"(directional entropy {directional_entropy:.2f}, score {committee_result.score:.2f}, "
                    f"veto floor {ADVERSARIAL_SCORE_FLOOR:.2f})"
                )
                dashboard.append("")
    
                if committee_result.vetoed:
                    dashboard.append("FINAL.......... NO TRADE")
                    dashboard.append(f"Reason......... {committee_result.veto_reason}")
                    dashboard.append("==============================")
                    print("\n".join(dashboard), flush=True)
                    _state.latest_scan_results[symbol] = {"score": committee_result.score, "action": "VETO", "price": current_price}
                    return

                # ADVERARIAL BRAIN VETO — Brain B (execution/risk) + Brain C (validation/anti-overfit)
                # Don't optimize toward consensus; require each brain's domain to pass
                # Brain B (execution cost > edge), Brain C (regime validation)
                # and the HIGH-disagreement floor -- shared with the backtester
                # via src/trade_decision.adversarial_veto_reasons().
                #
                # Brain C veto: statistical validity / anti-overfit.
                # This used to call _state.strategy.is_regime_validated(...) -- a
                # method that only exists on AdaptiveMetaLearner. TradingStrategy
                # has no such attribute, so EVERY evaluation raised
                # AttributeError and the bare `except: pass` swallowed it: the
                # gate was dead code disguised as a fail-safe. It is now wired to
                # the real learner (the same singleton the decision gate uses),
                # and it only vetoes once that regime has enough realized
                # outcomes to be judged. "No validation data yet" is not evidence
                # of overfitting, and vetoing on it would block every trade
                # forever -- the learner legitimately starts at 0 validated
                # regimes (see the "Insufficient regime samples: 0 < 10" gate
                # reason in the logs).
                learner = None
                validation_min_trades = None
                try:
                    from src.committee.adaptive_meta import VALIDATION_MIN_TRADES
                    from src.committee.committee import get_meta_learner

                    learner = get_meta_learner()
                    validation_min_trades = VALIDATION_MIN_TRADES
                except Exception as brain_c_err:
                    logger.debug(f"Brain C validation gate unavailable (non-fatal): {brain_c_err}")
                try:
                    veto_reasons = adversarial_veto_reasons(
                        signal, committee_result, learner, validation_min_trades
                    )
                except Exception as brain_c_err:
                    # A learner that raises mid-check must not take down the
                    # cheap Brain B / disagreement checks with it.
                    logger.debug(f"Brain C validation gate unavailable (non-fatal): {brain_c_err}")
                    veto_reasons = adversarial_veto_reasons(signal, committee_result)
                adversarial_veto = bool(veto_reasons)

                if adversarial_veto:
                    dashboard.append("FINAL.......... NO TRADE")
                    dashboard.append(f"Reason......... Adversarial veto: {'; '.join(veto_reasons)}")
                    dashboard.append("==============================")
                    print("\n".join(dashboard), flush=True)
                    _state.latest_scan_results[symbol] = {"score": committee_result.score, "action": "VETO_ADVERSARIAL", "price": current_price}
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

            # --- COMMITTEE-OVERRIDE MIN-HOLD GATE ---
            # A held long whose strategy would have emitted a discretionary
            # close (momentum/trend/mean-reversion/breakout/grid/scalp) can
            # still reach here as a committee "sell" if the committee overrode
            # that close. strategies.py's own gate never sees this path, so
            # without this check the committee override bypassed MIN_HOLD
            # entirely -- reproducing the exact 2026-10-02 churn (BTC/USD
            # closed on "[MOMENTUM] Momentum: Loss of bullish momentum" after
            # 177-179s, ~0.5% fees for a ~0.1% gross move). This gate holds a
            # discretionary exit to MIN_HOLD_MINUTES; every PRICE-BASED risk
            # exit (stop loss, trailing stop, profit target, max hold) was
            # already handled above and returned, so it is never blocked here.
            if (
                signal["action"] == "sell"
                and current_position is not None
                and not settings.COMMITTEE_MIN_HOLD_EXEMPT
                and settings.MIN_HOLD_MINUTES > 0
                and strategy is not None
                and not getattr(strategy, "backtest", False)
                and TradingStrategy._is_discretionary_close(str(signal.get("reason", "")))
            ):
                # get_position_held_minutes returns None on first sighting (it
                # records the clock) -- fail open then, matching the strategy's
                # own gate. A non-numeric result (broken/mocked strategy) also
                # fails open so a legitimate exit is never blocked.
                held_minutes = strategy.get_position_held_minutes(symbol, current_position)
                if isinstance(held_minutes, (int, float)) and held_minutes < settings.MIN_HOLD_MINUTES:
                    logger.info(
                        f"[{symbol}] Min-hold gate: committee discretionary sell "
                        f"'{signal.get('reason', '')}' held back "
                        f"({held_minutes:.1f} min < {settings.MIN_HOLD_MINUTES:g} min)"
                    )
                    _state.latest_scan_results[symbol] = {
                        "score": committee_result.score,
                        "action": "HOLD_MIN_HOLD",
                        "price": current_price,
                    }
                    return


            if signal["action"] in ["buy", "sell"]:
                # --- ROLLING SOFT LOSS LIMIT (new-entry block only) ---
                # If realized P&L over the last LOSS_LIMIT_WINDOW_HOURS is at
                # or below -|ROLLING_LOSS_LIMIT_PCT|% of equity, refuse NEW
                # entries for this cycle. Open positions still fall through and
                # are managed normally (SL/TP/trailing/min-hold) — this never
                # liquidates and never touches the hard killswitch in risk.py.
                # Fail-open: if the P&L query errors it returns 0.0 and trading
                # proceeds.
                #
                # It MUST sit inside this entry branch (after the trailing-stop
                # check above and after generate_trading_signal's price-based
                # exits), never at the top of the function. As a top-of-function
                # early return it also skipped the exit checks: a losing day
                # (exactly what trips this limit) left already-open losers
                # unmanaged, so they were never stopped out and bled further as
                # the market kept falling. Reproduced 2026-10-05 (crypto fell
                # 08:00-12:00, the bot kept holding into it) and again in
                # tests/test_integration_trading_loop.py.
                #
                # A sell while long is an EXIT, not an entry (the strategy
                # emits "close" for its own exits, but a committee-overridden
                # "sell" reaches this branch and closes the long). Never gate
                # it: the backtester closes the held long on sell/close and
                # only applies entries_blocked() to new buys, so blocking it
                # here would trap the position and diverge from the backtest.
                _is_exit_order = signal["action"] == "sell" and current_position is not None
                if settings.ROLLING_LOSS_LIMIT_PCT > 0 and not _is_exit_order:
                    try:
                        # Equity source: peak_equity is maintained by the risk
                        # manager's 5s account poll (risk.py:460) and is the
                        # freshest equity value available without an extra
                        # exchange round-trip in this hot path. Fail-safe
                        # fallback to settings.ACCOUNT_BASE mirrors risk.py:686.
                        equity = float(risk_manager.peak_equity) if risk_manager.peak_equity > 0 \
                            else float(getattr(settings, "ACCOUNT_BASE", 0.0))
                        rolling_pnl = await asyncio.to_thread(
                            get_recent_realized_pnl, settings.LOSS_LIMIT_WINDOW_HOURS
                        )
                        limit_abs = -abs(settings.ROLLING_LOSS_LIMIT_PCT) / 100.0 * equity
                        if equity > 0 and rolling_pnl <= limit_abs:
                            logger.warning(
                                f"[ROLLING_LOSS_LIMIT] {symbol}: blocking new entry — "
                                f"realized P&L last {settings.LOSS_LIMIT_WINDOW_HOURS:g}h "
                                f"is ${rolling_pnl:.2f} (limit ${limit_abs:.2f}, "
                                f"{settings.ROLLING_LOSS_LIMIT_PCT:g}% of equity ${equity:.2f})"
                            )
                            return
                    except Exception as rl_err:
                        logger.warning(
                            f"[ROLLING_LOSS_LIMIT] check failed (fail-open, entry allowed): {rl_err}"
                        )
                # ─── REGIME SWITCH CHECK (Only for ENTRY, not EXIT) ───
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
    
                # ─── BANNED SYMBOLS CHECK ───
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

                # ─── RECONCILIATION-INCOMPLETE CHECK (CR-12) ───
                # Startup/retry reconciliation couldn't fetch positions, so we
                # genuinely don't know what's open on the exchange right now.
                # Blanket-block new entries until a retry succeeds rather than
                # trading on an unknown state.
                if _state.reconciliation_incomplete:
                    dashboard.append("Risk........... VETO")
                    dashboard.append("FINAL.......... NO TRADE")
                    dashboard.append("Reason......... Reconciliation Incomplete")
                    dashboard.append("==============================")
                    print("\n".join(dashboard), flush=True)
                    return

                # ─── UNRESOLVED ORDER CHECK (CR-2) ───
                # A stale/still-open order for this symbol exists on the
                # exchange (from before a crash, or still resolving normally).
                # Entering now risks a second buy for the same signal / 2x
                # intended exposure once the first order also fills. Found via
                # an external crash-recovery audit, confirmed by reproduction,
                # 2026-09-22.
                if symbol.replace("/", "") in _state.symbols_with_unresolved_orders:
                    dashboard.append("Risk........... VETO")
                    dashboard.append("FINAL.......... NO TRADE")
                    dashboard.append("Reason......... Unresolved Order Pending")
                    dashboard.append("==============================")
                    print("\n".join(dashboard), flush=True)
                    return
    
                # AUTHORITATIVE POSITION REFRESH before sizing/sizing checks.
                # The `positions` snapshot passed from the main loop was taken
                # at cycle start; if another symbol's background task filled
                # an order earlier this same cycle, the exchange's positions
                # have advanced but our snapshot is stale. That staleness is the
                # root cause of "order not found" -> duplicate position ->
                # "insufficient balance" cascades: the bot sizes against a
                # phantom-free position set, then gets rejected mid-fill.
                # Re-fetch here so exposure/reservation math is accurate.
                try:
                    positions = await ex.get_positions()
                except Exception as pos_refresh_err:
                    logger.warning(
                        f"[{symbol}] Authoritative position refresh failed: "
                        f"{_describe_exception(pos_refresh_err)} — using cycle-start snapshot"
                    )

                # Check position limit before entering
                # Reuse the `positions` list already fetched at the top of this
                # function (bot.py:213) instead of letting update_account_status()
                # fetch it again -- measured: this redundancy previously cost 2-3x
                # get_positions()/get_account() calls per symbol per cycle.
                risk_status = await risk_manager.update_account_status(positions=positions)
                # HARD KILLSWITCH: block new entries if daily loss or drawdown
                # limit was breached (detected by update_account_status just now
                # or by the background monitor). Exits for existing positions
                # still flow through (trailing stop / SL / TP below).
                if risk_status["status"] == "killswitch_activated":
                    logger.critical(
                        f"[{symbol}] KILLSWITCH active ({risk_status.get('reason', 'unknown')}) — "
                        "blocking new entry"
                    )
                    return
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
                if risk_status["status"] == "error":
                    # Exchange/risk status fetch failed (e.g. circuit breaker
                    # OPEN, network error, auth failure). Refuse new entries --
                    # sizing against guessed equity or placing orders through
                    # a down exchange just generates noise and leaked
                    # reservations. Existing positions can still exit via
                    # trailing stops / SL / TP paths that bypass the circuit.
                    error_msg = risk_status.get("error", "risk_update_failed")
                    dashboard.append("Risk........... VETO")
                    dashboard.append("FINAL.......... NO TRADE")
                    dashboard.append(f"Reason......... Risk update failed: {error_msg}")
                    dashboard.append("==============================")
                    print("\n".join(dashboard), flush=True)
                    _state.latest_scan_results[symbol] = {
                        "score": committee_result.score,
                        "action": "VETO_ERROR",
                        "price": current_price,
                        "reason": error_msg,
                    }
                    return

    
                # Calculate position size
                # Estimate expected return using committee score, regime, and consensus
                # Score represents P(win), regime provides base edge, entropy
                # measures consensus. Shared with the backtester; returns a
                # fraction (0.03 = 3%), capped to [-2%, +5%].
                expected_return_pct = estimate_expected_return(
                    signal.get("regime", "neutral"), committee_result.score, committee_result.entropy
                )
                position_size, sizing_status = risk_manager.calculate_position_size(
                    symbol,
                    current_price,
                    signal["regime"],
                    atr=signal.get("atr"),
                    confidence=signal.get("confidence", 1.0),
                    expected_return_pct=expected_return_pct,
                    current_equity=risk_status.get("equity"),
                    drawdown_pct=risk_status.get("drawdown_pct"),
                    side=signal["action"],  # "buy" or "sell" for market impact
                    # Reuse the on-chain data analyze_market_regime already
                    # fetched (async, thread-offloaded) into signal["features"]
                    # instead of letting calculate_position_size make its own
                    # blocking re-fetch(es) -- was happening TWICE internally
                    # per call. Confirmed via a measured performance audit
                    # 2026-09-21: ~600-900ms of redundant blocking HTTP per
                    # trade signal.
                    deriv_data=signal.get("features"),
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

                # Uncertainty framework: transition-risk conviction + position-scale cap.
                # "transition probability ↑ → conviction ↓ → position size ↓"
                # (apply_uncertainty_scaling is fail-safe: degrades to a no-op on bad input)
                # Then the oracle regime multiplier (buys), the committee
                # confidence multiplier, and the MAX_SINGLE_TRADE_USD re-cap --
                # all in src/trade_decision.apply_entry_size_multipliers(),
                # shared with the backtester.
                _pre_uncertainty_qty = position_size
                position_size, uncertainty_mult = apply_entry_size_multipliers(
                    position_size,
                    signal,
                    committee_result,
                    current_price,
                    oracle_multiplier=regime_flag.get("oracle_multiplier", 1.0) if signal["action"] == "buy" else 1.0,
                )
                if uncertainty_mult < 0.999:
                    logger.info(
                        f"📉 Uncertainty scaling applied for {symbol}: {uncertainty_mult:.2f}x "
                        f"(transition_risk={signal.get('transition_risk_pct', 0.0):.0f}%, "
                        f"position_scale={signal.get('position_scale', 1.0):.2f})"
                    )
                committee_mult = getattr(committee_result, "size_multiplier", 1.0)
                logger.info(
                    f"📊 Sizing multipliers applied (committee {committee_mult:.2f}x, score "
                    f"{committee_result.score:.2f}) → Final Qty: {position_size} (was {_pre_uncertainty_qty})"
                )

                # The MAX_SINGLE_TRADE_USD re-cap after the multipliers
                # above (uncertainty scaling, oracle regime multiplier, committee
                # confidence multiplier). calculate_position_size() already
                # enforces this cap on ITS OWN output (risk.py:844), but nothing
                # re-checked it after these three multipliers could push size back
                # over -- committee_mult alone can reach 1.75x
                # (calculate_confidence_size_multiplier's documented max).
                # Verified live 2026-09-22: a position sized to exactly the
                # $2,500 cap by calculate_position_size became $4,375 (75% over
                # cap, $175 vs the intended $100 dollar-risk budget) after a
                # 1.75x committee multiplier, with no re-check anywhere before
                # create_order. Found via an external financial-correctness
                # audit, independently reproduced with the exact same numbers.
                # (Now applied inside apply_entry_size_multipliers above.)

                # Fix #1: Cap sell qty to available position (prevents 403 insufficient balance loop)
                if signal["action"] == "sell":
                    sell_pos = None
                    for p in positions:
                        if p["symbol"].replace("/", "") == symbol.replace("/", ""):
                            sell_pos = p
                            break
                    if sell_pos is None or float(sell_pos.get("qty", 0)) <= 0:
                        logger.warning(f"[{symbol}] Sell vetoed: no position to sell")
                        # Re-print the dashboard with the veto so the console
                        # doesn't leave "EXECUTE SELL" as the last visible
                        # verdict for this cycle (observed 2026-10-05: the
                        # SOL veto was correct but looked like a phantom
                        # announcement because nothing followed it).
                        dashboard.append("FINAL.......... VETO")
                        dashboard.append("Reason......... No position to sell (order would be rejected)")
                        dashboard.append("==============================")
                        print("\n".join(dashboard), flush=True)
                        return
                    available = float(sell_pos["qty"])
                    # leave tiny dust, use 0.99 cap to avoid rounding rejection
                    max_sell = round(available * 0.999, 6)
                    if position_size > max_sell:
                        logger.warning(f"[{symbol}] Sell qty capped {position_size} -> {max_sell} (available {available})")
                        position_size = max_sell
                    if position_size <= 0:
                        return
    
                # ENTRY_RACE_GUARD: positions are fetched once per cycle and
                # shared by every symbol evaluated in that cycle, so a fill
                # from the previous cycle is invisible here -- the scale-in
                # gates above then see "no position" and the entry goes out
                # again at full size (observed live 2026-10-05: two equal-size
                # ETH buys 59s apart). Veto buys for a short window after any
                # fill on this symbol. Sells are unaffected: the sell-qty cap
                # below requires an existing position, and the post-close
                # buy-cooldown already covers re-entry after a close.
                if signal["action"] == "buy":
                    _last_fill = _state.last_fill_times.get(symbol, 0.0)
                    _since_fill = time.time() - _last_fill
                    if _last_fill > 0 and _since_fill < settings.ENTRY_RACE_GUARD_SECONDS:
                        logger.warning(
                            f"[{symbol}] Buy vetoed: a fill on this symbol was placed "
                            f"{_since_fill:.0f}s ago (< {settings.ENTRY_RACE_GUARD_SECONDS}s "
                            f"race guard) and may not yet be reflected in the cycle-start "
                            f"positions snapshot -- re-entering now risks a duplicate "
                            f"full-size entry."
                        )
                        dashboard.append("FINAL.......... VETO")
                        dashboard.append(
                            f"Reason......... Entry race guard "
                            f"({_since_fill:.0f}s since last fill)"
                        )
                        dashboard.append("==============================")
                        print("\n".join(dashboard), flush=True)
                        return

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
                        await risk_manager.release_reserved_exposure(approved_notional)
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
                            await risk_manager.release_reserved_exposure(approved_notional)
                            return
                            
                        # Gate 2: Minimum time since last add
                        if now - add_info["last_add_time"] < POSITION_ADD_MIN_SECONDS:
                            logger.info(f"[{symbol}] Scale-in vetoed: minimum time between adds not met ({now - add_info['last_add_time']:.0f}s < {POSITION_ADD_MIN_SECONDS}s)")
                            await risk_manager.release_reserved_exposure(approved_notional)
                            return
                            
                        # Gate 3: Minimum score improvement
                        committee_score = committee_result.score
                        if committee_score - add_info["last_add_score"] < POSITION_ADD_MIN_SCORE_INCREASE:
                            logger.info(f"[{symbol}] Scale-in vetoed: insufficient score improvement ({committee_score:.3f} - {add_info['last_add_score']:.3f} < {POSITION_ADD_MIN_SCORE_INCREASE})")
                            await risk_manager.release_reserved_exposure(approved_notional)
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
                        _crash_state_writer.mark_dirty()
                    else:
                        # No existing position - reset tracking
                        _state.position_adds[symbol] = {"count": 0, "last_add_time": 0.0, "last_add_score": 0.0}
                        _crash_state_writer.mark_dirty()

                # Reserve a new-position slot to prevent multiple concurrently-evaluated
                # symbols in the same cycle from all passing a stale, cycle-start
                # position-count check and collectively exceeding MAX_OPEN_POSITIONS.
                # Only applies to a genuinely new entry -- a scale-in add to an
                # existing position doesn't increase the open-position count.
                is_new_entry = current_position is None
                if is_new_entry:
                    same_regime_open_count = _count_same_regime_open_positions(symbol, positions or [], strategy)
                    slot_ok, slot_reason = await risk_manager.reserve_position_slot(
                        symbol, risk_status.get("open_positions", 0), same_regime_open_count=same_regime_open_count,
                        current_positions=positions or [],
                    )
                    if not slot_ok:
                        logger.warning(f"[{symbol}] Order vetoed: {slot_reason}")
                        await risk_manager.release_reserved_exposure(approved_notional)
                        return

                # Final exchange-minimum-order-size check for a fresh/added BUY.
                # risk_manager.calculate_position_size() already applies this floor
                # to its OWN output, but several multipliers are applied to
                # position_size AFTER that call returns (uncertainty scaling,
                # oracle regime multiplier, committee confidence multiplier,
                # exposure-headroom scaling, scale-in decay) -- any of which can
                # push an already-floor-clearing size back below the exchange
                # minimum. Confirmed live 2026-09-21: calculate_position_size
                # returned a $9.70 ETH size, the committee multiplier (0.62x)
                # shrank it to $6.02, and Alpaca rejected it every time
                # (403, "cost basis must be >= minimal amount of order 10").
                # Scoped to buy only -- a "sell" here always means reducing/
                # closing an existing position (see the sell-qty-cap above),
                # where blocking a small sell could trap the account holding
                # a dust position it can never reduce.
                if signal["action"] == "buy":
                    min_order_usd = getattr(settings, "MIN_ORDER_USD", 10.0)
                    # Fee-drag floor: crypto fees are proportional, but tiny
                    # repeated entries still bleed spread+fees on every round
                    # trip, and the $10 exchange minimum forces dust-size
                    # fills on small accounts (observed live 2026-10-05: five
                    # ~$10 ETH fills netted -$0.18, almost entirely fees).
                    # Lift the effective minimum to
                    # MIN_ENTRY_NOTIONAL_EQUITY_PCT of equity when equity is
                    # known; the setting at 0 (or unknown equity) keeps the
                    # plain exchange minimum. The bump/veto logic below is
                    # unchanged -- it just targets this larger floor.
                    _equity = float(risk_status.get("equity") or 0.0)
                    _equity_floor = settings.MIN_ENTRY_NOTIONAL_EQUITY_PCT * _equity
                    if _equity_floor > min_order_usd:
                        min_order_usd = _equity_floor
                    final_notional = current_price * position_size
                    if 0 < final_notional < min_order_usd:
                        if final_notional >= min_order_usd * 0.5:
                            # Close enough that bumping up is a small, bounded
                            # deviation -- do it rather than waste the trade
                            # (mirrors calculate_position_size's own bump logic).
                            bumped_size = math.ceil(min_order_usd / current_price * 1_000_000) / 1_000_000
                            bumped_notional = bumped_size * current_price
                            # The bump increases notional beyond what
                            # check_and_reserve_exposure already approved above
                            # -- reserve the delta too, or the order can exceed
                            # its own exposure reservation. Found via an
                            # external financial-correctness audit, confirmed
                            # by reading the code, 2026-09-22: this bump was
                            # previously unconditional, with no re-reservation.
                            extra_needed = bumped_notional - approved_notional
                            if extra_needed > 0:
                                # Reuse the same current_exposure risk_status
                                # already fetched above (line ~1137) -- no
                                # await happens between that fetch and here,
                                # so it isn't stale, and reusing it avoids
                                # another redundant get_account()/get_positions()
                                # round trip. reserved_total (which DOES
                                # reflect the reservation from the first
                                # check_and_reserve_exposure call above) is
                                # re-read fresh inside the lock regardless.
                                # Found verifying an external concurrency
                                # audit's "redundant get_account() calls"
                                # claim, 2026-09-22.
                                extra_approved, extra_reason = await risk_manager.check_and_reserve_exposure(
                                    extra_needed, current_exposure=risk_status.get("current_exposure")
                                )
                                if extra_approved < extra_needed:
                                    if extra_approved > 0:
                                        await risk_manager.release_reserved_exposure(extra_approved)
                                    logger.warning(
                                        f"[{symbol}] Order vetoed: bumping to exchange minimum needs "
                                        f"${extra_needed:.2f} more exposure headroom than already reserved "
                                        f"(${approved_notional:.2f}), but only ${extra_approved:.2f} was "
                                        f"available ({extra_reason})."
                                    )
                                    await risk_manager.release_reserved_exposure(approved_notional)
                                    # This veto (and the two others in this
                                    # min-order-bump block) fire AFTER
                                    # reserve_position_slot() above, so a slot
                                    # reservation exists here too for a fresh
                                    # entry -- releasing only the exposure
                                    # reservation leaked it, needlessly
                                    # blocking sibling symbols for up to the
                                    # slot TTL. Found via an external
                                    # financial-correctness audit, confirmed
                                    # by reading the code, 2026-09-22.
                                    if is_new_entry:
                                        await risk_manager.release_position_slot(symbol)
                                    return
                                approved_notional += extra_approved
                            logger.info(
                                f"[{symbol}] Post-multiplier size bump to exchange minimum: "
                                f"${final_notional:.2f} -> ${min_order_usd:.2f} notional "
                                f"({position_size} -> {bumped_size})"
                            )
                            position_size = bumped_size
                        else:
                            logger.warning(
                                f"[{symbol}] Order vetoed: final size after all multipliers "
                                f"is ${final_notional:.2f} notional, well below the "
                                f"${min_order_usd:.2f} exchange minimum -- would need to bump "
                                f">2x to place, too large a deviation."
                            )
                            await risk_manager.release_reserved_exposure(approved_notional)
                            if is_new_entry:
                                await risk_manager.release_position_slot(symbol)
                            return

                # Place order
                # A sell while holding is an EXIT (committee-overridden), not an
                # entry -- cancel the protective stop before it closes, so the
                # resting sell never lingers after the position is flat.
                if signal["action"] == "sell":
                    await _cancel_protective_stop(ex, symbol)
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
                    await risk_manager.release_reserved_exposure(approved_notional)
                    if is_new_entry:
                        await risk_manager.release_position_slot(symbol)
                    logger.error(f"[{symbol}] Order placement failed: {_describe_exception(order_e)}")
                    raise
                # Key the entry ledger row on the snapshot it actually belongs
                # to. A genuinely fresh entry creates a snapshot with
                # committee_result.decision_id, but a SCALE-IN -- and also a
                # "new entry" that save_decision_snapshot FOLDS into a still-open
                # snapshot for the symbol -- belongs to that existing snapshot.
                # Recording the buy under anything else means the snapshot close
                # (get_entry_fee_estimate keys on the snapshot's decision_id)
                # misses the fee and under-subtracts the round trip. Resolve the
                # open snapshot first; fall back to the committee id when none
                # exists yet (the fresh-entry case, where the snapshot is created
                # below with that same id). get_open_snapshot is cached per cycle,
                # so the snapshot block further down re-reads it for free.
                ledger_decision_id = committee_result.decision_id
                try:
                    from src.db import get_open_snapshot
                    snap = await asyncio.to_thread(get_open_snapshot, symbol)
                    if snap and snap.get("decision_id"):
                        ledger_decision_id = snap["decision_id"]
                except Exception as snap_err:
                    logger.debug(
                        f"[{symbol}] snapshot lookup for entry ledger key "
                        f"failed (non-fatal): {snap_err}"
                    )
                await asyncio.to_thread(
                    _persist_order_record, order_result, symbol, signal["action"], client_order_id,
                    decision_id=ledger_decision_id,
                )
                # Also write the ENTRY-leg fill to the orders ledger (not just
                # the exit legs). This records the entry fee estimate so the
                # later snapshot close can subtract the round trip's entry
                # fee, not only the exit fee (see _record_committee_outcome /
                # db.get_entry_fee_estimate). Fail-safe: _persist_exit_order
                # only writes when the order actually filled.
                await _persist_exit_order(
                    order_result, symbol, signal["action"], client_order_id,
                    ledger_decision_id,
                )
                # Record the fill time for the ENTRY_RACE_GUARD check above:
                # the next cycle's positions snapshot may not yet include this
                # fill, so its entry path must not treat the symbol as flat.
                _state.last_fill_times[symbol] = time.time()

# Deliberately NOT releasing either reservation here on success --
                # reproduced with real asyncio race tests that releasing
                # EITHER one immediately on a successful fill reopens the
                # exact race both reservations exist to prevent, for any
                # sibling symbol whose evaluation (data fetch, committee
                # brains, possibly an LLM call) reaches its own reserve step
                # later in the SAME cycle. `current_exposure` and
                # `open_position_count` are each one stale, cycle-start
                # snapshot shared by every concurrently-evaluated symbol; a
                # reservation freed mid-cycle lets a later sibling reserve
                # against that same stale snapshot too, and the two
                # collectively exceed the cap the reservation exists to
                # enforce (measured: a 3-way race with a $10k exposure cap
                # and $8k already-stale exposure landed at $13.4k -- 34% over
                # cap -- when released eagerly on success). Both are instead
                # left to expire via their own 30s TTL prune (see
                # check_and_reserve_exposure / reserve_position_slot) --
                # costs a few phantom-reserved seconds of new-entry
                # throughput after a fill, never a cap breach. The next
                # cycle's fresh update_account_status()/open_position_count
                # will reflect the fill regardless, once the TTL clears it.

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
                await send_telegram_alert(f"📈 <b>Order Executed</b>\nSymbol: {symbol}\nAction: {signal['action'].upper()}\nQty: {position_size}\nPrice: ${current_price:.2f}")
    
                # Persist committee decision snapshot for the adaptive meta-learner
                # (fail-safe; closed out with realized PnL when the position exits).
                # For a scale-in ADD, do NOT create a second open snapshot -- that
                # would record only the add's price/qty and leave the original
                # snapshot as a ghost. Instead update the existing snapshot's
                # entry price (weighted average) and total qty so the exit math
                # in _record_committee_outcome is correct (audit finding F-A).
                # Basis/qty for the exchange-side protective stop (armed below).
                # A scale-in overrides these with the weighted-average entry
                # and combined qty so the stop covers the whole position.
                # decision_id likewise tracks the snapshot this stop protects:
                # a fresh entry creates its own, while a scale-in must link to
                # the ORIGINAL snapshot it folds into -- otherwise the resting
                # stop's later fill would correlate to the add's throwaway
                # decision_id and the real exit would not be found on restart.
                stop_basis = filled_price if filled_price > 0 else current_price
                stop_qty = float(position_size)
                stop_decision_id = committee_result.decision_id
                try:
                    if is_new_entry:
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
                    else:
                        # Scale-in: fold this add into the existing open snapshot
                        # (audit finding F-A). Weighted-average the entry price
                        # over the REAL fill prices, not the signal-time price.
                        from src.db import get_open_snapshot, update_decision_snapshot_position
                        snap = await asyncio.to_thread(get_open_snapshot, symbol)
                        if snap:
                            add_fill = filled_price if filled_price > 0 else current_price
                            prev_entry = float(snap.get("entry_price", 0.0))
                            prev_qty = float(snap.get("qty", 0.0))
                            if prev_entry > 0 and prev_qty > 0:
                                new_qty = prev_qty + position_size
                                new_avg = (prev_entry * prev_qty + add_fill * position_size) / new_qty
                                await asyncio.to_thread(
                                    update_decision_snapshot_position,
                                    snap["decision_id"],
                                    entry_price=new_avg,
                                    qty=new_qty,
                                )
                                logger.info(f"[SCALE-IN] Snapshot {snap['decision_id']} updated: entry ${prev_entry:.2f} -> ${new_avg:.2f} (weighted by real fill), qty {prev_qty} -> {new_qty}")
                                stop_basis = new_avg
                                stop_qty = new_qty
                                stop_decision_id = snap["decision_id"]
                except Exception as db_e:
                    logger.warning(f"Decision snapshot persist failed for {symbol} (non-fatal): {db_e}")

                # Arm/replace the exchange-side protective stop so the position
                # is capped even between the bot's 60s exit scans. Fail-safe and
                # a no-op unless PROTECTIVE_STOPS_ENABLED. The snapshot's
                # decision_id links the resting stop to the position it covers.
                await _arm_protective_stop(
                    ex, symbol, stop_qty, stop_basis * (1.0 - settings.STOP_LOSS_PCT),
                    decision_id=stop_decision_id,
                )

            elif signal["action"] == "close" and current_position:

                # Close existing position
                # Cancel the resting protective stop FIRST: once the position is
                # flat a leftover stop_limit sell could open a short (crypto is
                # long-only, so it would just be rejected -- but it must never
                # be left resting against a position that no longer exists).
                await _cancel_protective_stop(ex, symbol)
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
                    bypass_circuit_breaker=True,
                )
                await _persist_exit_order(
                    order_result, symbol, side, client_order_id,
                    await _snapshot_decision_id(symbol),
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

                    # Record fill costs for the dynamic transaction cost model --
                    # previously only the entry/buy path did this, so the model
                    # only ever learned from entry-leg fills and never from exits
                    # (which slip worst, since stop-loss exits fire in fast
                    # markets). Symmetric to the buy-path recording above. Found
                    # via an external financial-correctness audit, confirmed by
                    # reading the code, 2026-09-22.
                    if filled_price * qty > 0:
                        fee_bps = (commission / (filled_price * qty)) * 10000
                        risk_manager.record_fill_costs(symbol, fee_bps, slippage_bps)

                    logger.info(f"[TRADE] Position closed: {symbol} (was {qty}) - reason: {signal.get('reason', 'unknown')}")
                    logger.debug(f"[TRADE] Close order result: {order_result}")
                    await send_telegram_alert(f"✅ <b>Position Closed</b>\nSymbol: {symbol}\nReason: {signal.get('reason', 'unknown')}")
                    _state.cooldowns[symbol] = time.time() + settings.COOLDOWN_SECONDS_BUY
                    # Reset position pyramid/scale-in tracking on close
                    _state.position_adds.pop(symbol, None)
                    _crash_state_writer.mark_dirty()
                    # Clear trailing peaks/troughs for this symbol
                    if _state.strategy is not None and hasattr(_state.strategy, '_trailing_peaks'):
                        _state.strategy._trailing_peaks.pop(symbol, None)
                    if _state.strategy is not None and hasattr(_state.strategy, '_trailing_troughs'):
                        _state.strategy._trailing_troughs.pop(symbol, None)
                    # Audit F-A/F-B: record against the exchange's actual avg
                    # entry / total qty, the real fill price, minus commission.
                    await _record_committee_outcome(
                        symbol,
                        filled_price if filled_price > 0 else current_price,
                        exit_reason=signal.get('reason', 'unknown'),
                        entry_price=float(current_position.get("avg_entry_price", 0.0)),
                        qty=qty_abs,
                        commission=commission,
                    )
    
        except Exception as e:
            logger.error(f"Error processing {symbol}: {_describe_exception(e)}")
    
    
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
            # Call update_account_status() once and reuse the result for both
            # the killswitch and exposure-cap checks. Previously this called
            # check_killswitch_conditions() (which itself calls
            # update_account_status()) AND then update_account_status() again,
            # producing 2x the failed network calls and 2x the error log spam
            # every cycle whenever the circuit breaker was OPEN.
            status = await risk_manager.update_account_status()
            if status.get("status") == "killswitch_activated":
                logger.critical("KILLSWITCH ACTIVATED - Liquidating all positions")
                await send_telegram_alert("🛑 <b>KILLSWITCH ACTIVATED</b>\nLiquidating all positions immediately.")
                await risk_manager.liquidate_all_positions()
                _state._shutdown_requested = True
                return  # Exit the task, let main loop handle shutdown

            if status.get("status") == "exposure_limit_exceeded":
                logger.warning("Exposure cap breached - reducing")
                await risk_manager.reduce_exposure_to_cap()
            await asyncio.sleep(settings.KILLSWITCH_CHECK_INTERVAL_SEC)
        except Exception as e:
            logger.error(f"Killswitch monitor error: {e}")
            await asyncio.sleep(settings.KILLSWITCH_CHECK_INTERVAL_SEC)

def _promotion_marks() -> dict[str, str | None]:
    return {bundle: model_store.promoted_at(bundle) for bundle in model_store.BUNDLES}


def _reload_promoted_models(before: dict[str, str | None]) -> None:
    """Trainers run as subprocesses and promote into src.model_store; the
    brains cache their model in this process. Drop the cache of any bundle
    that was promoted so the next decision uses the new model instead of
    waiting for a restart. Untouched bundles keep their cache (and the live
    transformer keeps its online-learning updates)."""
    for bundle, mark in _promotion_marks().items():
        if mark == before.get(bundle):
            continue
        try:
            if bundle == "transformer":
                from src.committee.bayesian_transformer import reset_bayesian_transformer
                from src.committee.transformer_brain import reset_fast_ensemble_predictor, reset_ml_predictor
                reset_ml_predictor()
                reset_fast_ensemble_predictor()
                reset_bayesian_transformer()
            elif bundle == "decision_transformer":
                from src.committee.decision_transformer import reset_decision_transformer
                reset_decision_transformer()
            elif bundle == "ppo":
                from src.committee.rl_meta import reset_ppo_model
                reset_ppo_model()
            logger.info(f"New '{bundle}' model promoted; it will be loaded on the next decision.")
        except Exception as e:
            logger.error(f"Failed to reload promoted '{bundle}' model (restart will pick it up): {e}")


async def run_periodic_analyzer() -> None:
    """Background task to run the analyzer script periodically."""
    import os
    script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'weekly_analyzer.py')
    
    while True:
        try:
            logger.info("Running periodic analyzer script...")
            log_training_job_start("analyzer", script_path)
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            log_training_job_result("analyzer", returncode=process.returncode, script=script_path)
            
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
            log_training_job_start("automl", script_path)
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
            log_training_job_result("automl", returncode=process.returncode, script=script_path)
                
        except Exception as e:
            logger.error(f"Error running AutoML pipeline: {e}")
            await asyncio.sleep(3600)

async def run_periodic_cull() -> None:
    """Background task to run the Evolution Cull on the 1st of every month."""
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
            log_training_job_start("cull", script_path)
            marks = _promotion_marks()
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            _reload_promoted_models(marks)
            
            if process.returncode == 0:
                logger.info(f"Evolution Cull completed successfully:\n{stdout.decode().strip()}")
            else:
                logger.error(f"Evolution Cull failed with code {process.returncode}:\n{stderr.decode().strip()}")
            log_training_job_result("cull", returncode=process.returncode, script=script_path)
                
        except Exception as e:
            logger.error(f"Error running Evolution Cull: {e}")
            await asyncio.sleep(86400)

async def run_periodic_research() -> None:
    """Background task to run the Automatic Researcher every Sunday morning."""
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
            log_training_job_start("research", script_path)
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
            log_training_job_result("research", returncode=process.returncode, script=script_path)
                
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
            log_training_job_start("transformer_replay", script_path)
            marks = _promotion_marks()
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            _reload_promoted_models(marks)

            if process.returncode == 0:
                logger.info(f"Transformer replay fine-tune completed successfully:\n{stdout.decode().strip()}")
            else:
                logger.error(f"Transformer replay fine-tune failed with code {process.returncode}:\n{stderr.decode().strip()}")
            log_training_job_result("transformer_replay", returncode=process.returncode, script=script_path)

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
            log_training_job_start("ppo_retrain", script_path)
            marks = _promotion_marks()
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            _reload_promoted_models(marks)

            if process.returncode == 0:
                logger.info(f"PPO Meta-Learner retraining completed successfully:\n{stdout.decode().strip()}")
            else:
                logger.error(f"PPO Meta-Learner retraining failed with code {process.returncode}:\n{stderr.decode().strip()}")
            log_training_job_result("ppo_retrain", returncode=process.returncode, script=script_path)

        except Exception as e:
            logger.error(f"Error running PPO Meta-Learner retraining: {e}")
            await asyncio.sleep(3600)

async def run_periodic_decision_transformer_retrain() -> None:
    """Background task to retrain the Decision Transformer weekly.
    
    Trains the offline RL sequence model on accumulated closed decision
    snapshots (backtest + live trades). Scheduled on Sunday 8 AM
    (2 hours after PPO retraining) to avoid resource contention.
    """
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
            log_training_job_start("decision_transformer", script_path)
            marks = _promotion_marks()
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            _reload_promoted_models(marks)

            if process.returncode == 0:
                logger.info(f"Decision Transformer retraining completed successfully:\n{stdout.decode().strip()}")
            else:
                logger.error(f"Decision Transformer retraining failed with code {process.returncode}:\n{stderr.decode().strip()}")
            log_training_job_result("decision_transformer", returncode=process.returncode, script=script_path)

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
            log_training_job_start("pbt")
            
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
            log_training_job_result("pbt")
            
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
            log_training_job_start("ood_retrain")

            from src.committee.decision_transformer import BRAINS, REGIMES
            from src.db import get_closed_decision_snapshots
            from src.ood_discriminator import (
                OOD_BOOTSTRAP_MIN_HISTORY,
                OOD_MODEL_PATH,
                build_ood_state_vector,
                get_ood_discriminator,
            )

            ood_disc = get_ood_discriminator()

            # "Historical" class = REAL closed trades. The old code refused to
            # train until _is_trained was already True -- a deadlock, so the
            # OOD veto never armed at all.
            closed_decisions = await asyncio.to_thread(get_closed_decision_snapshots, limit=5000)
            if len(closed_decisions) < OOD_BOOTSTRAP_MIN_HISTORY:
                logger.info(
                    f"OOD retrain waiting for history: "
                    f"{len(closed_decisions)}/{OOD_BOOTSTRAP_MIN_HISTORY} closed decisions"
                )
                await asyncio.sleep(3600)
                continue

            # Build historical state vectors
            historical_states = []
            for dec in closed_decisions:
                regime = dec.get("regime", "default")
                features = dec.get("features", {}) or {}
                brain_votes = dec.get("brain_votes", {}) or {}
                historical_states.append(build_ood_state_vector(regime, features, brain_votes, REGIMES, BRAINS))

            historical_states = np.array(historical_states)

            # Both classes come from the same real closed-trade pool: older
            # trades are the historical reference, the most recent are the
            # "live" sample, so a shift in the newest state distribution
            # (regime/feature drift) is what gets flagged. A disjoint tail
            # (not the whole array) keeps the two classes separate.
            live_states = historical_states[-min(200, len(historical_states)):]
            historical_states = historical_states[: max(1, len(historical_states) - len(live_states))]

            if len(live_states) < 50:
                logger.warning("Not enough live states for OOD retraining")
                await asyncio.sleep(3600)
                continue

            # Retrain
            acc = ood_disc.train_on_data(historical_states, live_states)
            logger.info(f"OOD Discriminator retrained: val_acc={acc:.3f}")

            # OOD_MODEL_PATH lives under model_store.store_dir() (data/models/ood/),
            # the persistent data volume -- never the image's models/ dir, which
            # is wiped on every redeploy.
            ood_disc.save()
            logger.info(f"OOD Discriminator saved to {OOD_MODEL_PATH}")
            log_training_job_result("ood_retrain", returncode=0)

        except Exception as e:
            logger.error(f"Error in OOD Discriminator retraining: {e}")
            log_training_job_result("ood_retrain", returncode=1)

        # Run every hour
        await asyncio.sleep(3600)

async def run_periodic_post_mortem() -> None:
    """Background task to run the Post-Mortem AI every Saturday morning."""
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
            log_training_job_start("post_mortem", script_path)
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
            log_training_job_result("post_mortem", returncode=process.returncode, script=script_path)
                
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
    cutoff_date = datetime.now(UTC) - timedelta(days=retention_days)
    cutoff_iso = cutoff_date.isoformat()

    while True:
        cutoff_iso = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        try:
            logger.info(f"Starting database maintenance - removing records older than {retention_days} days ({cutoff_iso})")

            from sqlalchemy import delete, text

            from src.db import DecisionSnapshot, ShadowTrade, get_engine

            engine = get_engine()
            with engine.begin() as conn:
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

            # VACUUM must run OUTSIDE any transaction (SQLite forbids it inside
            # one: "cannot VACUUM from within a transaction"). Run it on a fresh
            # connection after the deletes commit. No-op on PostgreSQL where
            # the bot uses a real background autovacuum.
            if str(engine.url).startswith("sqlite"):
                with engine.connect() as conn:
                    conn.execute(text("VACUUM"))
                    conn.commit()

            logger.info(f"Database maintenance completed: {deleted_snapshots} snapshots, {deleted_shadow_trades} shadow trades deleted")

        except Exception as e:
            logger.error(f"Error during database maintenance: {e}")

        await asyncio.sleep(7 * 24 * 3600)

def _bootstrap_adaptive_learner_from_history() -> int:
    """Rebuild the adaptive meta-learner from the DB's closed-trade history.

    Without this, per-regime sample counts grow only one per closed trade
    AFTER process start, so on an account that closes a handful of trades a
    day the 30-sample live gate takes weeks to reach and every adaptive
    decision source stays in shadow indefinitely (observed live 2026-10-05:
    'Insufficient regime samples: 0 < 30 / 1 < 30' on every cycle).

    Snapshots are replayed oldest-first through the learner's normal update()
    path via AdaptiveMetaLearner.rebuild_from_history(), which resets to cold
    start first -- so restarts can never double-count: sample counts always
    equal the number of closed snapshots actually in the DB.

    Fail-safe: returns 0 without touching learner state if the DB yields no
    closed snapshots (including a DB outage) -- an empty history must NOT
    reset an already-learned state file.
    """
    from src.committee.committee import get_meta_learner
    from src.db import get_closed_decision_snapshots

    learner = get_meta_learner()
    if learner is None:
        return 0
    snaps = get_closed_decision_snapshots(limit=10000)
    if not snaps:
        return 0
    # get_closed_decision_snapshots returns most-recent-first; the learner
    # must see trades in the order they actually happened.
    histories = []
    for snap in reversed(snaps):
        pnl = float(snap.get("realized_pnl") or 0.0)
        return_pct = float(snap.get("return_pct") or 0.0)
        if return_pct == 0.0 and pnl != 0.0:
            # Legacy rows closed before return_pct was recorded: reconstruct
            # it the same way the live exit path does (net pnl / notional,
            # bot._record_committee_outcome).
            notional = float(snap.get("entry_price") or 0.0) * float(snap.get("qty") or 0.0)
            if notional > 0:
                return_pct = pnl / notional * 100.0
        histories.append({
            "regime": snap.get("regime", "default"),
            "final_action": snap.get("final_action", "hold"),
            "brain_votes": snap.get("brain_votes", {}) or {},
            "net_pnl": pnl,
            "return_pct": return_pct,
        })
    return learner.rebuild_from_history(histories)


async def run_trading_bot() -> None:
    """Main trading bot loop."""

    # Declared before anything else in the try block so the `finally` below
    # can always reference it -- otherwise an exception raised during early
    # startup (before the tasks are created) hits an UnboundLocalError in
    # `finally` that masks the real error.
    active_tasks: set[asyncio.Task] = set()

    try:
        # Install structured stdout + the persistent rotating file log
        # (data/logs/apex_bot.log, 20 MB x 5) before anything else logs, so
        # startup and every TRAINING_JOB line reach the on-disk file.
        configure_structlog()
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

        # Replace an untrainable replay buffer in the data volume with the one
        # shipped in the image (see src/seed_data.py). Never blocks startup.
        try:
            from src.seed_data import bootstrap_replay_buffer
            await asyncio.to_thread(bootstrap_replay_buffer)
        except Exception as e:
            logger.error(f"Replay buffer seed check failed (non-fatal): {e}")

        # Initialize database (handle connection failures gracefully)
        try:
            corruption_detected = init_db()
            logger.info("Database connected successfully")
            if corruption_detected:
                # init_db() silently rebuilt a corrupt DB -- every decision
                # snapshot, adaptive-learner sample, and closed-trade record
                # from before this point is gone. This previously produced
                # only a logger.warning inside db.py with nothing downstream
                # ever alerted (distinct from the connection-totally-fails
                # case below, which already alerted). Confirmed as a real
                # gap 2026-09-21 cross-checking an external audit.
                try:
                    await get_alerting_engine().alert_data_integrity_failure(
                        "sqlite_corruption_rebuild", 1,
                        {"impact": "database was corrupt and has been rebuilt empty -- "
                                   "all decision snapshots, adaptive learner samples, and "
                                   "closed-trade history prior to this restart are lost"},
                    )
                except Exception as alert_e:
                    logger.error(f"Failed to send DB-corruption alert: {alert_e}")
        except Exception as e:
            logger.warning(f"Database connection failed (will retry later): {e}")
            logger.info("Running in offline mode - some features may be limited")
            # Audit F6: a broken/corrupt DB previously degraded to offline mode
            # with ONLY this log line -- no snapshots, no meta-learner updates,
            # no max-hold exit, indefinitely. Escalate it as a critical alert.
            try:
                await get_alerting_engine().alert_system_health(
                    "database", "down",
                    {"error": str(e), "impact": "no decision snapshots, no adaptive learning, no max-hold exit"},
                )
            except Exception as alert_e:
                logger.error(f"Failed to send database-down alert: {alert_e}")

        # Bootstrap the adaptive meta-learner from closed-trade history so
        # its per-regime sample gates count lifetime closed trades, not just
        # trades closed since this process started. Must run BEFORE the first
        # trading decision (the same learner singleton backs the committee's
        # decision-source gates). Fail-safe: a DB outage yields no snapshots
        # and leaves the loaded learner state untouched -- see
        # _bootstrap_adaptive_learner_from_history.
        try:
            _bootstrapped = await asyncio.to_thread(_bootstrap_adaptive_learner_from_history)
            if _bootstrapped:
                logger.info(
                    f"Adaptive learner bootstrapped from {_bootstrapped} closed "
                    f"decision snapshot(s)"
                )
        except Exception as boot_err:
            logger.warning(f"Adaptive learner bootstrap failed (non-fatal): {boot_err}")

        logger.info(settings.log_config())

        # Initialize exchange
        _state.ex = AlpacaExchange()
        try:
            await _state.ex.load()
            logger.info("Alpaca exchange connected")
        except RuntimeError as e:
            # Re-raise ONLY authentication/configuration errors which are fatal
            # Other RuntimeErrors (network issues, etc.) should fall through to offline mode
            if "authentication failed" in str(e).lower() or "unauthorized" in str(e).lower():
                logger.error(f"Alpaca authentication failed (fatal): {e}")
                raise
            # Other RuntimeErrors - treat as transient, start in offline mode
            logger.warning(f"Alpaca exchange connection failed on startup: {e}. Bot will start in offline/retry mode.")
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
        # cache_ttl defaults to 60s, same as LOOP_INTERVAL_SEC's default -- in
        # practice a full cycle (sleep + processing overhead) almost always
        # takes slightly longer than exactly 60s, so the regime cache was
        # nearly always just-expired by the next check, refetching
        # multi-timeframe bars every cycle regardless. Give it real headroom
        # over the actual configured loop interval instead.
        _state.strategy = TradingStrategy(_state.ex, cache_ttl=settings.LOOP_INTERVAL_SEC * 1.5)
        _state.risk_manager = RiskManager(_state.ex)
        # Share one protective-stop registry so risk.py's emergency flatten
        # paths (killswitch / exposure reduction) cancel the same resting
        # stops bot.py armed.
        _state.risk_manager.protective_stops = _state.protective_stops
        logger.info("Trading strategy and risk manager initialized")

        # Restore crash-recovery state (peak_prices, cooldowns, position_adds,
        # peak_equity) from the last successful flush so trailing stops and
        # killswitch logic don't start from scratch after a crash.
        try:
            from src.persistent_state import load_persistent_state
            counts = apply_crash_recovery_state(load_persistent_state())
            logger.info(
                f"Restored crash-recovery state: "
                f"peak_equity={_state.risk_manager.peak_equity:.2f}, "
                f"{counts['peak_prices']} peak prices, "
                f"{counts['cooldowns']} cooldowns, "
                f"{counts['position_adds']} position_adds restored"
            )
        except Exception as restore_err:
            logger.warning(f"Could not restore crash-recovery state (non-fatal): {restore_err}")

        # Startup reconciliation (audit F4): close ghost 'open' decision
        # snapshots whose positions no longer exist on the exchange, and warn
        # about exchange positions with no snapshot. Fully fail-safe.
        try:
            await reconcile_open_snapshots(_state.ex)
        except Exception as rec_e:
            logger.warning(f"Startup snapshot reconciliation failed (non-fatal): {rec_e}")

        try:
            held_positions = await _state.ex.get_positions()
            held_symbols = {p["symbol"].replace("/", "") for p in held_positions} if held_positions else set()
            _prune_stale_restored_state(held_symbols)
        except Exception as prune_err:
            logger.warning(f"Could not prune stale restored peaks (non-fatal): {prune_err}")

        # Initialize AlertingEngine -- reuse the process-wide singleton that
        # src/committee/committee.py already uses for brain-failure alerts, so
        # cooldowns, dedup keys, and escalation counters live in ONE state
        # store instead of two independent engines that could each send the
        # same alert within the other's cooldown window.
        alerting_engine = get_alerting_engine()
        alerting_engine.risk_manager = _state.risk_manager
        alerting_engine.exchange = _state.ex
        logger.info("Alerting engine initialized")

        # Make sure the model warmup (fired above) has actually finished
        # before we start evaluating live signals. By this point it has
        # been running concurrently with DB init, exchange auth, and the
        # FastAPI server start, so this await is typically a no-op.
        # Add timeout to prevent indefinite blocking on model load.
        try:
            await asyncio.wait_for(model_warmup_task, timeout=60.0)
            logger.info("Transformer model warmup complete")
        except TimeoutError:
            logger.warning("Transformer model warmup timed out after 60s (will fall back to signal-only voting)")
            model_warmup_task.cancel()
        except Exception as e:
            logger.warning(f"Transformer model warmup failed (will fall back to signal-only voting): {e}")

        # Helper: log any unhandled exception from a background task before
        # removing it from active_tasks. Without this, a crashing killswitch /
        # heartbeat / analyzer silently disappears with no log entry and never
        # restarts -- leaving safety mechanisms offline.
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

        # Start crash-state flush heartbeat (CR-5b) -- bounds worst-case
        # crash-recovery state loss to flush_interval (5s) instead of the
        # main loop's full LOOP_INTERVAL_SEC (default 60s).
        crash_flush_task = asyncio.create_task(crash_state_flush_heartbeat_loop(), name="crash_state_flush")
        active_tasks.add(crash_flush_task)
        crash_flush_task.add_done_callback(_on_task_done)
        logger.info("Crash-state flush heartbeat started")

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

        # Start periodic state cleanup (runs every 5 minutes)
        async def state_cleanup_loop() -> None:
            while not _state._shutdown_requested:
                try:
                    await asyncio.sleep(300)  # 5 minutes
                    cleaned = _state.cleanup_stale_state(max_age_seconds=3600)
                    if any(v > 0 for v in cleaned.values()):
                        logger.info(f"Periodic state cleanup: {cleaned}")
                    
                    # Also clean RiskManager state
                    if _state.risk_manager is not None:
                        rm_cleaned = _state.risk_manager.cleanup_stale_state(
                            max_age_seconds=3600,
                            active_symbols=settings.SYMBOLS
                        )
                        if any(v > 0 for v in rm_cleaned.values()):
                            logger.info(f"RiskManager cleanup: {rm_cleaned}")

                    # Also clean TradingStrategy state (_trailing_peaks/_trailing_troughs
                    # for symbols no longer in the active trading universe)
                    if _state.strategy is not None:
                        strat_cleaned = _state.strategy.cleanup_stale_state(active_symbols=set(settings.SYMBOLS))
                        if any(v > 0 for v in strat_cleaned.values()):
                            logger.info(f"TradingStrategy cleanup: {strat_cleaned}")
                except Exception as e:
                    logger.error(f"State cleanup error: {e}")
                    await asyncio.sleep(60)

        cleanup_task = asyncio.create_task(state_cleanup_loop(), name="state_cleanup")
        active_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(_on_task_done)
        logger.info("Periodic state cleanup task started")

# Start deployment registry heartbeat
        async def deployment_heartbeat_loop():
            while not _state._shutdown_requested:
                try:
                    await asyncio.sleep(60)  # Heartbeat every 60 seconds
                    cleanup_stale()
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

                await flush_crash_recovery_state()

                # --- HARD KILLSWITCH CHECK (every cycle) ---
                # Block all new entries and flatten positions when the daily-loss
                # or drawdown limit is breached. This runs every cycle (not just
                # via the background monitor_killswitch task) so there is no
                # window where the main loop can open a new position after the
                # limit has tripped.
                if _state.risk_manager is not None and _state.risk_manager.is_killswitch_active():
                    logger.critical(
                        f"KILLSWITCH ACTIVE ({_state.risk_manager.killswitch_reason}) — "
                        "blocking all new entries and flattening open positions"
                    )
                    # Flatten all open positions immediately
                    try:
                        positions = await _state.ex.get_positions(bypass_circuit_breaker=True)
                        for pos in positions:
                            sym = pos["symbol"]
                            qty = float(pos.get("qty", 0))
                            if qty == 0:
                                continue
                            side = "sell" if qty > 0 else "buy"
                            qty_abs = abs(qty)
                            client_order_id = f"killswitch_{sym}_{side}_{qty_abs}_{int(time.time())}"
                            await _state.ex.create_order(
                                symbol=sym, qty=qty_abs, side=side, type="market",
                                client_order_id=client_order_id,
                                bypass_circuit_breaker=True,
                            )
                            logger.critical(f"KILLSWITCH FLATTENED: closed {qty} {sym} ({side})")
                        await send_telegram_alert(
                            f"🛑 <b>KILLSWITCH ACTIVE</b>\n"
                            f"Reason: {_state.risk_manager.killswitch_reason}\n"
                            "All new entries blocked. Flattening open positions."
                        )
                    except Exception as flush_e:
                        logger.error(f"KILLSWITCH flatten failed: {_describe_exception(flush_e)}")
                    await asyncio.sleep(settings.LOOP_INTERVAL_SEC)
                    continue

                # Fetch positions once per cycle (was fetched redundantly per symbol)
                try:
                    positions = await _state.ex.get_positions()
                except Exception as pos_e:
                    logger.error(f"[MAIN_LOOP] Error fetching positions: {_describe_exception(pos_e)}")
                    positions = []

                # Refresh unresolved-order tracking once per cycle too (CR-2)
                # so a symbol blocked by a stale/pending order unblocks itself
                # as soon as that order actually resolves, instead of staying
                # blocked for the rest of the process lifetime after the one
                # startup check.
                try:
                    open_orders = await _state.ex.get_orders(status="new", limit=100)
                    _state.symbols_with_unresolved_orders = {
                        o.get("symbol", "?").replace("/", "") for o in open_orders
                    }
                except Exception as ord_e:
                    logger.debug(f"[MAIN_LOOP] Error fetching open orders (non-fatal): {ord_e}")

                for symbol, bar_result in zip(settings.SYMBOLS, bar_results, strict=False):
                    try:
                        if isinstance(bar_result, Exception):
                            logger.error(f"[MAIN_LOOP] Error fetching bar for {symbol}: {bar_result}")
                            continue
                        latest_bar_df = bar_result
                        if latest_bar_df.is_empty():
                            continue

                        current_price = latest_bar_df["close"][0]

                        # Feed the rolling price history used for real
                        # cross-asset correlation (RiskManager.reserve_position_slot).
                        # No extra API calls -- reuses the bar already fetched
                        # for this cycle's signal processing.
                        if _state.risk_manager is not None:
                            _state.risk_manager.record_price_for_correlation(symbol, current_price)

                        # Stale-price check: a bar timestamp far in the past
                        # (exchange feed stuck, cached data being replayed,
                        # clock skew) can silently feed position sizing/stop
                        # checks with an outdated price -- see
                        # ADVERSARIAL_AUDIT_2026-09-20.md §16, missing-alerts.
                        # Fire-and-forget so a slow alert path never adds
                        # latency to the per-symbol dispatch loop.
                        try:
                            bar_ts_raw = latest_bar_df["timestamp"][0]
                            bar_dt = bar_ts_raw if isinstance(bar_ts_raw, datetime) else datetime.fromisoformat(str(bar_ts_raw).replace("Z", "+00:00"))
                            if bar_dt.tzinfo is None:
                                bar_dt = bar_dt.replace(tzinfo=UTC)
                            age_sec = (datetime.now(UTC) - bar_dt).total_seconds()
                            if age_sec > settings.STALE_PRICE_MAX_AGE_SEC:
                                # Tracked (not bare fire-and-forget) so the task
                                # isn't only weakly referenced by the event loop
                                # -- matches this same function's active_tasks
                                # pattern used for process_signal_for_symbol
                                # below, and exchange.py's _background_tasks
                                # pattern for its own alert tasks. Found
                                # inconsistent (bare create_task) 2026-09-21
                                # verifying an external review's claim.
                                stale_alert_task = asyncio.create_task(
                                    get_alerting_engine().alert_stale_price(symbol, age_sec, settings.STALE_PRICE_MAX_AGE_SEC)
                                )
                                active_tasks.add(stale_alert_task)
                                stale_alert_task.add_done_callback(active_tasks.discard)
                        except Exception as stale_check_err:
                            logger.debug(f"[{symbol}] Stale-price check skipped (non-fatal): {stale_check_err}")

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
                        logger.error(f"[MAIN_LOOP] Error processing {symbol}: {_describe_exception(sym_e)}")

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
        # Force-persist crash-recovery state on graceful shutdown too, so even
        # a controlled restart keeps trailing peaks/cooldowns/scale-in counts.
        try:
            await flush_crash_recovery_state(force=True)
        except Exception:
            pass
        if _state.ex:
            await _state.ex.close()
        logger.info("Shutdown complete.")


def run_bot() -> None:
    """Synchronous wrapper for async bot."""
    asyncio.run(run_trading_bot())
