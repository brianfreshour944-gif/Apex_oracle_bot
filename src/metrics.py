"""Prometheus metrics for the trading bot."""  
 
from __future__ import annotations 
 
from src.config import settings 
from src.logging_config import get_logger 
 
logger = get_logger("metrics") 
 
try: 
    from prometheus_client import Counter, Gauge, Histogram 
    _PROM_AVAILABLE = True 
except Exception: 
    _PROM_AVAILABLE = False 
    logger.warning("prometheus_client not installed - metrics disabled") 
 
if _PROM_AVAILABLE:
    LOOP_DURATION = Histogram("bot_loop_seconds", "Time spent in one trading cycle")
    ORDERS_TOTAL = Counter("bot_orders_total", "Orders placed", ["side", "status"])
    CYCLES_TOTAL = Counter("bot_cycles_total", "Trading cycles completed")
    ERRORS_TOTAL = Counter("bot_errors_total", "Errors by stage", ["stage"])
    DRAWDOWN = Gauge("bot_drawdown_pct", "Current drawdown percent")
    EQUITY = Gauge("bot_equity_usd", "Current account equity")
    OPEN_POSITIONS = Gauge("bot_open_positions", "Number of open positions")
    CIRCUIT_STATE = Gauge("bot_circuit_state", "Circuit breaker state", ["name"])
    REGIME_COUNTS = Counter("bot_regime_total", "Signals by regime", ["regime"])
    ADAPTIVE_BRAIN_WEIGHT = Gauge("bot_adaptive_brain_weight", "Adaptive per-brain weight", ["brain", "regime"])
    ADAPTIVE_MODEL_VERSION = Gauge("bot_adaptive_model_version", "Adaptive learner state version")
    ADAPTIVE_SAMPLE_COUNT = Gauge("bot_adaptive_sample_count", "Realized outcomes learned from")
    ADAPTIVE_LAST_UPDATE = Gauge("bot_adaptive_last_update_ts", "Unix ts of last adaptive weight update")
else:
    class _Noop: 
        def __getattr__(self, _): 
            def _noop(*a, **k): 
                return _NullCtx() 
            return _noop 
    class _NullCtx: 
        def __enter__(self): 
            return self 
        def __exit__(self, *a): 
            return False 
        def inc(self, *a, **k): 
            pass 
        def set(self, *a, **k): 
            pass 
        def observe(self, *a, **k): 
            pass 
        def time(self, *a, **k): 
            return _NullCtx() 
    _m = _Noop()
    LOOP_DURATION = CYCLES_TOTAL = ERRORS_TOTAL = REGIME_COUNTS = _m
    ORDERS_TOTAL = DRAWDOWN = EQUITY = OPEN_POSITIONS = CIRCUIT_STATE = _m
    ADAPTIVE_BRAIN_WEIGHT = ADAPTIVE_MODEL_VERSION = _m
    ADAPTIVE_SAMPLE_COUNT = ADAPTIVE_LAST_UPDATE = _m
 
def record_order(side, status): 
    if _PROM_AVAILABLE: 
        ORDERS_TOTAL.labels(side=side, status=status).inc() 
 
def record_error(stage): 
    if _PROM_AVAILABLE: 
        ERRORS_TOTAL.labels(stage=stage).inc() 
 
def record_regime(regime): 
    if _PROM_AVAILABLE: 
        REGIME_COUNTS.labels(regime=regime).inc() 
 
def update_portfolio(equity, drawdown_pct, open_positions): 
    if _PROM_AVAILABLE: 
        EQUITY.set(equity) 
        DRAWDOWN.set(drawdown_pct) 
        OPEN_POSITIONS.set(open_positions) 
 
def update_circuit(name, state):
    if _PROM_AVAILABLE:
        mapping = {"closed": 0, "open": 1, "half_open": 2}
        CIRCUIT_STATE.labels(name=name).set(mapping.get(state, -1))


def update_adaptive_metrics(snapshot):
    """Publish current adaptive learner state to Prometheus.

    ``snapshot`` is the dict returned by ``AdaptiveMetaLearner.snapshot()``:
    {version, sample_count, last_update, weights: {regime: {brain: weight}}}.
    """
    if not _PROM_AVAILABLE or not snapshot:
        return
    try:
        ADAPTIVE_MODEL_VERSION.set(float(snapshot.get("version", 0)))
        ADAPTIVE_SAMPLE_COUNT.set(float(snapshot.get("sample_count", 0)))
        last = snapshot.get("last_update")
        if last:
            from datetime import datetime
            try:
                ts = datetime.fromisoformat(str(last).replace("Z", "+00:00")).timestamp()
                ADAPTIVE_LAST_UPDATE.set(ts)
            except Exception:
                pass
        for regime, weights in (snapshot.get("weights") or {}).items():
            for brain, weight in weights.items():
                ADAPTIVE_BRAIN_WEIGHT.labels(brain=brain, regime=regime).set(float(weight))
    except Exception as e:
        logger.warning(f"update_adaptive_metrics failed: {e}")


async def alert_weight_change(regime, old_weights, new_weights, sample_count, threshold=0.05):
    """Send a Telegram alert when adaptive weights move materially for a regime.

    Returns True if an alert was dispatched. ``threshold`` is the minimum
    absolute change in any single brain weight that counts as material.
    """
    try:
        deltas = {
            b: new_weights.get(b, 0.0) - old_weights.get(b, 0.0)
            for b in set(new_weights) | set(old_weights)
        }
        max_delta = max((abs(d) for d in deltas.values()), default=0.0)
        if max_delta < threshold:
            return False

        from src.telegram_alerts import send_telegram_alert

        lines = [
            "🧠 <b>Adaptive Weights Updated</b>",
            f"Regime: <b>{regime}</b> | Samples: {sample_count}",
        ]
        for brain in sorted(deltas, key=lambda b: abs(deltas[b]), reverse=True):
            arrow = "▲" if deltas[brain] > 0 else "▼"
            lines.append(
                f"{arrow} {brain}: {old_weights.get(brain, 0.0):.2f} → {new_weights.get(brain, 0.0):.2f}"
            )
        return await send_telegram_alert("\n".join(lines))
    except Exception as e:
        logger.warning(f"alert_weight_change failed: {e}")
        return False
