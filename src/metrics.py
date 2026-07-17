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
