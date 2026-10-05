"""Backtest <-> live parity (audit 2026-10-04).

The backtester simulated a different bot from the one trading live: it
opened shorts (live is long-only), skipped the adversarial vetoes, never
applied MAX_HOLD_HOURS (the check uses the wall clock and the simulated
position had no entry time), ignored the killswitches / rolling-loss limit /
post-close cooldown, ran one symbol at a time with no portfolio caps, and
dropped the still-open final position from the trade list.

Signals are scripted by patching TradingStrategy.generate_trading_signal so
each live rule is exercised in isolation; no network, LLM or DB.
"""
import datetime as dt
import os

os.environ.setdefault("ALPACA_API_KEY", "dummy")
os.environ.setdefault("ALPACA_SECRET_KEY", "dummy")

from unittest.mock import AsyncMock, patch

import polars as pl
import pytest

import src.backtest as bt
from src.committee.models import BrainVote, CommitteeResult
from src.config import settings

T0 = dt.datetime(2026, 9, 1, tzinfo=dt.UTC)


def _bars(n: int, price: float = 100.0, step: float = 0.0) -> pl.DataFrame:
    rows = []
    for i in range(n):
        p = price + step * i
        rows.append({"t": (T0 + dt.timedelta(hours=i)).isoformat(), "open": p, "high": p * 1.001,
                     "low": p * 0.999, "close": p, "volume": 1000.0})
    return pl.DataFrame(rows)


def _signal(action: str, price: float, **extra) -> dict:
    s = {"action": action, "regime": "trending", "atr": price * 0.01, "confidence": 1.0,
         "expected_return_pct": 0.03, "features": {}, "reason": f"scripted_{action}"}
    s.update(extra)
    return s


def _script(actions: dict[str, list[str]], **extra):
    """Per-symbol action list indexed by call count for that symbol."""
    calls: dict[str, int] = {}

    async def fake(self, symbol, current_price, position=None):
        i = calls.get(symbol, 0)
        calls[symbol] = i + 1
        seq = actions[symbol]
        return _signal(seq[i] if i < len(seq) else "hold", current_price, **extra)

    return fake


def _committee(action="buy", score=0.8, votes=None, **kw):
    return CommitteeResult(action=action, score=score, votes=votes or [], **kw)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(settings, "MAX_SINGLE_TRADE_USD", 2500.0)
    monkeypatch.setattr(settings, "MIN_ORDER_USD", 10.0)
    monkeypatch.setattr(settings, "MAX_HOLD_HOURS", 8.0)
    monkeypatch.setattr(settings, "COOLDOWN_SECONDS_BUY", 300)
    monkeypatch.setattr(settings, "MAX_DRAWDOWN_STOP", -10.0)
    monkeypatch.setattr(settings, "DAILY_LOSS_LIMIT", -3.0)
    monkeypatch.setattr(settings, "ROLLING_LOSS_LIMIT_PCT", 1.0)
    # no live Binance fetch from calculate_position_size
    with patch("src.onchain_data.fetch_derivatives_data_sync", return_value={}):
        yield


async def _run(symbol_actions, n=12, use_committee=False, committee=None, **kw):
    bars = {s: _bars(n) for s in symbol_actions}
    with patch.object(bt.TradingStrategy, "generate_trading_signal", _script(symbol_actions, **kw.pop("signal_extra", {}))):
        if committee is not None:
            with patch.object(bt, "run_committee", AsyncMock(return_value=committee)):
                return await bt.run_portfolio_backtest(bars, start_equity=10_000.0, use_committee=use_committee, **kw)
        return await bt.run_portfolio_backtest(bars, start_equity=10_000.0, use_committee=use_committee, **kw)


@pytest.mark.asyncio
async def test_never_opens_short():
    """Live ignores a sell with no long position ('Sell vetoed: no position to sell')."""
    r = await _run({"BTC/USD": ["sell", "close"] * 6})
    assert r.n_trades == 0
    assert all(t.side == "long" for t in r.trades)


@pytest.mark.asyncio
async def test_sell_while_long_closes():
    r = await _run({"BTC/USD": ["buy", "hold", "sell"]})
    assert r.n_trades == 1
    assert r.trades[0].side == "long"


@pytest.mark.asyncio
async def test_max_hold_enforced_on_simulated_clock():
    r = await _run({"BTC/USD": ["buy"] + ["hold"] * 20}, n=20)
    assert r.trades, "position was never closed"
    t = r.trades[0]
    assert t.reason == "max_hold_time"
    held = dt.datetime.fromisoformat(t.exit_time) - dt.datetime.fromisoformat(t.entry_time)
    assert held == dt.timedelta(hours=8)


@pytest.mark.asyncio
async def test_final_open_position_recorded_as_trade():
    r = await _run({"BTC/USD": ["buy"] + ["hold"] * 4}, n=5)
    assert r.n_trades == 1
    assert r.trades[0].reason == "end_of_backtest"


@pytest.mark.asyncio
async def test_brain_b_veto_blocks_entry():
    r = await _run({"BTC/USD": ["buy"] * 6}, n=6, use_committee=True, committee=_committee(),
                   signal_extra={"expected_edge_bps": 5.0, "execution_cost_bps": 20.0})
    assert r.n_trades == 0


@pytest.mark.asyncio
async def test_disagreement_veto_blocks_entry():
    votes = [BrainVote(name=n, action=a, confidence=0.9, weight=0.25, regime="trending", reason="t") for n, a in
             (("a", "buy"), ("b", "sell"), ("c", "buy"), ("d", "sell"))]
    r = await _run({"BTC/USD": ["buy"] * 6}, n=6, use_committee=True,
                   committee=_committee(score=0.50, votes=votes))
    assert r.n_trades == 0


@pytest.mark.asyncio
async def test_committee_veto_blocks_entry():
    r = await _run({"BTC/USD": ["buy"] * 6}, n=6, use_committee=True,
                   committee=_committee(vetoed=True, veto_reason="sentinel"))
    assert r.n_trades == 0


@pytest.mark.asyncio
async def test_cooldown_after_close_blocks_immediate_reentry(monkeypatch):
    monkeypatch.setattr(settings, "COOLDOWN_SECONDS_BUY", 2 * 3600)
    # buy@0, close@1, buy@2 is inside the 2h cooldown, buy@3 is allowed
    r = await _run({"BTC/USD": ["buy", "close", "buy", "buy", "hold"]}, n=5)
    entries = [dt.datetime.fromisoformat(t.entry_time) for t in r.trades]
    assert entries == [T0, T0 + dt.timedelta(hours=3)]


@pytest.mark.asyncio
async def test_max_open_positions_enforced(monkeypatch):
    monkeypatch.setattr(settings, "MAX_OPEN_POSITIONS", 1)
    r = await _run({"BTC/USD": ["buy"] + ["hold"] * 3, "ETH/USD": ["buy"] + ["hold"] * 3}, n=4)
    assert r.n_trades == 1


@pytest.mark.asyncio
async def test_portfolio_cap_limits_notional(monkeypatch):
    monkeypatch.setattr(settings, "MAX_PORTFOLIO_PCT", None)
    monkeypatch.setattr(settings, "MAX_PORTFOLIO_VALUE", 500.0)
    r = await _run({"BTC/USD": ["buy"] + ["hold"] * 3}, n=4)
    assert r.trades and r.trades[0].qty * r.trades[0].entry_price <= 500.0 + 1e-6


@pytest.mark.asyncio
async def test_caps_off_for_single_symbol_wrapper(monkeypatch):
    """run_backtest (promotion gate / walk-forward) keeps percent-of-equity
    sizing: account caps are a portfolio-run concern."""
    monkeypatch.setattr(settings, "MAX_PORTFOLIO_PCT", None)
    monkeypatch.setattr(settings, "MAX_PORTFOLIO_VALUE", 500.0)
    with patch.object(bt.TradingStrategy, "generate_trading_signal", _script({"BTC/USD": ["buy"] + ["hold"] * 3})):
        r = await bt.run_backtest("BTC/USD", bars=_bars(4), start_equity=10_000.0, use_committee=False)
    assert r.trades and r.trades[0].qty * r.trades[0].entry_price > 500.0


def test_guard_drawdown_trips_and_liquidates():
    g = bt.SimulatedAccountGuards(start_equity=10_000.0)
    assert g.update(T0, 10_000.0, has_positions=True) is None
    assert g.update(T0 + dt.timedelta(hours=1), 8_800.0, has_positions=True) == "liquidate"
    assert g.entries_blocked()


def test_guard_drawdown_clears_after_flat_cooldown(monkeypatch):
    monkeypatch.setattr(settings, "DRAWDOWN_KILLSWITCH_COOLDOWN_HOURS", 24.0)
    monkeypatch.setattr(settings, "DAILY_LOSS_LIMIT", -50.0)
    g = bt.SimulatedAccountGuards(start_equity=10_000.0)
    g.update(T0, 10_000.0, has_positions=True)
    g.update(T0 + dt.timedelta(hours=1), 8_800.0, has_positions=True)
    g.update(T0 + dt.timedelta(hours=10), 8_800.0, has_positions=False)
    assert g.entries_blocked()
    g.update(T0 + dt.timedelta(hours=26), 8_800.0, has_positions=False)
    assert not g.entries_blocked()


def test_guard_daily_loss_trips_and_clears_next_day():
    g = bt.SimulatedAccountGuards(start_equity=10_000.0)
    g.update(T0, 10_000.0, has_positions=True)
    assert g.update(T0 + dt.timedelta(hours=1), 9_650.0, has_positions=True) == "liquidate"  # -3.5% day
    assert g.entries_blocked()
    g.update(T0 + dt.timedelta(days=1, hours=1), 9_650.0, has_positions=False)
    assert not g.entries_blocked()


def test_guard_rolling_loss_blocks_entries():
    g = bt.SimulatedAccountGuards(start_equity=10_000.0)
    g.update(T0, 10_000.0, has_positions=False)
    g.record_close(T0 + dt.timedelta(hours=1), pnl=-150.0)  # -1.5% of peak in 6h window
    g.update(T0 + dt.timedelta(hours=2), 9_850.0, has_positions=False)
    assert g.entries_blocked()
    g.update(T0 + dt.timedelta(hours=8), 9_850.0, has_positions=False)  # window passed
    assert not g.entries_blocked()


def test_calmar_uses_drawdown_magnitude():
    r = bt.BacktestResult(symbol="X", start_equity=100.0, end_equity=110.0)
    r.equity_curve = [100.0, 120.0, 90.0, 110.0]
    r.trades = [bt.BacktestTrade(symbol="X", side="long", entry_price=1, exit_price=1.1, qty=1,
                                 entry_time="a", exit_time="b", pnl=10.0, pnl_pct=10.0, reason="x")]
    bt._finalize_result(r, 110.0)
    assert r.max_drawdown_pct < 0
    assert r.calmar not in (0.0, float("inf"))


# ── Timeframes, warm-up, entry funnel ─────────────────────────────────────────
# Regression: BacktestExchange.get_bars ignored `timeframe`, so the strategy's
# regime analysis (live: analyze_market_regime(timeframe="1D")) ran on HOURLY
# bars. Hourly ATR% (~0.5%) sits under the 1.25% low-vol cutoff ~99% of the
# time vs ~0% for daily ATR%, so a 60-day backtest labelled 4239/4296 bars
# "low_volatility" -- a regime the live bot is essentially never in.

def _hourly(n_hours: int, start=T0) -> pl.DataFrame:
    rows = [{"t": (start + dt.timedelta(hours=i)).isoformat(), "open": 100.0 + i, "high": 101.0 + i,
             "low": 99.0 + i, "close": 100.5 + i, "volume": 10.0} for i in range(n_hours)]
    return pl.DataFrame(rows)


@pytest.mark.asyncio
async def test_exchange_resamples_daily_point_in_time():
    ex = bt.BacktestExchange({"X": _hourly(60)})
    ex.current_time = (T0 + dt.timedelta(hours=29)).isoformat()   # day 2, 06:00 bar
    d = await ex.get_bars("X", "1D", 100)
    assert len(d) == 2                                   # day 1 + in-progress day 2
    day1, day2 = d.row(0, named=True), d.row(1, named=True)
    assert (day1["open"], day1["close"], day1["high"], day1["low"]) == (100.0, 123.5, 124.0, 99.0)
    assert (day2["open"], day2["close"]) == (124.0, 129.5)   # only bars <= current_time
    assert day1["volume"] == 240.0


@pytest.mark.asyncio
async def test_exchange_hourly_and_finer_unchanged():
    ex = bt.BacktestExchange({"X": _hourly(10)})
    ex.current_time = (T0 + dt.timedelta(hours=4)).isoformat()
    assert len(await ex.get_bars("X", "1Hour", 100)) == 5
    assert len(await ex.get_bars("X", "5Min", 100)) == 5   # can't synthesize finer bars
    assert len(await ex.get_bars("X", "4Hour", 100)) == 2


@pytest.mark.asyncio
async def test_start_time_skips_warmup_bars_but_keeps_history():
    seen = []

    async def fake(self, symbol, current_price, position=None):
        bars = await self.exchange.get_bars(symbol, "1Hour", 1000)
        seen.append(len(bars))
        return _signal("hold", current_price)

    with patch.object(bt.TradingStrategy, "generate_trading_signal", fake):
        r = await bt.run_portfolio_backtest({"BTC/USD": _bars(10)}, start_equity=10_000.0, use_committee=False,
                                            start_time=(T0 + dt.timedelta(hours=6)).isoformat())
    assert seen == [7, 8, 9, 10]          # only evaluated from hour 6, full history visible
    assert len(r.equity_curve) == 4


@pytest.mark.asyncio
async def test_entry_funnel_counts_rejections():
    r = await _run({"BTC/USD": ["buy"] * 6}, n=6, use_committee=True, committee=_committee(),
                   signal_extra={"expected_edge_bps": 5.0, "execution_cost_bps": 20.0})
    f = r.entry_funnel
    assert f["buy_signals"] == 6
    assert f["veto_brain_b"] == 6
    assert f.get("opened", 0) == 0
