"""Tests for the net-expectancy research harness.

Covers the two behaviours the task explicitly asks to test: the cost model and
the same-bar stop/target rule. All tests exercise the real simulator (no mocks).

Run:  python -m pytest research/tests/test_net_expectancy.py -q
"""

import numpy as np
import pandas as pd
import pytest

from research import net_expectancy as ne


def _bars(opens, highs, lows, closes):
    ts = pd.date_range("2024-01-01", periods=len(opens), freq="1h", tz="UTC")
    return pd.DataFrame({"ts": ts, "open": opens, "high": highs,
                         "low": lows, "close": closes, "volume": 1.0})


# ── Cost model ──────────────────────────────────────────────────────────────
def test_cost_model_tier1_matches_alpaca():
    c = ne.CostModel()
    assert c.taker_bps == 25.0 and c.maker_bps == 15.0 and c.slippage_bps == 5.0
    # taker-in / taker-out = 2*(25+5)
    assert c.taker_round_trip_bps == 60.0
    assert c.entry_cost_bps("market") == 30.0
    assert c.entry_cost_bps("limit") == 15.0     # maker fee, no slippage
    assert c.exit_cost_bps() == 30.0


def test_cost_model_is_configurable():
    c = ne.CostModel(taker_bps=10, maker_bps=2, slippage_bps=0)
    assert c.taker_round_trip_bps == 20.0
    assert c.entry_cost_bps("limit") == 2.0


def test_net_is_gross_minus_round_trip_cost():
    # Signal bar 0; entry at bar 1 open = 100; target +3% -> exit 103 exactly.
    df = _bars([100, 100, 101, 102, 103, 103],
               [100, 101, 103.5, 103, 103, 103],
               [100, 100, 100, 101, 102, 103],
               [100, 101, 103, 103, 103, 103])
    sig = np.zeros(len(df), bool)
    sig[0] = True
    trades, _ = ne.simulate("BTC/USD", df, sig, ne.CostModel())
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_price == 100.0                 # next bar's OPEN
    assert t.exit_price == pytest.approx(103.0)   # +3% target level
    assert t.reason == "profit_target"
    assert t.gross_pct == pytest.approx(3.0)
    assert t.net_pct == pytest.approx(3.0 - 0.60)  # 60 bps round trip


def test_limit_entry_pays_maker_fee_not_taker():
    df = _bars([100, 100, 101, 102, 103, 103],
               [100, 101, 103.5, 103, 103, 103],
               [100, 99, 100, 101, 102, 103],   # bar1 low 99 <= limit 100 -> fill
               [100, 101, 103, 103, 103, 103])
    sig = np.zeros(len(df), bool)
    sig[0] = True
    mkt, _ = ne.simulate("BTC/USD", df, sig, ne.CostModel(), entry_type="market")
    lim, nf = ne.simulate("BTC/USD", df, sig, ne.CostModel(), entry_type="limit")
    assert nf == 0 and len(lim) == 1
    # limit entry saves (taker+slip) - maker = (25+5) - 15 = 15 bps
    assert lim[0].net_pct - mkt[0].net_pct == pytest.approx(0.15)


# ── Same-bar stop/target rule ───────────────────────────────────────────────
def test_same_bar_stop_and_target_takes_stop_first():
    # Bar 1 touches BOTH +3% (103) and -4% (96). Must resolve as stop_loss.
    df = _bars([100, 100, 101, 102, 103, 103],
               [100, 106.0, 101, 102, 103, 103],   # high 106 > tp 103
               [100, 95.0, 100, 101, 102, 103],    # low 95 < sl 96
               [100, 100, 101, 102, 103, 103])
    sig = np.zeros(len(df), bool)
    sig[0] = True
    trades, _ = ne.simulate("BTC/USD", df, sig, ne.CostModel())
    assert trades[0].reason == "stop_loss"
    assert trades[0].exit_price == pytest.approx(96.0)   # 100 * (1 - 0.04)


def test_stop_first_only_applies_on_same_bar():
    # Bar 1 hits target only; bar 2 hits stop only -> profit_target wins.
    df = _bars([100, 100, 102, 101, 100, 99],
               [100, 103.5, 102, 101, 100, 99],
               [100, 100, 99, 96.5, 99, 98],
               [100, 103, 101, 97, 99, 98])
    sig = np.zeros(len(df), bool)
    sig[0] = True
    trades, _ = ne.simulate("BTC/USD", df, sig, ne.CostModel())
    assert trades[0].reason == "profit_target"


def test_stop_loss_level_and_net():
    df = _bars([100, 100, 99, 98, 97, 96],
               [100, 100, 99, 98, 97, 96],
               [100, 96.0, 97, 96, 95, 95],
               [100, 99, 98, 97, 96, 95])
    sig = np.zeros(len(df), bool)
    sig[0] = True
    trades, _ = ne.simulate("BTC/USD", df, sig, ne.CostModel())
    assert trades[0].reason == "stop_loss"
    assert trades[0].gross_pct == pytest.approx(-4.0)
    assert trades[0].net_pct == pytest.approx(-4.0 - 0.60)


# ── Fill timing, max-hold, limit non-fill ───────────────────────────────────
def test_entry_uses_next_bar_open_not_signal_bar_close():
    df = _bars([100, 250, 250, 250, 250, 250],
               [100, 250, 250, 250, 250, 250],
               [100, 250, 250, 250, 250, 250],
               [100, 250, 250, 250, 250, 250])
    sig = np.zeros(len(df), bool)
    sig[0] = True
    trades, _ = ne.simulate("BTC/USD", df, sig, ne.CostModel())
    assert trades[0].entry_price == 250.0   # bar 1 open, NOT bar 0 close (100)


def test_max_hold_exits_at_time_limit():
    n = 30
    flat = [100.0] * n
    df = _bars(flat, flat, flat, flat)  # never touches tp/sl
    sig = np.zeros(n, bool)
    sig[0] = True
    trades, _ = ne.simulate("BTC/USD", df, sig, ne.CostModel(), max_hold_hours=4.0)
    assert trades[0].reason == "max_hold"
    assert trades[0].bars_held == 4


def test_limit_entry_unfilled_when_price_never_trades_through():
    df = _bars([100, 105, 106, 107, 108, 109],
               [100, 106, 107, 108, 109, 110],
               [100, 104, 105, 106, 107, 108],   # bar1 low 104 > limit 100
               [100, 105, 106, 107, 108, 109])
    sig = np.zeros(len(df), bool)
    sig[0] = True
    trades, nf = ne.simulate("BTC/USD", df, sig, ne.CostModel(), entry_type="limit")
    assert trades == [] and nf == 1


# ── Metrics plumbing ────────────────────────────────────────────────────────
def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    net = rng.normal(0.5, 1.0, size=500)
    lo, hi = ne.bootstrap_ci(net)
    assert lo < net.mean() < hi


def test_summarize_profit_factor_and_dd():
    df = _bars([100, 100, 101, 102, 103, 103, 104, 105, 106, 107],
               [100, 103.5, 101, 102, 103, 103, 104, 105, 106, 107],
               [100, 100, 100, 101, 102, 96, 104, 105, 106, 107],
               [100, 103, 101, 102, 103, 97, 104, 105, 106, 107])
    sig = np.zeros(len(df), bool)
    sig[0] = True
    sig[6] = True
    trades, _ = ne.simulate("BTC/USD", df, sig, ne.CostModel())
    s = ne.summarize(trades, "t")
    assert s["trades"] == 2
    assert 0.0 <= s["win_rate"] <= 1.0
    assert s["max_dd_pct"] <= 0.0


def test_equal_subperiods_partition():
    df = _bars([100] * 12, [100] * 12, [100] * 12, [100] * 12)
    sig = np.zeros(12, bool)
    sig[[0, 5, 9]] = True
    trades, _ = ne.simulate("BTC/USD", df, sig, ne.CostModel(), max_hold_hours=2.0)
    rows = ne.equal_subperiods(trades, df["ts"].iloc[0], df["ts"].iloc[-1], 4)
    assert len(rows) == 4
    assert sum(r["trades"] for r in rows) == len(trades)
