"""Tests for the edge-study harness math (hand-computed values).

These pin the backtest arithmetic and the bootstrap so the negative result is
reproducible, not merely asserted. No network, no DB.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.edge_study import (
    backtest,
    block_bootstrap_p,
    buy_hold,
    h1_momentum,
)


def test_buy_hold_matches_hand_computation():
    # 100 -> 110 is +10%, minus 56 bps round-trip = 0.10 - 0.0056 = 0.0944
    out = buy_hold([100.0, 110.0], 56.0)
    assert abs(out["equity"] - 1.0944) < 1e-9


def test_long_flat_single_trade_is_net_of_cost():
    # Always-long signal; 100 -> 110 over two bars, one round trip.
    px = [100.0, 105.0, 110.0]
    out = backtest(px, lambda p, i: 1, 56.0, allow_short=False)
    assert out["n_trades"] == 1
    assert abs(out["trades"][0] - (0.10 - 0.0056)) < 1e-9


def test_long_short_inverts_when_signal_is_zero():
    # Signal 0 -> target -1 (short). A rising market must LOSE for the short.
    px = [100.0, 110.0]
    out = backtest(px, lambda p, i: 0, 56.0, allow_short=True)
    assert out["n_trades"] == 1
    assert out["trades"][0] < 0  # short into a +10% move, plus cost


def test_momentum_lookahead_guard_returns_flat_before_lookback():
    px = [1.0] * 10
    assert h1_momentum(px, 24, 5) == 0   # not enough history -> flat, no look-ahead


def test_bootstrap_negative_mean_gives_high_pvalue():
    trades = [-0.01] * 50                # uniformly losing
    assert block_bootstrap_p(trades) == 1.0


def test_bootstrap_strong_positive_mean_gives_low_pvalue():
    trades = [0.02] * 50                 # uniformly winning
    assert block_bootstrap_p(trades) == 0.0
