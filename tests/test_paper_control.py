"""Tests for the buy-and-hold control account (src/paper_control.py).

The control is the bar the bot must beat, so its math must be right: equal
weight at entry, marked to market, and it must never re-initialize (which would
silently reset the benchmark to 0%).
"""

from __future__ import annotations

import json

import pytest

from src import paper_control


@pytest.fixture(autouse=True)
def _tmp_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_control, "_LEDGER_PATH", str(tmp_path / "control.json"))


def test_initializes_equal_weight_at_100():
    ledger = paper_control.initialize({"BTC/USD": 1000.0, "ETH/USD": 200.0, "SOL/USD": 50.0})
    assert ledger["initial_usd"] == 100.0
    # Equal-weight: each position worth $100/3 at entry (ledger rounds to 4dp).
    for pos in ledger["positions"].values():
        assert pos["qty"] * pos["entry_price"] == pytest.approx(100.0 / 3, rel=1e-3)
    assert sum(ledger["weights"].values()) == pytest.approx(1.0, abs=1e-3)


def test_value_tracks_price_moves():
    ledger = paper_control.initialize({"BTC/USD": 1000.0, "ETH/USD": 200.0, "SOL/USD": 50.0})
    # Every coin doubles -> holdings double to $200, minus the $0.25 entry fee.
    s = paper_control.value(ledger, {"BTC/USD": 2000.0, "ETH/USD": 400.0, "SOL/USD": 100.0})
    assert s["equity_usd"] == pytest.approx(199.75, abs=1e-3)
    assert s["return_pct"] == pytest.approx(99.75, abs=1e-3)


def test_update_initializes_once_then_appends():
    p1 = {"BTC/USD": 1000.0, "ETH/USD": 200.0, "SOL/USD": 50.0}
    paper_control.update(p1)
    # A later, much higher price must NOT reset the entry.
    p2 = {"BTC/USD": 3000.0, "ETH/USD": 200.0, "SOL/USD": 50.0}
    s2 = paper_control.update(p2)
    ledger = paper_control.load_ledger()
    assert ledger["positions"]["BTC/USD"]["entry_price"] == 1000.0
    assert len(ledger["snapshots"]) == 2
    assert s2["equity_usd"] > 100.0


def test_missing_price_marks_at_last_known_not_zero():
    p1 = {"BTC/USD": 1000.0, "ETH/USD": 200.0, "SOL/USD": 50.0}
    ledger = paper_control.initialize(p1)
    # ETH price unavailable this call -> must hold its entry value, not zero out.
    s = paper_control.value(ledger, {"BTC/USD": 1000.0, "SOL/USD": 50.0})
    assert s["per_symbol"]["ETH/USD"]["value_usd"] == pytest.approx(100.0 / 3, rel=1e-3)
    # Holdings still $100, less the $0.25 entry fee.
    assert s["equity_usd"] == pytest.approx(99.75, abs=1e-3)


def test_summary_empty_before_first_run():
    assert paper_control.summary() == {}


def test_entry_fee_is_charged_at_25bps():
    ledger = paper_control.initialize({"BTC/USD": 1000.0, "ETH/USD": 200.0, "SOL/USD": 50.0})
    # 0.25% of $100 notional = $0.25 total, split equally across three coins.
    assert ledger["entry_fee_bps"] == 25.0
    assert ledger["entry_fees_usd"] == pytest.approx(0.25, abs=1e-4)
    # Immediately after entry (prices unchanged) equity is $100 - $0.25 = $99.75.
    s = paper_control.value(ledger, {"BTC/USD": 1000.0, "ETH/USD": 200.0, "SOL/USD": 50.0})
    assert s["equity_usd"] == pytest.approx(99.75, abs=1e-4)
    assert s["return_pct"] == pytest.approx(-0.25, abs=1e-4)


def test_entry_fee_reduces_return_vs_no_fee():
    prices = {"BTC/USD": 1000.0, "ETH/USD": 200.0, "SOL/USD": 50.0}
    with_fee = paper_control.initialize(prices, entry_fee_bps=25.0)
    no_fee = paper_control.initialize(prices, entry_fee_bps=0.0)
    up = {"BTC/USD": 2000.0, "ETH/USD": 400.0, "SOL/USD": 100.0}
    sv = paper_control.value(with_fee, up)["equity_usd"]
    sn = paper_control.value(no_fee, up)["equity_usd"]
    assert sv == pytest.approx(sn - 0.25, rel=1e-6)


def test_ledger_is_written_atomically():
    paper_control.update({"BTC/USD": 1000.0, "ETH/USD": 200.0, "SOL/USD": 50.0})
    with open(paper_control._LEDGER_PATH) as f:
        data = json.load(f)
    assert data["initial_usd"] == 100.0
