"""Deterministic unit tests for the frozen 72h momentum harness.

These pin the harness math with hand-computed values. No network, no DB.
They do not modify or depend on existing tests.
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.momentum72_study import (
    COST_ONE_WAY,
    MOM_LOOKBACK,
    VOL_TARGET,
    block_bootstrap_p,
    buy_hold,
    information_coefficient,
    log_returns,
    run,
    weights,
)


def _ramp(n: int, start: float = 100.0, rate: float = 1.01) -> list[float]:
    return [start * (rate ** i) for i in range(n)]


def _reference_engine(closes: list[float], long_short: bool = False, delay: int = 0):
    """Independent, literal transcription of the frozen spec §3.

    Written from the spec text (not from run()) so it is a genuine cross-check.
    Returns (gross_log, net_log, trades_gross, trades_net).
    """
    import math as _math
    import statistics as _st

    n = len(closes)
    lr = [ _math.log(closes[j + 1] / closes[j]) for j in range(n - 1) ]
    # weights, per spec: mom over 72, sample stdev of the 72 returns ending at i
    w = [0.0] * n
    for i in range(n):
        if i < MOM_LOOKBACK:
            continue
        mom = _math.log(closes[i] / closes[i - MOM_LOOKBACK])
        window = lr[i - MOM_LOOKBACK:i]
        if len(window) < MOM_LOOKBACK:
            continue
        sd = _st.stdev(window)
        if sd <= 0:
            continue
        scale = min(1.0, VOL_TARGET / sd)
        w[i] = (scale if mom > 0 else -scale) if long_short else (scale if mom > 0 else 0.0)

    gross = net = 0.0
    prev = 0.0
    tg: list[float] = []
    tn: list[float] = []
    cg = cn = 0.0
    in_t = False
    for i in range(n - 1):
        src = i - delay
        tgt = w[src] if src >= 0 else 0.0
        dw = abs(tgt - prev)
        rg = tgt * lr[i]
        rn = rg - dw * COST_ONE_WAY / 1e4
        gross += rg
        net += rn
        if prev == 0 and tgt != 0:
            in_t, cg, cn = True, 0.0, 0.0
        if in_t:
            cg += rg
            cn += rn
            if tgt == 0:
                tg.append(cg)
                tn.append(cn)
                in_t = False
        prev = tgt
    if in_t:
        tg.append(cg)
        tn.append(cn)
    return gross, net, tg, tn


def test_log_returns_basic():
    lr = log_returns([100.0, 110.0])
    assert abs(lr[0] - math.log(1.1)) < 1e-12


def test_warmup_has_zero_exposure():
    closes = _ramp(MOM_LOOKBACK + 5)
    w = weights(closes)
    # bars before the lookback cannot have a signal
    assert all(x == 0.0 for x in w[:MOM_LOOKBACK])
    # after warmup in a rising series, exposure is positive
    assert w[-1] > 0.0


def test_flat_when_momentum_negative():
    # strictly falling series -> momentum < 0 -> flat (long/flat)
    closes = [100.0 * (0.99 ** i) for i in range(MOM_LOOKBACK + 10)]
    w = weights(closes)
    assert all(x == 0.0 for x in w[MOM_LOOKBACK:])


def test_vol_scaling_is_capped_at_one():
    # tiny volatility -> VOL_TARGET/sd would exceed 1 -> must be capped at 1.0
    closes = [100.0 + 0.0001 * i for i in range(MOM_LOOKBACK + 10)]
    w = weights(closes)
    assert max(w) <= 1.0
    assert w[-1] == 1.0


def test_exposure_inverse_to_volatility():
    # Construct two rising series with different noise levels; the noisier one
    # must receive smaller (or equal) exposure.
    calm = [100.0 * (1.001 ** i) for i in range(MOM_LOOKBACK + 50)]
    wild = [100.0 * (1.001 ** i) * (1 + 0.02 * ((-1) ** i))
            for i in range(MOM_LOOKBACK + 50)]
    wc, ww = weights(calm), weights(wild)
    assert wc[-1] >= ww[-1]


def test_cost_charged_on_exposure_change():
    # A single entry at full exposure then hold: exactly one change of size 1.
    closes = _ramp(MOM_LOOKBACK + 3)
    r = run(closes)
    assert r["n_exposure_changes"] == 1
    assert abs(r["cost_bps"] - COST_ONE_WAY) < 1e-9


def test_gross_exceeds_net_by_cost():
    closes = _ramp(MOM_LOOKBACK + 5)
    r = run(closes)
    assert r["gross_log"] > r["net_log"]


def test_buy_hold_matches_hand_computation():
    closes = [100.0, 110.0]
    bh = buy_hold(closes, cost_bps=56.0)
    expected = math.exp(math.log(1.1) - 0.0056) - 1
    assert abs(bh["net_total_pct"] / 100 - expected) < 1e-9


def test_no_lookahead_weight_is_causal():
    # Truncating the future must not change past weights.
    closes = _ramp(MOM_LOOKBACK + 100)
    w_full = weights(closes)
    w_cut = weights(closes[:MOM_LOOKBACK + 50])
    assert w_full[:MOM_LOOKBACK + 50] == w_cut


def test_bootstrap_negative_mean_high_pvalue():
    assert block_bootstrap_p([-0.01] * 40) == 1.0


def test_bootstrap_positive_mean_low_pvalue():
    assert block_bootstrap_p([0.02] * 40) == 0.0


def test_vol_target_is_the_frozen_constant():
    assert VOL_TARGET == 0.005497


# --- Regression tests for the corrected return accounting (spec §3) ----------
# Each pins an INDEPENDENTLY hand-computed value, not a mere property.


def test_primary_return_timing_matches_spec_jump_case():
    """Spec §3: return(i+1) = w_i*logret_{i+1}. The jump case isolates it.

    closes = [100]*72 + [101, 110]: the only non-zero return is bar 73,
    log(110/101). The signal at bar 72 is w=1.0 (rising) so the PRIMARY must
    earn exactly that return; a one-bar-late implementation would earn 0.
    """
    closes = [100.0] * 72 + [101.0, 110.0]
    w = weights(closes)
    assert w[72] == 1.0
    expected = math.log(110.0 / 101.0)          # 0.08535984895115685
    assert abs(run(closes, delay=0)["gross_log"] - expected) < 1e-12
    # The one-bar-delayed sensitivity earns nothing here (signal arrives too late).
    assert abs(run(closes, delay=1)["gross_log"] - 0.0) < 1e-12


def test_delay_one_equals_spec_one_bar_delayed_convention(monkeypatch):
    """Spec: sensitivity is execution delayed by one bar => w_{i-1}*logret_{i+1}."""
    monkeypatch.setattr("scripts.momentum72_study.MOM_LOOKBACK", 2)
    monkeypatch.setattr("scripts.momentum72_study.VOL_LOOKBACK", 2)
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
    w = weights(closes)
    lr = log_returns(closes)
    primary = sum(w[i] * lr[i] for i in range(len(closes) - 1))
    delayed = sum((w[i - 1] if i >= 1 else 0.0) * lr[i] for i in range(len(closes) - 1))
    assert abs(run(closes, delay=0)["gross_log"] - primary) < 1e-12
    assert abs(run(closes, delay=1)["gross_log"] - delayed) < 1e-12
    assert abs(primary - delayed) > 1e-6          # the two differ: not a no-op


def test_run_matches_literal_spec_transcription_on_a_mixed_series():
    """Cross-check run() against an independent transcription of spec §3."""
    closes = [100.0]
    step = [1.0, -0.5, 2.0, -1.5, 0.3, 0.7, -2.2, 1.1, -0.4, 0.9, -0.8, 1.6]
    for s in step * 8:
        closes.append(closes[-1] + s)
    ref_gross, ref_net, ref_tg, ref_tn = _reference_engine(closes)
    r = run(closes)
    assert abs(r["gross_log"] - ref_gross) < 1e-12
    assert abs(r["net_log"] - ref_net) < 1e-12
    assert len(r["trades_gross"]) == len(ref_tg)
    assert all(abs(a - b) < 1e-12 for a, b in zip(r["trades_gross"], ref_tg, strict=False))
    assert all(abs(a - b) < 1e-12 for a, b in zip(r["trades_net"], ref_tn, strict=False))


def test_full_round_trip_net_charges_entry_and_exit(monkeypatch):
    """A closed trade nets gross MINUS entry cost MINUS exit cost (56 bps total).

    lookback shrunk to 2; closes [100,100,100,200,100,100,100].
      w[3] = min(1, VOL_TARGET / stdev([log(100/100), log(200/100)]))
           = min(1, 0.005497 / (log(2)/sqrt(2)))
      gross = w[3] * log(100/200)
      net   = gross - w[3]*28bps (entry) - w[3]*28bps (exit)
    """
    monkeypatch.setattr("scripts.momentum72_study.MOM_LOOKBACK", 2)
    monkeypatch.setattr("scripts.momentum72_study.VOL_LOOKBACK", 2)
    closes = [100.0, 100.0, 100.0, 200.0, 100.0, 100.0, 100.0]
    lr = log_returns(closes)
    sd = statistics.stdev([lr[1], lr[2]])
    w3 = min(1.0, VOL_TARGET / sd)
    expected_gross = w3 * lr[3]
    expected_net = expected_gross - w3 * COST_ONE_WAY / 1e4 - w3 * COST_ONE_WAY / 1e4
    r = run(closes)
    assert len(r["trades_net"]) == 1
    assert abs(r["trades_gross"][0] - expected_gross) < 1e-12
    assert abs(r["trades_net"][0] - expected_net) < 1e-12
    # And the closed trade's net must be strictly below its gross (costs charged).
    assert r["trades_net"][0] < r["trades_gross"][0]


def test_final_open_trade_is_not_liquidated(monkeypatch):
    """Spec §3 defines no exit event, so a run open at the last bar pays no exit
    cost; it accrues its per-bar returns and only the entry cost it incurred.

    lookback 2; closes [100,101,102,103,104] => w = [0,0,1,1,1] (open at end).
      net = log(103/102) - 28bps (entry) + log(104/103)
          = log(104/102) - 0.0028
    """
    monkeypatch.setattr("scripts.momentum72_study.MOM_LOOKBACK", 2)
    monkeypatch.setattr("scripts.momentum72_study.VOL_LOOKBACK", 2)
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    assert weights(closes)[-1] > 0.0
    expected = math.log(104.0 / 102.0) - COST_ONE_WAY / 1e4
    r = run(closes)
    assert r["n_trades"] == 1
    assert abs(r["trades_net"][0] - expected) < 1e-12
    # Regression guard: the old harness recorded this as 0.0.
    assert abs(r["trades_net"][0]) > 1e-6


def test_information_coefficient_uses_the_primary_convention(monkeypatch):
    """IC pairs w_i with logret_{i+1} — the same timing as run(delay=0)."""
    monkeypatch.setattr("scripts.momentum72_study.MOM_LOOKBACK", 2)
    monkeypatch.setattr("scripts.momentum72_study.VOL_LOOKBACK", 2)
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
    w = weights(closes)
    lr = log_returns(closes)
    xs, ys = w[:-1], lr
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=False))
    expected = num / ((sum((a - mx) ** 2 for a in xs) ** 0.5)
                      * (sum((b - my) ** 2 for b in ys) ** 0.5))
    assert abs(information_coefficient(closes) - expected) < 1e-12
