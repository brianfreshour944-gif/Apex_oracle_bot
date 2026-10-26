"""Ledger-based position-outcome reconstruction (2026-10-25 accounting audit).

The deployed database held two impossible return_pct values (ETH -804.76%,
SOL -70218.31%) from dust partial exits: the close path used the FINAL exit
leg's quantity as if it were the whole position, then subtracted the FULL
entry fee, so a ~$0.04 gross was charged a whole-position fee. These tests pin
the corrected contract:

  - realized P&L is reconstructed from EVERY fill linked to the decision_id,
  - the stored quantity reconciliation is the base-asset entry fee, so a
    fee-sized residual passes and a real imbalance fails closed,
  - an outcome that cannot be assembled from the ledger is labelled
    "unreconciled" instead of carrying a guessed number,
  - all three close paths (standard, flatten, reconcile) share one definition.
"""


import pytest

import src.bot as bot
import src.db as db
from src.config import settings


@pytest.fixture
def fresh_db(tmp_path):
    original = settings.DATABASE_URL
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    db._open_snapshot_cache.clear()
    settings.DATABASE_URL = "sqlite:///" + str(tmp_path / "bot.db").replace("\\", "/")
    db.init_db()
    yield
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    db._open_snapshot_cache.clear()
    settings.DATABASE_URL = original


def _snapshot(decision_id="dec-1", symbol="ETH/USD", action="buy",
              entry_price=100.0, qty=1.0, realized_pnl=0.0, return_pct=0.0):
    db.save_decision_snapshot(
        decision_id=decision_id, symbol=symbol, regime="trending",
        final_action=action, confidence=0.7, size_multiplier=1.0,
        entry_price=entry_price, qty=qty, brain_votes={"transformer": action},
    )
    if realized_pnl or return_pct:
        db.update_decision_snapshot_position(decision_id, entry_price=entry_price, qty=qty)
    return decision_id


def _closed(decision_id):
    from sqlalchemy import select

    from src.db import DecisionSnapshot, get_db_session
    with get_db_session() as session:
        return session.execute(
            select(DecisionSnapshot).where(DecisionSnapshot.decision_id == decision_id)
        ).scalar_one()


def _buy(order_id, decision_id, symbol, qty, price, commission=0.0):
    db.save_order_record(
        order_id=order_id, decision_id=decision_id, symbol=symbol, side="buy",
        qty=qty, filled_qty=qty, filled_avg_price=price, commission=commission,
        status="filled", type="market",
    )


def _sell(order_id, decision_id, symbol, qty, price, commission=0.0):
    db.save_order_record(
        order_id=order_id, decision_id=decision_id, symbol=symbol, side="sell",
        qty=qty, filled_qty=qty, filled_avg_price=price, commission=commission,
        status="filled", type="market",
    )


def test_reconstruct_sums_all_linked_fills(fresh_db):
    """A scale-in buy and a scale-out sell must both be summed, not just one leg."""
    _snapshot("dec-sums")
    _buy("b1", "dec-sums", "ETH/USD", 0.5, 100.0, commission=0.1)
    _buy("b2", "dec-sums", "ETH/USD", 0.5, 110.0, commission=0.1)
    _sell("s1", "dec-sums", "ETH/USD", 0.6, 120.0, commission=0.2)
    _sell("s2", "dec-sums", "ETH/USD", 0.4, 130.0, commission=0.2)

    out = db.reconstruct_position_outcome("dec-sums")
    assert out["outcome_status"] == "reconciled"
    basis = 0.5 * 100.0 + 0.5 * 110.0            # 105
    proceeds = 0.6 * 120.0 + 0.4 * 130.0         # 124
    fees = 0.1 + 0.1 + 0.2 + 0.2                 # 0.6
    assert out["realized_pnl"] == pytest.approx(proceeds - basis - fees)  # 18.4
    assert out["return_pct"] == pytest.approx((proceeds - basis - fees) / basis * 100.0)


def test_reconstruct_fee_sized_residual_passes_as_reconciled(fresh_db):
    """The observed base-asset entry fee (25 bps of qty) must not fail the
    reconciliation: buy 1.0, sell 0.9975 is a fee, not a mismatch."""
    _snapshot("dec-fee")
    _buy("b1", "dec-fee", "ETH/USD", 1.0, 100.0, commission=0.25)
    _sell("s1", "dec-fee", "ETH/USD", 0.9975, 101.0, commission=0.25)
    out = db.reconstruct_position_outcome("dec-fee")
    assert out["outcome_status"] == "reconciled"
    assert out["residual_qty"] == pytest.approx(0.0025)


def test_entry_fee_withheld_in_base_is_not_double_counted(fresh_db):
    """The deployed convention: the entry fee is withheld in the BASE asset, so
    sell_qty == buy_qty*(1 - 0.0025). That fee is already inside the reduced
    proceeds and must NOT be subtracted again. This is the bug the 2026-10-25
    review caught: an earlier revision subtracted the entry fee a second time.

    buy 1.0 @ 100 (comm 0.25 quote), sell 0.9975 @ 101 (comm 0.25 quote).
    Net cash flow = 0.9975*101 - 1.0*100 - 0.25(exit) = 0.4975.
    Subtracting the entry fee again would wrongly give 0.2475."""
    _snapshot("dec-base", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-base", "ETH/USD", 1.0, 100.0, commission=0.25)
    _sell("s1", "dec-base", "ETH/USD", 0.9975, 101.0, commission=0.25)
    out = db.reconstruct_position_outcome("dec-base")
    assert out["outcome_status"] == "reconciled"
    assert out["realized_pnl"] == pytest.approx(0.4975)
    assert out["entry_fees"] == pytest.approx(0.25)
    assert out["exit_fees"] == pytest.approx(0.25)


def test_entry_fee_charged_in_quote_is_subtracted_in_full(fresh_db):
    """The other clean convention: quantities match exactly (no withholding), so
    the entry fee is a separate quote charge and IS subtracted in full."""
    _snapshot("dec-quote", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-quote", "ETH/USD", 1.0, 100.0, commission=0.25)
    _sell("s1", "dec-quote", "ETH/USD", 1.0, 110.0, commission=0.30)
    out = db.reconstruct_position_outcome("dec-quote")
    assert out["outcome_status"] == "reconciled"
    # 110 - 100 - 0.30(exit) - 0.25(entry, charged in quote) = 9.45
    assert out["realized_pnl"] == pytest.approx(9.45)


def test_unexplained_residual_fails_closed(fresh_db):
    """A residual that matches NEITHER fee convention (not 0, not the entry fee)
    is not a fee at all -- the cost is unknown, so it must fail closed rather
    than guess a partial fee."""
    _snapshot("dec-unexpl", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-unexpl", "ETH/USD", 1.0, 100.0, commission=0.25)
    # residual 0.0015 = 0.15% of notional: neither 0 nor the 0.25% entry fee.
    _sell("s1", "dec-unexpl", "ETH/USD", 0.9985, 101.0, commission=0.25)
    out = db.reconstruct_position_outcome("dec-unexpl")
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "unexplained_residual"
    assert out["realized_pnl"] is None


# --- Fee-inference matrix (2026-10-25 review, requirement 1) -----------------
# The convention must be *inferred per row* from the recorded quantities, not
# assumed globally. These pin the discriminating evidence, not just arithmetic.


def test_fee_inference_zero_fee_matches_quantities(fresh_db):
    """Exact quantities with ZERO recorded commission on both legs. A recorded
    0 means 'no fee recorded', not 'no fee charged' (Alpaca reports 0.0 on every
    fill; the real fee is estimated, never observed). The schema cannot tell a
    verified zero from an absent fee, so this must NOT be certified as a
    fee-free P&L -- that is the false-certification the 2026-10-25 review
    raised. The row is unreconciled and carries no number."""
    _snapshot("dec-zerofee", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-zerofee", "ETH/USD", 1.0, 100.0, commission=0.0)
    _sell("s1", "dec-zerofee", "ETH/USD", 1.0, 110.0, commission=0.0)
    out = db.reconstruct_position_outcome("dec-zerofee")
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "unverified_fee_provenance"
    assert out["realized_pnl"] is None


def test_fee_inference_missing_fee_data_fails_closed(fresh_db):
    """No commission recorded on either leg and the quantities differ: we cannot
    tell whether the gap is a fee or an unexplained imbalance, and with no fee
    recorded the cost is unverified either way. Fails closed -- now on the fee
    provenance gate, which fires before the residual classifier."""
    _snapshot("dec-nofee", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-nofee", "ETH/USD", 1.0, 100.0, commission=0.0)
    _sell("s1", "dec-nofee", "ETH/USD", 0.9975, 110.0, commission=0.0)
    out = db.reconstruct_position_outcome("dec-nofee")
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "unverified_fee_provenance"
    assert out["realized_pnl"] is None


def test_zero_commissions_with_exact_quantities_do_not_certify_unknown_fees(fresh_db):
    """Reviewer-requested (2026-10-25, primary blocker). Exact buy/sell
    quantities with commission=0 on BOTH legs. The residual is 0, so the old
    classifier took the quote-charged branch with fee_rate 0 and certified a
    fee-free P&L. That is wrong: OrderRecord.commission is a plain float with no
    'verified' flag, and Alpaca reports 0.0 on every fill, so 0 means 'no fee
    recorded', not 'no fee charged'. The model cannot distinguish a verified
    zero from an absent fee, therefore this must be unreconciled with no number
    -- never a certified fee-inclusive P&L built on unverified costs."""
    _snapshot("dec-zerocert", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-zerocert", "ETH/USD", 1.0, 100.0, commission=0.0)
    _sell("s1", "dec-zerocert", "ETH/USD", 1.0, 110.0, commission=0.0)
    out = db.reconstruct_position_outcome("dec-zerocert")
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "unverified_fee_provenance"
    assert out["realized_pnl"] is None
    assert out["return_pct"] is None
    # A nonzero fee on both legs is certifiable -- the gate is about missing
    # evidence, not about refusing to ever certify.
    _snapshot("dec-zerocert2", entry_price=100.0, qty=1.0)
    _buy("b2", "dec-zerocert2", "ETH/USD", 1.0, 100.0, commission=0.25)
    _sell("s2", "dec-zerocert2", "ETH/USD", 1.0, 110.0, commission=0.30)
    ok = db.reconstruct_position_outcome("dec-zerocert2")
    assert ok["outcome_status"] == "reconciled"
    assert ok["realized_pnl"] == pytest.approx(9.45)   # 10 - 0.25 - 0.30


def test_partial_exit_then_reentry_is_not_silently_merged(fresh_db):
    """Reviewer-requested (2026-10-25, blocker 2). Buy, sell MOST but not all,
    buy again, sell the rest -- with the leftover between the fee-sized residual
    and the 0.5% tolerance, and the aggregate quantities balancing.

    The counter closes a cycle once cumulative sells reach (1 - 0.005) of
    cumulative buys. That threshold is BELOW true flat (1.0), so it closes MORE
    eagerly than reality. Here the first sell (0.996 = 99.6%) already closes the
    cycle, so the re-entry starts a second one and the row fails closed as
    multiple lifecycles -- the safe outcome.

    SCOPE OF THE GUARANTEE (narrowed 2026-10-25 review): this shows the counter
    does not UNDER-split when the interior residual is at or below the
    tolerance. It does NOT establish that it never merges -- a residual ABOVE
    the tolerance leaves the cycle open and can merge two round trips (see
    test_interior_residual_above_tolerance_can_merge_and_certify). The
    enumeration behind this test covers only the sequences and quantities it
    states, under the fee convention it assumes; it is evidence, not proof of
    universal correctness."""
    _snapshot("dec-reentry", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-reentry", "ETH/USD", 1.0, 100.0, commission=0.25)
    _sell("s1", "dec-reentry", "ETH/USD", 0.996, 101.0, commission=0.25)
    _buy("b2", "dec-reentry", "ETH/USD", 1.0, 102.0, commission=0.25)
    _sell("s2", "dec-reentry", "ETH/USD", 0.996, 103.0, commission=0.25)
    out = db.reconstruct_position_outcome("dec-reentry")
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "multiple_position_lifecycles"
    assert out["lifecycles"] == 2
    assert out["realized_pnl"] is None


def test_interior_residual_above_tolerance_is_failed_closed(fresh_db):
    """REGRESSION CASE 1 (2026-10-25 review, approved fix): buy -> partial sell
    -> buy -> sell must be classified unreconciled when the residual stays above
    the tolerance.

    Before the fix the counter folded the re-entry into the still-open cycle and,
    because the aggregate returned to flat, the row was CERTIFIED reconciled as a
    single lifecycle -- a false certification. The ambiguous-history guard now
    fails it closed. Fees are the DEPLOYED 25 bps (< tol); the 0.6% interior
    residual comes from a PARTIAL EXIT, so this was reachable on the deployed
    configuration, not only at a >0.5% fee."""
    _snapshot("dec-merge", entry_price=100.0, qty=1.0)
    # Round trip 1: a PARTIAL exit leaves 0.6% residual (> 0.5% tol) -> cycle OPEN.
    _buy("b1", "dec-merge", "ETH/USD", 1.0, 100.0, commission=0.25)
    _sell("s1", "dec-merge", "ETH/USD", 0.994, 101.0, commission=0.25)
    # Round trip 2 re-buys and over-sells, returning the AGGREGATE to flat.
    _buy("b2", "dec-merge", "ETH/USD", 1.0, 100.0, commission=0.25)
    _sell("s2", "dec-merge", "ETH/USD", 1.006, 101.0, commission=0.25)
    out = db.reconstruct_position_outcome("dec-merge")
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "ambiguous_lifecycle_history"
    assert out["realized_pnl"] is None
    assert out["return_pct"] is None


def test_legitimate_scale_in_stays_eligible_for_reconciliation(fresh_db):
    """REGRESSION CASE 2 (2026-10-25 review): buy -> buy -> sell is a normal
    scale-in (ONE position) and must NOT be flagged ambiguous. The guard only
    fires on a buy that follows a SELL with a residual still above tolerance, so
    a pyramid entry is unaffected and remains reconcilable."""
    _snapshot("dec-scalein", entry_price=100.0, qty=1.5)
    _buy("b1", "dec-scalein", "ETH/USD", 1.0, 100.0, commission=0.25)
    _buy("b2", "dec-scalein", "ETH/USD", 0.5, 102.0, commission=0.1275)
    _sell("s1", "dec-scalein", "ETH/USD", 1.49625, 105.0, commission=0.3926)
    out = db.reconstruct_position_outcome("dec-scalein")
    assert out["outcome_status"] == "reconciled"
    assert out["reason"] is None
    assert out["realized_pnl"] is not None


def test_single_position_closed_in_multiple_sell_legs_is_not_ambiguous(fresh_db):
    """REGRESSION CASE 3 (2026-10-25 review): buy -> partial sell -> sell (a
    single position closed across multiple sell legs, NO intervening buy) must
    not be marked ambiguous. The guard requires a buy after the partial sell."""
    _snapshot("dec-scalingout", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-scalingout", "ETH/USD", 1.0, 100.0, commission=0.25)
    _sell("s1", "dec-scalingout", "ETH/USD", 0.5, 101.0, commission=0.126)
    _sell("s2", "dec-scalingout", "ETH/USD", 0.4975, 102.0, commission=0.1269)
    out = db.reconstruct_position_outcome("dec-scalingout")
    assert out["outcome_status"] == "reconciled"
    assert out["reason"] is None


def test_multiple_completed_lifecycles_remain_unreconciled(fresh_db):
    """REGRESSION CASE 4 (2026-10-25 review): two COMPLETED lifecycles under one
    decision_id stay unreconciled under the existing multiple-lifecycle rule
    (the ambiguous-history guard is an ADDITION, not a replacement)."""
    _snapshot("dec-two", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-two", "ETH/USD", 1.0, 100.0, commission=0.25)
    _sell("s1", "dec-two", "ETH/USD", 0.9975, 101.0, commission=0.25)
    _buy("b2", "dec-two", "ETH/USD", 1.0, 105.0, commission=0.26)
    _sell("s2", "dec-two", "ETH/USD", 0.9975, 106.0, commission=0.26)
    out = db.reconstruct_position_outcome("dec-two")
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "multiple_position_lifecycles"
    assert out["lifecycles"] == 2


def test_buy_after_genuinely_flat_cycle_uses_multiple_lifecycle_rule(fresh_db):
    """REGRESSION CASE 5 (2026-10-25 review): when the previous cycle is
    genuinely flat (sold >= bought*(1-tol)) the counter CLOSES it, so a later buy
    starts a SECOND lifecycle -- reported as multiple_position_lifecycles, NOT as
    ambiguous. The ambiguous reason is reserved for a still-open residual."""
    _snapshot("dec-flat", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-flat", "ETH/USD", 1.0, 100.0, commission=0.25)
    _sell("s1", "dec-flat", "ETH/USD", 0.996, 101.0, commission=0.25)  # flat within tol
    _buy("b2", "dec-flat", "ETH/USD", 1.0, 102.0, commission=0.25)
    _sell("s2", "dec-flat", "ETH/USD", 0.996, 103.0, commission=0.25)
    out = db.reconstruct_position_outcome("dec-flat")
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "multiple_position_lifecycles"
    assert out["lifecycles"] == 2


def test_fee_estimation_disabled_makes_ledger_rows_uncertifiable(fresh_db, monkeypatch):
    """Explicit operating condition (2026-10-25 review). With
    ESTIMATED_TAKER_FEE_BPS=0 -- which config documents as "disables fee
    estimation" -- no live fill can carry a nonzero commission, so the
    fee-provenance gate marks EVERY ledger-assembled round trip unreconciled.
    This is deliberate and conservative: with no fee evidence at all, the
    system must not certify a fee-inclusive P&L. Operators who disable
    estimation should expect all reconstructed P&L to read unreconciled."""
    monkeypatch.setattr(settings, "ESTIMATED_TAKER_FEE_BPS", 0.0)
    from src.exchange import AlpacaExchange
    # The adapter produces commission 0.0 when estimation is disabled...
    info = AlpacaExchange._apply_estimated_commission(
        {"status": "filled", "filled_avg_price": 100.0, "filled_qty": 1.0,
         "qty": 1.0, "commission": 0.0}
    )
    assert info["commission"] == 0.0
    # ...so the persisted row has no fee evidence and cannot be certified.
    _snapshot("dec-norate", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-norate", "ETH/USD", 1.0, 100.0, commission=0.0)
    _sell("s1", "dec-norate", "ETH/USD", 1.0, 110.0, commission=0.0)
    out = db.reconstruct_position_outcome("dec-norate")
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "unverified_fee_provenance"
    assert out["realized_pnl"] is None


def test_fee_inference_ambiguous_between_conventions_fails_closed(fresh_db):
    """A residual equal to half the recorded entry fee sits between 'base-
    withheld' and 'quote-charged' and must not be silently assigned either."""
    _snapshot("dec-ambig", entry_price=100.0, qty=1.0)
    # entry fee rate 0.5% of notional; residual only 0.25% -> ambiguous.
    _buy("b1", "dec-ambig", "ETH/USD", 1.0, 100.0, commission=0.50)
    _sell("s1", "dec-ambig", "ETH/USD", 0.9975, 110.0, commission=0.50)
    out = db.reconstruct_position_outcome("dec-ambig")
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "unexplained_residual"


def test_fee_inference_is_scale_free(fresh_db):
    """The base-vs-quote decision must not depend on position size: a tiny
    position and a large one with the same fee rate take the same branch."""
    _snapshot("dec-small", symbol="ETH/USD", entry_price=100.0, qty=0.001)
    _buy("b1", "dec-small", "ETH/USD", 0.001, 100.0, commission=0.00025)
    _sell("s1", "dec-small", "ETH/USD", 0.0009975, 101.0, commission=0.00025)
    small = db.reconstruct_position_outcome("dec-small")

    _snapshot("dec-large", symbol="SOL/USD", entry_price=100.0, qty=1000.0)
    _buy("b2", "dec-large", "SOL/USD", 1000.0, 100.0, commission=250.0)
    _sell("s2", "dec-large", "SOL/USD", 997.5, 101.0, commission=250.0)
    large = db.reconstruct_position_outcome("dec-large")

    assert small["outcome_status"] == large["outcome_status"] == "reconciled"
    # Same relative move -> same return_pct regardless of notional.
    assert small["return_pct"] == pytest.approx(large["return_pct"])


# --- Multiple position lifecycles (2026-10-25 review, requirement 2) ---------


def test_reused_decision_id_with_two_round_trips_is_unreconciled(fresh_db):
    """The real c6a6d0423cc2 shape: one decision_id holding two separate
    buy->sell lifecycles must NOT be summed into a single outcome."""
    _snapshot("dec-two", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-two", "ETH/USD", 1.0, 100.0, commission=0.25)
    _sell("s1", "dec-two", "ETH/USD", 0.9975, 101.0, commission=0.25)
    _buy("b2", "dec-two", "ETH/USD", 1.0, 105.0, commission=0.26)
    _sell("s2", "dec-two", "ETH/USD", 0.9975, 106.0, commission=0.26)
    out = db.reconstruct_position_outcome("dec-two")
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "multiple_position_lifecycles"
    assert out["lifecycles"] == 2
    assert out["realized_pnl"] is None


def test_single_lifecycle_scale_in_is_still_reconciled(fresh_db):
    """A scale-in (buy, buy, then scale out) is ONE lifecycle and must stay
    reconciled -- the guard must not over-trigger on ordinary pyramid entries."""
    _snapshot("dec-pyramid", entry_price=100.0, qty=1.5)
    _buy("b1", "dec-pyramid", "ETH/USD", 1.0, 100.0, commission=0.25)
    _buy("b2", "dec-pyramid", "ETH/USD", 0.5, 102.0, commission=0.1275)
    _sell("s1", "dec-pyramid", "ETH/USD", 1.49625, 105.0, commission=0.3926)
    out = db.reconstruct_position_outcome("dec-pyramid")
    assert out["outcome_status"] == "reconciled"
    assert out["realized_pnl"] is not None


def test_lifecycle_order_uses_fill_time_not_submit_time(fresh_db):
    """Lifecycles must be counted in EXECUTION order. Here the submit order is
    b,b,s,s (one scale-in position) but the fills land b,s,b,s (two round
    trips). Counting by submitted_at would merge them into ONE lifecycle and
    certify a bogus P&L; counting by filled_at must fail closed."""
    import datetime as _dt
    t = _dt.datetime(2026, 10, 1, tzinfo=_dt.UTC)
    _snapshot("dec-ordering", entry_price=100.0, qty=1.0)

    def _leg(oid, side, qty, price, comm, sub_s, fill_s):
        db.save_order_record(order_id=oid, decision_id="dec-ordering", symbol="ETH/USD",
                             side=side, qty=qty, filled_qty=qty, filled_avg_price=price,
                             commission=comm, status="filled", type="market",
                             submitted_at=t + _dt.timedelta(seconds=sub_s),
                             filled_at=t + _dt.timedelta(seconds=fill_s))

    # submitted_at order: b1(1) b2(2) s1(3) s2(4)  -> looks like one position
    # filled_at    order: b1(1) s1(2) b2(3) s2(4)  -> two round trips
    _leg("b1", "buy", 1.0, 100.0, 0.25, sub_s=1, fill_s=1)
    _leg("b2", "buy", 1.0, 105.0, 0.26, sub_s=2, fill_s=3)
    _leg("s1", "sell", 0.9975, 101.0, 0.25, sub_s=3, fill_s=2)
    _leg("s2", "sell", 0.9975, 106.0, 0.26, sub_s=4, fill_s=4)

    out = db.reconstruct_position_outcome("dec-ordering")
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "multiple_position_lifecycles"
    assert out["lifecycles"] == 2
    assert out["realized_pnl"] is None


def test_reconstruct_real_imbalance_fails_closed(fresh_db):
    """A 3.93x quantity imbalance (the real outlier 1fd274986506) must NOT be
    reported as a reconciled P&L."""
    _snapshot("dec-bad")
    _buy("b1", "dec-bad", "SOL/USD", 0.151743, 200.0)
    _sell("s1", "dec-bad", "SOL/USD", 0.747381, 190.0)
    out = db.reconstruct_position_outcome("dec-bad")
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "quantity_mismatch"
    assert out["realized_pnl"] is None


def test_reconstruct_no_linked_exit_is_unreconciled(fresh_db):
    """The 11 BTC rows have decision_id=NULL on their exit sells, so no sell
    links to them: the outcome is uncertified, never a partial guess."""
    _snapshot("dec-noexit")
    _buy("b1", "dec-noexit", "BTC/USD", 0.001, 80000.0)
    out = db.reconstruct_position_outcome("dec-noexit")
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "no_linked_exit_fills"
    assert out["realized_pnl"] is None


def test_reconstruct_caller_fill_substitution_is_unreconciled(fresh_db):
    """When the ledger has buys but no sell (fire-and-forget write lost), the
    caller's observed exit is used to compute a value, but the row is labelled
    unreconciled so it is excluded from certified statistics."""
    _snapshot("dec-sub")
    _buy("b1", "dec-sub", "ETH/USD", 1.0, 100.0, commission=0.0)
    out = db.reconstruct_position_outcome(
        "dec-sub", exit_price=105.0, exit_qty=1.0, exit_commission=0.0,
    )
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "leg_substituted_from_caller"
    assert out["realized_pnl"] == pytest.approx(5.0)


def test_ledger_sell_commission_is_used_when_present(fresh_db):
    """When the closing sell IS on the ledger, its recorded commission is the
    fee -- the caller-supplied exit_commission must be ignored, so the fee is
    counted exactly once and never doubled."""
    _snapshot("dec-ledgerfee", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-ledgerfee", "ETH/USD", 1.0, 100.0, commission=0.25)
    _sell("s1", "dec-ledgerfee", "ETH/USD", 0.9975, 105.0, commission=0.30)
    out = db.reconstruct_position_outcome(
        "dec-ledgerfee",
        # A caller-supplied fee that must NOT be added on top of the ledger's.
        exit_price=105.0, exit_qty=0.9975, exit_commission=99.0,
    )
    assert out["outcome_status"] == "reconciled"
    assert out["exit_fees"] == pytest.approx(0.30)          # ledger's fee, not 99.0
    # 0.9975*105 - 1.0*100 - 0.30(exit) = 4.4375 ; entry fee withheld in base.
    assert out["realized_pnl"] == pytest.approx(4.4375)


def test_caller_supplied_exit_commission_is_counted_exactly_once(fresh_db):
    """No closing sell on the ledger, so the caller supplies qty, price AND
    commission. The fee must be subtracted exactly once, and the row must stay
    unreconciled -- a substituted exit is a fallback estimate, never certified
    accounting (this is the risk.py flatten-path shape after the rev-4 fix)."""
    _snapshot("dec-subfee", entry_price=100.0, qty=2.0)
    _buy("b1", "dec-subfee", "ETH/USD", 2.0, 100.0, commission=0.50)
    out = db.reconstruct_position_outcome(
        "dec-subfee", exit_price=90.0, exit_qty=2.0, exit_commission=0.45,
    )
    assert out["outcome_status"] == "unreconciled"
    assert out["reason"] == "leg_substituted_from_caller"
    # gross (90-100)*2 = -20.00, minus the caller's exit fee 0.45, minus the
    # ledger entry fee 0.50 (residual 0 -> quote-charged convention, so the
    # entry fee is a separate cash charge) = -20.95. Each fee exactly once.
    assert out["exit_fees"] == pytest.approx(0.45)
    assert out["entry_fees"] == pytest.approx(0.50)
    assert out["realized_pnl"] == pytest.approx(-20.95)


def test_reconstruct_no_fills_at_all(fresh_db):
    _snapshot("dec-empty")
    out = db.reconstruct_position_outcome("dec-empty")
    assert out["outcome_status"] == "unreconciled"
    assert out["realized_pnl"] is None


def test_partial_exit_no_longer_produces_absurd_return_pct(fresh_db):
    """The core defect: a dust final leg must not be charged the whole entry
    fee. Before the fix this produced the -804% / -70218% rows."""
    _snapshot("dec-dust", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-dust", "ETH/USD", 1.0, 100.0, commission=0.25)
    # Scale out: 0.9999 then a 0.0001 dust leg.
    _sell("s1", "dec-dust", "ETH/USD", 0.9999, 100.05, commission=0.25)
    _sell("s2", "dec-dust", "ETH/USD", 0.0001, 100.0, commission=0.0001)

    out = db.reconstruct_position_outcome("dec-dust")
    assert out["outcome_status"] == "reconciled"
    assert abs(out["return_pct"]) < 100.0  # a real ~-0.45% move, not -804%


@pytest.mark.asyncio
async def test_standard_close_records_outcome_status(fresh_db, monkeypatch):
    """The standard close path writes outcome_status and the reconciled P&L."""
    _snapshot("dec-live", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-live", "ETH/USD", 1.0, 100.0, commission=0.25)
    _sell("s1", "dec-live", "ETH/USD", 1.0, 110.0, commission=0.25)

    await bot._record_committee_outcome(
        "ETH/USD", 110.0, exit_reason="signal_close",
        entry_price=100.0, qty=1.0, commission=0.25,
    )
    snap = _closed("dec-live")
    assert snap.outcome_status == "reconciled"
    # (110-100)*1 - 0.25 - 0.25 = 9.5
    assert snap.realized_pnl == pytest.approx(9.5)
    assert snap.return_pct == pytest.approx(9.5)


@pytest.mark.asyncio
async def test_standard_close_unreconciled_when_exit_not_on_ledger(fresh_db, monkeypatch):
    """A close whose sell never made it to the ledger is still closed (so the
    cache stops serving 'open') but labelled unreconciled."""
    _snapshot("dec-u", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-u", "ETH/USD", 1.0, 100.0, commission=0.0)
    await bot._record_committee_outcome(
        "ETH/USD", 105.0, exit_reason="signal_close",
        entry_price=100.0, qty=1.0, commission=0.0,
    )
    snap = _closed("dec-u")
    assert snap.status == "closed"
    assert snap.outcome_status == "unreconciled"
    assert snap.realized_pnl == pytest.approx(5.0)  # from the caller's fill


@pytest.mark.asyncio
async def test_flatten_and_standard_close_agree(fresh_db, monkeypatch):
    """The flatten path must produce the same P&L as the standard close for an
    identical round trip."""
    _snapshot("dec-a", entry_price=100.0, qty=2.0)
    _buy("b1", "dec-a", "ETH/USD", 2.0, 100.0, commission=0.5)
    _sell("s1", "dec-a", "ETH/USD", 2.0, 90.0, commission=0.45)

    await bot._record_committee_outcome(
        "ETH/USD", 90.0, exit_reason="signal_close",
        entry_price=100.0, qty=2.0, commission=0.45,
    )
    standard = _closed("dec-a")

    _snapshot("dec-b", entry_price=100.0, qty=2.0)
    _buy("b2", "dec-b", "ETH/USD", 2.0, 100.0, commission=0.5)
    _sell("s2", "dec-b", "ETH/USD", 2.0, 90.0, commission=0.45)

    from src.risk import RiskManager
    rm = RiskManager.__new__(RiskManager)
    await rm._close_open_snapshot("ETH/USD", 90.0, 2.0, "exposure_reduction")
    flatten = _closed("dec-b")

    assert standard.realized_pnl == pytest.approx(flatten.realized_pnl)
    assert standard.outcome_status == flatten.outcome_status == "reconciled"
    # (90-100)*2 - 0.5 - 0.45 = -20.95
    assert standard.realized_pnl == pytest.approx(-20.95)


@pytest.mark.asyncio
async def test_flatten_path_passes_exit_commission_when_ledger_write_failed(fresh_db):
    """The asymmetry the 2026-10-25 review flagged: if the flatten path's
    fail-safe ledger write did NOT land, the exit leg is substituted from the
    caller -- and the caller's fee must still reach the reconstruction. Without
    the exit_commission kwarg the substituted exit would be silently fee-free
    and the flatten P&L would omit a cost the standard close path includes."""
    _snapshot("dec-nofee", entry_price=100.0, qty=2.0)
    _buy("b1", "dec-nofee", "ETH/USD", 2.0, 100.0, commission=0.50)
    # No sell row: the ledger write failed, so only the caller knows the exit.

    from src.risk import RiskManager
    rm = RiskManager.__new__(RiskManager)
    await rm._close_open_snapshot(
        "ETH/USD", 90.0, 2.0, "exposure_reduction", exit_commission=0.45,
    )
    snap = _closed("dec-nofee")
    assert snap.status == "closed"
    assert snap.outcome_status == "unreconciled"  # substituted exit, not certified
    # (90-100)*2 - 0.45(caller exit fee) - 0.50(entry) = -20.95; the fee is in.
    assert snap.realized_pnl == pytest.approx(-20.95)


def test_additive_migration_adds_outcome_status_to_legacy_db(tmp_path):
    """A DB created before outcome_status existed must gain the column
    idempotently, and existing rows must read back as NULL (unknown)."""
    import sqlite3

    legacy = tmp_path / "legacy.db"
    con = sqlite3.connect(str(legacy))
    con.execute(
        "CREATE TABLE decision_snapshots ("
        "decision_id VARCHAR(64) PRIMARY KEY, symbol VARCHAR(32), regime VARCHAR(48), "
        "final_action VARCHAR(16), confidence FLOAT, size_multiplier FLOAT, "
        "entry_price FLOAT, qty FLOAT, votes_json TEXT, feature_snapshot_json TEXT, "
        "causal_reasoning_json TEXT, tensor_state_json TEXT, status VARCHAR(16), "
        "exit_reason VARCHAR(48), max_favorable_pct FLOAT, max_adverse_pct FLOAT, "
        "realized_pnl FLOAT, return_pct FLOAT, holding_period_sec FLOAT, "
        "created_at DATETIME, closed_at DATETIME)"
    )
    con.execute(
        "INSERT INTO decision_snapshots (decision_id, symbol, status, realized_pnl, return_pct) "
        "VALUES ('old-1', 'BTC/USD', 'closed', -1.5, -2.0)"
    )
    con.commit()
    con.close()

    original = settings.DATABASE_URL
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    db._open_snapshot_cache.clear()
    settings.DATABASE_URL = "sqlite:///" + str(legacy).replace("\\", "/")
    try:
        db.init_db()
        db.init_db()  # idempotent: must not raise on the second run
        with db.get_db_session() as session:
            row = session.get(db.DecisionSnapshot, "old-1")
        assert row is not None
        assert row.outcome_status is None  # unknown, never fabricated as reconciled
    finally:
        if db._engine is not None:
            db._engine.dispose()
        db._engine = None
        db._tables_ensured = False
        db._open_snapshot_cache.clear()
        settings.DATABASE_URL = original


# ── MFE/MAE capture (report §3) ─────────────────────────────────────────────

class _FakeStrategy:
    def __init__(self):
        self._trailing_peaks = {}
        self._trailing_troughs = {}


class _FakeRiskManager:
    def __init__(self):
        import threading
        self.peak_prices = {}
        self._peak_prices_lock = threading.Lock()


@pytest.fixture
def excursion_state(monkeypatch):
    """Install minimal strategy/risk state so the recorder can read peaks."""
    strategy = _FakeStrategy()
    rm = _FakeRiskManager()
    monkeypatch.setattr(bot._state, "strategy", strategy)
    monkeypatch.setattr(bot._state, "risk_manager", rm)
    return strategy, rm


@pytest.mark.asyncio
async def test_mfe_mae_recorded_from_caller_supplied_excursions(fresh_db, excursion_state, monkeypatch):
    """The regression from report §3: the standard close path used to pop
    _trailing_peaks/_trailing_troughs BEFORE the recorder read them, so every
    trade recorded 0.0/0.0. The fix has the CALLER capture the values before it
    clears, and pass them in; the recorder must record them."""
    strategy, _rm = excursion_state
    _snapshot("dec-mfe", symbol="BTC/USD", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-mfe", "BTC/USD", 1.0, 100.0)
    _sell("s1", "dec-mfe", "BTC/USD", 1.0, 110.0)
    # Price ran to 115 (favorable) and dipped to 92 (adverse) during the hold.
    # The caller reads these, then clears (as the standard close path does).
    strategy._trailing_peaks["BTC/USD"] = 115.0
    strategy._trailing_troughs["BTC/USD"] = 92.0
    captured_peak = strategy._trailing_peaks.pop("BTC/USD", None)
    captured_trough = strategy._trailing_troughs.pop("BTC/USD", None)

    await bot._record_committee_outcome(
        "BTC/USD", 110.0, exit_reason="signal_close",
        entry_price=100.0, qty=1.0, commission=0.0,
        excursion_peak=captured_peak, excursion_trough=captured_trough,
    )
    snap = _closed("dec-mfe")
    assert snap.max_favorable_pct == pytest.approx(15.0)   # (115-100)/100*100
    assert snap.max_adverse_pct == pytest.approx(8.0)      # (100-92)/100*100


@pytest.mark.asyncio
async def test_recorder_clear_timing_matches_head(fresh_db, excursion_state, monkeypatch):
    """The recorder's own clear runs on its happy path only, exactly as at HEAD.
    The caller owns excursion-state clearing on every other path, so the recorder
    must NOT add or move clears (that would retime trailing-stop state)."""
    strategy, rm = excursion_state
    _snapshot("dec-timing", symbol="BTC/USD", entry_price=100.0, qty=1.0)
    _buy("b1", "dec-timing", "BTC/USD", 1.0, 100.0)
    _sell("s1", "dec-timing", "BTC/USD", 1.0, 110.0)
    strategy._trailing_peaks["BTC/USD"] = 115.0
    strategy._trailing_troughs["BTC/USD"] = 92.0
    rm.peak_prices["BTC/USD"] = 115.0

    await bot._record_committee_outcome(
        "BTC/USD", 110.0, exit_reason="signal_close",
        entry_price=100.0, qty=1.0, commission=0.0,
    )
    # Happy path: recorder clears, same as HEAD.
    assert "BTC/USD" not in strategy._trailing_peaks
    assert "BTC/USD" not in strategy._trailing_troughs
    assert "BTC/USD" not in rm.peak_prices


@pytest.mark.asyncio
async def test_recorder_early_return_does_not_clear(fresh_db, excursion_state, monkeypatch):
    """An invalid exit price returns early WITHOUT clearing excursion state --
    matching HEAD. The caller is responsible for the clear on that path, so the
    recorder must not touch trailing-stop state here."""
    strategy, _rm = excursion_state
    _snapshot("dec-bail", symbol="BTC/USD", entry_price=100.0, qty=1.0)
    strategy._trailing_peaks["BTC/USD"] = 115.0
    strategy._trailing_troughs["BTC/USD"] = 92.0

    await bot._record_committee_outcome(
        "BTC/USD", -1.0, exit_reason="signal_close",   # invalid -> early return
        entry_price=100.0, qty=1.0, commission=0.0,
    )
    assert strategy._trailing_peaks.get("BTC/USD") == 115.0
    assert strategy._trailing_troughs.get("BTC/USD") == 92.0

# ── 8: adding the column must not change reconstruction output ────────────

def test_fee_source_does_not_change_reconstruction_output(fresh_db):
    # Consistent with the verified convention: entry fee 0.25 on notional 100
    # = 0.0025 rate, withheld in BASE, so sell_qty = buy_qty * 0.9975.
    db.save_order_record(
        order_id="r-buy", decision_id="dec-rec", symbol="ETH/USD", side="buy",
        qty=1.0, filled_qty=1.0, filled_avg_price=100.0, commission=0.25,
        status="filled",
    )
    db.save_order_record(
        order_id="r-sell", decision_id="dec-rec", symbol="ETH/USD", side="sell",
        qty=0.9975, filled_qty=0.9975, filled_avg_price=101.0, commission=0.25,
        status="filled",
    )
    before = db.reconstruct_position_outcome("dec-rec")
    assert before["outcome_status"] == "reconciled"

    # Populate provenance on the existing rows; output must be byte-identical.
    from src.db import OrderRecord, get_db_session
    with get_db_session() as session:
        for oid, src in (("r-buy", "estimated"), ("r-sell", "estimated")):
            row = session.get(OrderRecord, oid)
            row.fee_source = src
        session.commit()

    after = db.reconstruct_position_outcome("dec-rec")
    assert after == before
