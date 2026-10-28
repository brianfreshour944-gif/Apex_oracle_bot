# READ-ONLY REVIEW — `src/risk.py` accounting change (Commit B)

Date: 2026-10-25. **Read-only.** No edits, staging, commits, DB changes, push,
or deploy. `CAPTURE_SUBMISSION_QUOTE` remains **False**. Review ≠ deploy approval.

Scope: the `_close_open_snapshot` change in commit `ff4ac11` (B) and its
interaction with `src/db.reconstruct_position_outcome`, the flatten/killswitch
callers, and downstream P&L consumers.

---

## Q1. `reconstruct_position_outcome()` failure behavior

**Fail-closed and total.** Every failure path returns
`{"realized_pnl": None, "return_pct": None, "outcome_status": "unreconciled", "reason": ...}`:

| Reason | Trigger |
| --- | --- |
| `no_linked_entry_fills` | no ledger buy and no entry fallback |
| `no_linked_exit_fills` | no ledger sell and no exit fallback |
| `zero_entry_qty` | buy qty sums to 0 |
| `quantity_mismatch` | residual/buy_qty > `_QTY_RECONCILE_TOLERANCE` (0.005) |
| `zero_position_basis` | basis = 0 |
| `unverified_fee_provenance` | fee convention not certifiable |
| `unexplained_residual` | residual fits neither fee convention |
| `multiple_position_lifecycles` / `ambiguous_lifecycle_history` | >1 round trip |
| `reconstruction_error` | internal exception (caught; **never raises**) |

Critically, **a partial ledger calculation is never returned as the P&L** — the
numeric fields are `None`, not a guess. Missing fees/quantities therefore cannot
produce a plausible-but-wrong number.

## Q2. Do flatten/killswitch paths still close safely?

**Yes — verified empirically** (temp DB, real `RiskManager._close_open_snapshot`):

| Case | Row status | outcome_status | realized_pnl |
| --- | --- | --- | --- |
| No ledger fills at all | **closed** | unreconciled | −10.0 (caller leg substituted) |
| Quantity mismatch | **closed** | unreconciled | 0.0 (snapshot fallback) |
| Happy round trip | **closed** | reconciled | −20.95 ✓ |
| No open snapshot | no crash | — | — |

In every case the row reaches `status='closed'` and `get_open_snapshot()`
returns `None` afterwards, so the cache stops serving a stale `open` record.
The accounting change does **not** alter *whether* a position is closed — the
exchange order is placed before `_close_open_snapshot` is called, and the
callers never branch on its result (it returns `None`).

One note on Case 1: with no ledger rows the caller's exit leg is substituted, so
`realized_pnl` is *written* (−10.0) but labelled `unreconciled`. That number is a
best-effort substitute, not certified. Same shape as HEAD's fallback.

## Q3. Do downstream controls consume reconstructed P&L without checking `outcome_status`?

**Yes — one does, deliberately, and it affects trading.**

`get_recent_realized_pnl()` (`db.py:1703`) sums **every** closed row regardless
of `outcome_status`. It feeds the rolling loss-limit entry block
(`bot.py:2184`): if the rolling sum ≤ limit, the bot **blocks a new entry**.

The docstring states this is intentional (fail-safe: under-counting losses to
avoid a block is the worse error). Other consumers:

- Adaptive-learner bootstrap (`bot.py:3613`) and OOD bootstrap (`bot.py:3445`)
  also read all closed rows; no `outcome_status` filter.
- `get_closed_decision_snapshots` docstring explicitly says consumers *should*
  exclude non-`reconciled` rows but **this is not enforced**.

**This is pre-existing, not introduced by B.** At HEAD (`6c46fb1`)
`get_recent_realized_pnl` already summed all closed rows and the rolling loss
limit already existed. B adds the `LIMITATION` docstring but does not change the
filtering. So B does not *introduce* the unfiltered consumption — but B does
change *what numbers* flow into it (see Q4).

## Q4. Can the change affect the decision to close, not just how it is recorded?

**Two distinct answers.**

**(a) The close decision itself: no.** On the flatten/killswitch paths the
exchange order is created *before* `_close_open_snapshot` runs; the callers never
branch on its result. So B cannot cause or prevent a close.

**(b) Downstream entry decisions: yes, indirectly.** Because
`get_recent_realized_pnl` sums all closed rows unfiltered, and B changes the
`realized_pnl` written on the flatten paths, B **changes the values that feed the
rolling loss-limit entry block**. Concretely, before B the flatten path wrote
P&L from a single leg (and `0.0` when inputs were missing); after B it writes a
ledger-reconstructed value or `0.0` on failure. A different magnitude can flip
whether the rolling sum crosses the limit, which changes whether a *future entry*
is blocked.

So: B is an accounting change, but it is **not purely observational** — it
propagates into a trading gate through the pre-existing unfiltered consumer.
This is the most important finding and it is a *documented, deliberate*
limitation, not a defect. Fixing it (filter on `outcome_status`) would itself be
a trading-behavior change and is out of scope for the accounting patch.

## Q5. Test coverage — what is covered and what is not

**Covered:**
- `test_flatten_and_standard_close_agree` — flatten vs standard close parity.
- `test_flatten_path_passes_exit_commission_when_ledger_write_failed` —
  substituted exit keeps its fee; asserts `unreconciled` + correct P&L.
- All nine `reconstruct_*` fail-closed reasons asserted at the function level
  (`realized_pnl is None`).
- Loss-limit entry block exercised in `test_integration_trading_loop.py` — but
  with `get_recent_realized_pnl` **mocked** (`return_value=-500.0`), so the real
  DB sum path is not exercised there.

**Not covered (gaps):**
1. **No end-to-end test of the flatten else-branch** — i.e. reconstruction
   returns `None` on the flatten path and the row is closed with the
   `snap.get("realized_pnl") or 0.0` fallback. (My Case 2 above demonstrates it
   works, but no committed test asserts it.)
2. **No test that `get_recent_realized_pnl` sums unreconciled rows** — the
   behavior that couples B to the entry block is asserted only in a docstring.
3. **No test that a flatten close still clears the open-snapshot cache** when
   reconstruction is unreconciled (Q2 Case 1 behavior).
4. No test pinning the `_QTY_RECONCILE_TOLERANCE` boundary on the flatten path.

None of these are regressions; they are untested guarantees.

---

## Summary

| Question | Answer |
| --- | --- |
| Q1 fail-closed? | Yes; `None`+`unreconciled`, never raises |
| Q2 safe close on failure? | Yes; row always closes, cache clears |
| Q3 unfiltered consumer? | Yes — `get_recent_realized_pnl` → loss-limit entry block (pre-existing) |
| Q4 affects close decision? | No direct; **yes indirectly** via the entry block |
| Q5 coverage? | Function-level good; flatten else-branch and the coupling untested |

**Assessment:** B is safe as an *accounting* change — it fails closed and cannot
break the close path. Its only behavioral reach is through the pre-existing
unfiltered `get_recent_realized_pnl` consumer, which is deliberate and
documented. The gaps are test-coverage gaps, not defects.

**Boundary:** this review authorizes nothing downstream. Executor isolation and
capture enablement remain separate, explicitly authorized steps.
`CAPTURE_SUBMISSION_QUOTE` stays **False**. No deployment.
