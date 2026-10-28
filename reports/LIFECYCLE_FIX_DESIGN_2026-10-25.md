# APPROVED + INCORPORATED — fail closed on ambiguous lifecycle histories

Status: **incorporated into the uncommitted patch (2026-10-25), after explicit
reviewer approval.** Still NOT staged, NOT committed, NOT deployed.
`src/db.py` is a protected path, so this records the design and the approval
that authorized the change. Regression coverage: 5 named cases (see below).

## Problem (confirmed reachable at the deployed fee)

`reconstruct_position_outcome` fails closed when `_count_position_lifecycles`
returns > 1. But that counter closes a cycle only once cumulative sells reach
`(1 - tol)` of cumulative buys (`tol = 0.005`). A history where a PARTIAL EXIT
leaves a residual ABOVE `tol`, followed by a re-entry buy, keeps the cycle open;
if the aggregate later returns to flat the quantity gate passes and the row is
CERTIFIED `reconciled` as a single lifecycle — a false certification.

Reproduction (deployed 25 bps fees, no >tol fee rate needed):

```
buy 1.0 @ 100, sell 0.994 @ 101   # partial exit -> residual 0.6% > tol -> cycle OPEN
buy 1.0 @ 100, sell 1.006 @ 101   # over-sell -> aggregate flat
# current: lifecycles=1, outcome_status="reconciled"  (WRONG)
```

Deployed snapshot: 0 of 42 decisions with filled orders exhibit this history, so
it is a code-path risk, not an observed one. It is nonetheless reachable.

## Proposed change (2 hunks, additive)

1. New helper `_has_ambiguous_reentry(ordered_legs)` — True when a buy arrives
   while the open cycle has already had a SELL and the residual is still > tol.
   It deliberately requires a prior SELL in the cycle, so a plain scale-in
   (`buy, buy, sell`) is NOT flagged.
2. Call site: `if lifecycles > 1 or _has_ambiguous_reentry(ordered):` -> fail
   closed with `reason = "ambiguous_lifecycle_history"` (distinct from
   `multiple_position_lifecycles`, so the two are separable in logs).

```diff
@@ def reconstruct_position_outcome(...) call site
             lifecycles = _count_position_lifecycles(ordered)
-            if lifecycles > 1:
+            if lifecycles > 1 or _has_ambiguous_reentry(ordered):
                 return {
                     "realized_pnl": None, "return_pct": None,
-                    "outcome_status": "unreconciled", "reason": "multiple_position_lifecycles",
+                    "outcome_status": "unreconciled",
+                    "reason": ("multiple_position_lifecycles" if lifecycles > 1
+                               else "ambiguous_lifecycle_history"),
                     "lifecycles": lifecycles,
                 }
```

Full helper source and diff: see the `src/db.py` hunk in
`reports/UNCOMMITTED_PATCH_2026-10-25.md` (`def _has_ambiguous_reentry` and the
`lifecycles > 1 or _has_ambiguous_reentry(ordered)` call site), which is the
available artifact. (An earlier draft cited `reports/PROPOSED_LIFECYCLE_FIX.diff`;
that file was never created — reference corrected 2026-10-25.)

## Validation performed (proposal applied temporarily, then reverted)

- Full suite: **542 passed** with the fix applied; ruff `All checks passed!`.
- Exhaustive check (length 3–7, quantities {0.25,0.5,0.75,1.0}, totals passing
  the quantity gate): the guard flags **every** merge case and misses **zero**
  (`merge-with-interior-reentry MISSED: 0`), with **no** scale-in false positive.
- Deployed snapshot: flags **0** of 42 decisions (no behavior change on real data).
- Regression test `test_interior_residual_above_tolerance_is_failed_closed`
  **FAILS against the current (rev 12) code** and passes with the fix — it
  discriminates.

## Scope / blast radius

- Touches ONLY `src/db.py` classification + one test. No schema change, no
  migration, no strategy logic, no risk-limit behavior, no exchange code.
- Rows it newly marks unreconciled are ones the current code certifies from an
  ambiguous history. It can only move `reconciled -> unreconciled` (fail closed);
  it never certifies anything new.
- Downstream: a newly-unreconciled row keeps its computed number but changes
  label. The rolling loss limit ignores the label (sums all closed rows), so its
  value is unchanged; the learner/label sites likewise consume the value.

## Regression coverage (reviewer-required, all present)

1. `test_interior_residual_above_tolerance_is_failed_closed` — buy/partial sell/
   buy/sell, residual > tol -> unreconciled. **Fails without the guard.**
2. `test_legitimate_scale_in_stays_eligible_for_reconciliation` — buy/buy/sell
   stays reconcilable.
3. `test_single_position_closed_in_multiple_sell_legs_is_not_ambiguous` —
   buy/partial sell/sell is not ambiguous.
4. `test_multiple_completed_lifecycles_remain_unreconciled` — two completed
   lifecycles still `multiple_position_lifecycles`.
5. `test_buy_after_genuinely_flat_cycle_uses_multiple_lifecycle_rule` — a buy
   after a genuinely-flat cycle is `multiple_position_lifecycles`, not ambiguous.

## Status

Approved and incorporated. Cases 2-5 pass with AND without the guard (the guard
does not reject legitimate scale-ins); case 1 fails without it. Patch remains
uncommitted, undeployed, live trading frozen.
