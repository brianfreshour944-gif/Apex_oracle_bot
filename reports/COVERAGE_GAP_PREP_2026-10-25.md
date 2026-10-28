# COVERAGE-GAP PREP (READ-ONLY) — 4 tests for the B↔loss-limit interaction

Date: 2026-10-25. **Read-only.** No edits, staging, commits, DB changes, push,
or deploy. `CAPTURE_SUBMISSION_QUOTE` stays **False**.

Purpose: specify the four missing tests from `RISK_PY_READONLY_REVIEW_2026-10-25.md`
before any implementation. Every claim below was verified by running the real
code against a temp SQLite DB.

---

## Where the tests go

`tests/test_position_outcome_reconstruction.py` (existing file, B's suite).
It already has the fixtures needed: `fresh_db`, `_snapshot`, `_buy`, `_sell`,
`_closed`, and it already imports `src.bot`, `src.db`, `settings`.

No new fixture is required. No production code changes. No DB repairs.

## Existing coverage (so we don't duplicate)

| Existing test | Covers |
| --- | --- |
| `test_flatten_and_standard_close_agree` | flatten happy path = standard close |
| `test_flatten_path_passes_exit_commission_when_ledger_write_failed` | substituted exit keeps fee |
| `test_reconstruct_real_imbalance_fails_closed` | `quantity_mismatch` **at function level** |
| 9 `reconstruct_*` tests | every fail-closed reason at function level |

The gaps are the **flatten-path wiring** and the **consumer**, not the function.

---

## Gap 1 — flatten else-branch (recon returns `None`)

**Claim to pin:** when the ledger is inconsistent, the flatten path still closes
the row and writes the snapshot fallback `0.0` (not a guess, not a crash).

**Verified trigger:** flatten path with a quantity-mismatched ledger.

```
_snapshot("dec-flat-none", entry_price=100.0, qty=1.0)
_buy("b1","dec-flat-none","ETH/USD", 1.0, 100.0, commission=0.5)
_sell("s1","dec-flat-none","ETH/USD", 0.3, 110.0, commission=0.1)   # 0.7 residual -> mismatch
await RiskManager.__new__(RiskManager)._close_open_snapshot("ETH/USD", 110.0, 0.3, "exposure_reduction")
snap = _closed("dec-flat-none")
```

**Assertions (empirically confirmed):**
```
assert snap.status == "closed"
assert snap.outcome_status == "unreconciled"
assert snap.realized_pnl == 0.0        # snapshot fallback, else-branch
```
Observed run: `status=closed, outcome_status=unreconciled, realized_pnl=0.0`.

## Gap 2 — `get_recent_realized_pnl` sums unreconciled rows

**Claim to pin:** the loss-limit consumer includes unreconciled closed rows —
the coupling that makes B non-observational.

**Verified:** a single `close_decision_snapshot(..., outcome_status="unreconciled",
realized_pnl=-7.5)` row makes `get_recent_realized_pnl(24.0)` return **-7.5**.

```
db.save_decision_snapshot(decision_id="u1", symbol="ETH/USD", ...)
db.close_decision_snapshot("u1", realized_pnl=-7.5, return_pct=-7.5,
                           exit_reason="x", outcome_status="unreconciled")
assert db.get_recent_realized_pnl(24.0) == pytest.approx(-7.5)
```

**Add a contrast row** to make intent explicit: a `reconciled` row is included
too, so the test documents "sums ALL closed rows regardless of label". Optional
second assertion with two rows (one reconciled, one not) and their sum.

**Note:** this is a *characterization* test — it pins current deliberate
behavior. If the filter is ever added, this test must be updated on purpose.

## Gap 3 — flatten close clears the open-snapshot cache when unreconciled

**Claim to pin:** after an unreconciled flatten close, `get_open_snapshot`
stops serving the stale `open` record.

**Verified:** cache is populated before, `None` after `close_decision_snapshot`
with `outcome_status="unreconciled"`.

```
db.save_decision_snapshot(decision_id="c1", symbol="SOL/USD", ...)
assert db.get_open_snapshot("SOL/USD") is not None      # cache warm
await RiskManager.__new__(RiskManager)._close_open_snapshot("SOL/USD", 90.0, 1.0, "exposure_reduction")
assert db.get_open_snapshot("SOL/USD") is None           # cache cleared
```

Asserting via the **flatten path** (not `close_decision_snapshot` directly)
is what makes this cover the wiring.

## Gap 4 — tolerance boundary on the flatten path

**Claim to pin:** the `_QTY_RECONCILE_TOLERANCE` (0.005) comparison behaves the
same through the flatten path as at function level.

**⚠ Float gotcha found during prep — the obvious test is WRONG.**
The comparison is strict `>`:
```
1.0 - 0.995 == 0.0050000000000000044  ->  > 0.005  ->  quantity_mismatch
```
So a "residual exactly 0.005 passes" test would **fail**, and the real boundary
is the binary-float value, not the decimal 0.005. **Do not** write a test that
asserts residual == 0.005 reconciles.

**Proposed safe design** (clear margin on each side, both verified):

| Case | buy | sell | residual frac | expected |
| --- | --- | --- | --- | --- |
| below tol | 1.0 | 0.9955 | 0.0045 | `reconciled` |
| above tol | 1.0 | 0.994 | 0.006 | `unreconciled` (`quantity_mismatch`) |

Both confirmed by running the real function. Assert through the **flatten path**
so it covers the wiring, with `snap.outcome_status` and (for the below case)
`snap.realized_pnl is not None`.

---

## Traps the tests must avoid (both verified)

1. **`save_decision_snapshot` FOLDS a second save for the same symbol** into the
   existing open row (`[DATA_INTEGRITY] ... folded entry ... no duplicate`).
   Two `_snapshot()` calls with the same symbol produce **one** row.
   → Each new test must use a **distinct symbol** (or close the row first),
   exactly as the existing tests do (`dec-a`/ETH, `dec-b`/ETH are in *separate*
   tests; within one test the flatten test reuses ETH only after the first is
   closed).

2. **Flatten + `RiskManager.__new__(RiskManager)`** is the established pattern
   (skips `__init__`); `_close_open_snapshot` touches no other state. Reuse it.

## Impact assessment

| Question | Answer |
| --- | --- |
| Changes production behavior? | **No** — test-only file |
| Requires DB repair/migration? | **No** — `fresh_db` builds a temp DB |
| New fixtures/deps? | **No** |
| Weakens existing assertions? | **No** — purely additive tests |
| Overlaps existing tests? | Only at function level; the new ones cover flatten wiring + consumer |

## Proposed implementation (for separate authorization)

- Append 4 tests to `tests/test_position_outcome_reconstruction.py`.
- `@pytest.mark.asyncio` for gaps 1 and 3 (flatten is async); plain for 2 and 4
  unless the flatten path is used (then async).
- No `monkeypatch` needed except where a test wants to observe a warning.
- Expected suite delta: **+4 tests** (600 → 604 in the main worktree).

## Boundary

This prep authorizes **nothing**. Implementation is a separate authorization.
`CAPTURE_SUBMISSION_QUOTE` stays **False**. No push, no deployment. The four
tests improve confidence; they do **not** prove deploy-safety or trading edge.
