# COMMIT A — PREPARATION VALIDATION & FINAL SCOPE (READ-ONLY)

Date: 2026-10-25. **Read-only. No staging, commit, push, or deployment.**
Follows the authorized two-file default flip; validates A before staging.

Branch `w1-instrumentation-accounting`, HEAD `6c46fb1`, `main` `7a8ca12`,
**0 staged**.

---

## 1. Test rename — deferred into A (decision)

`test_capture_flag_defaults_on` → `test_capture_flag_defaults_off` is **not**
done yet. Per your fallback, the rename is **folded into the explicitly
authorized A work** rather than a separate cycle. The assertion is already
correct; only the name lags.

## 2. Validation results

| Check | Result |
| --- | --- |
| `tests/test_submission_quote_capture.py` + `tests/test_fee_provenance.py` | **22 passed** |
| Full suite | **600 passed**, 5 warnings, 26.3s |
| State files after suite | tracker `44136fa3…`, drift `82a78c0d…` — **unchanged** |

## 3. Migration validation — on a COPY (original untouched)

Copied `bot_snapshot.db` → `/tmp/mig_test.db` (sha `52e39180…`), ran `init_db()`
against the copy, then deleted the copy.

| Step | Result |
| --- | --- |
| `orders` before | 14 columns, **no** quote/fee columns |
| `orders` after run 1 | + `fee_source`, `mid_at_submit`, `spread_at_submit`, `quote_ts` |
| `orders` after run 2 | **identical** — idempotent |
| Original snapshot | sha `52e39180…` — **untouched** |

**CRITICAL FINDING:** run 1 also added `outcome_status` to `decision_snapshots`.
That column is **B's**, not A's. The current `_ensure_additive_columns()` dict
mixes scopes, so **committing A as-is would leak B's migration into A.** This is
exactly the split §5 resolves.

## 4. FINAL A/B/C HUNK MAP (per-hunk, precise)

### `src/exchange.py` — **all 6 hunks are A**
`@@ -11` (import `CryptoLatestQuoteRequest`), `@@ -193` (protocol),
`@@ -443` (`get_latest_quote`), `@@ -791` (pre-submit capture),
`@@ -803` (`submission_quote`), `@@ -840` (recovered-path `setdefault`).

### `src/db.py` — 11 hunks
| Hunk | Scope |
| --- | --- |
| `@@ -75,6 +75,28 @@` `DecisionSnapshot.outcome_status` | **B** |
| `@@ -155,6 +177,25 @@` `OrderRecord` fee+quote cols | **A** |
| `@@ -235,6 +276,10 @@` `save_order_record` kwargs | **A** |
| `@@ -255,6 +300,10 @@` `save_order_record` assigns | **A** |
| `@@ -335,6 +384,335 @@` `reconstruct_position_outcome` | **B** |
| `@@ -494,11 +872,50 @@` `_ensure_additive_columns` | **A+B — SPLIT (§5)** |
| `@@ -719,6 +1136,9 @@` `init_db` call | **A** |
| `@@ -1059,6 +1479,15 @@` `get_closed_decision_snapshots` | **B** |
| `@@ -1224,9 +1653,17 @@` `close_decision_snapshot` | **B+C** |
| `@@ -1254,6 +1691,7 @@` `close_decision_snapshot` | **B+C** |
| `@@ -1267,6 +1705,17 @@` `get_recent_realized_pnl` | **B** |

### `src/bot.py` — 13 hunks
| Hunk | Scope |
| --- | --- |
| `@@ -122`, `@@ -136`, `@@ -223` `_record_committee_outcome` | **C** |
| `@@ -165,51 +174,54 @@` | **B** |
| `@@ -245,6 +264,7 @@` | **B+C — SPLIT** |
| `@@ -273,6 +293,11 @@` (reconstructed-P&L NOTE) | **B** |
| `@@ -383,6 +408,10 @@` (reconstructed-P&L NOTE) | **B** |
| `@@ -984`, `@@ -1222` `reconcile_open_snapshots` | **B** |
| `@@ -1348,6 +1404,27 @@` `_persist_order_record` fee+quote | **A** |
| `@@ -1356,7 +1433,11 @@` pass kwargs | **A** |
| `@@ -2850,7 +2931,19 @@` peak/trough capture | **C** |
| `@@ -2864,6 +2957,8 @@` | **C** |

### `src/config.py` — 1 A hunk
`@@ -423,6 +424,14 @@` `CAPTURE_SUBMISSION_QUOTE` — now `default=False` (edit done).

## 5. THE SPLIT (proposed, no edits)

`_ensure_additive_columns()` is the only db.py hunk needing a hand split:

```python
# COMMIT A — contains ONLY A's columns:
additions = {
    "orders": [
        ("fee_source", "VARCHAR(16)"),
        ("mid_at_submit", "FLOAT"),
        ("spread_at_submit", "FLOAT"),
        ("quote_ts", "DATETIME"),
    ],
}
# COMMIT B — adds back B's column:
#     "decision_snapshots": [("outcome_status", "VARCHAR(16)")],
```

`fee_source` stays with **A** — verified: `reconstruct_position_outcome` (B core)
never reads it. Each commit's dict is valid and idempotent alone.

## 6. A's exact file set

```
src/config.py                                (1 hunk — done)
src/exchange.py                              (6 hunks)
src/db.py                                    (5 hunks; @@-494 split to A cols only)
src/bot.py                                   (2 hunks)
tests/test_submission_quote_capture.py       (untracked; + rename test)
tests/test_fee_provenance.py                 (untracked)
```

**Excluded:** `outcome_status` model + all B/C hunks; `test_position_outcome_reconstruction.py`.

## 7. Validation plan before staging A

1. A tests (22) + full suite (600) — **already pass**.
2. Ruff — `All checks passed!`.
3. State-file hashes unchanged — **already verified**.
4. Migration on a **copy**: A's dict adds exactly the 4 `orders` columns,
   idempotent, and adds **NO** `outcome_status`.
5. Contamination check on the **staged** A diff: no `outcome_status`,
   `reconstruct_position_outcome`, `excursion`, `max_fav`, `max_adv`.
6. Default check: `CAPTURE_SUBMISSION_QUOTE is False`.

## 8. Executor isolation — still required (unchanged)

Disabling capture by default **does not** resolve the shared API-key rate-limit
risk or worker starvation. Isolation remains necessary **before capture is
enabled in deployment**. A green suite does not substitute for it.

## Safety state

Branch `w1-instrumentation-accounting`; HEAD `6c46fb1`; `main` `7a8ca12`
untouched; **0 staged**; tracker/drift/snapshot/backup unchanged
(`44136fa3…` / `82a78c0d…` / `52e39180…` / `ddb2e987…`).
