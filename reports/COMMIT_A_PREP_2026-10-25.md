# COMMIT A — PREPARATION (READ-ONLY)

Date: 2026-10-25. **Read-only. No file modified, nothing staged, no commit.**
Deliverable for "Commit A preparation only". Reports the default decision and the
exact A-only scope, then stops.

Branch `w1-instrumentation-accounting`, HEAD `6c46fb1` (commit D), 1 ahead of
`main` (`7a8ca12`), 0 staged.

---

## 1. DEFAULT DECISION

**Recommendation: set `CAPTURE_SUBMISSION_QUOTE` default to `False` until
executor isolation is implemented.**

Rationale (evidence-backed from earlier phases):

- The capture runs on the **shared default executor** the order path uses. A
  hang can occupy a worker and delay `submit_order` (measured: 1.95s).
- A socket timeout is **not cleanly available** — `alpaca.common.rest._request`
  takes no `timeout`; it would need the private `_session`.
- Executor isolation (own pool) fixes the worker-starvation risk **structurally**
  but is a **separate, not-yet-implemented** change.
- Therefore, until isolation lands, the safe default is **`False`**: collect no
  quotes rather than couple measurement to the order path.

**Consequence for Commit A:** the config field must be added with
`default=False`, and the test that currently pins `True`
(`tests/test_submission_quote_capture.py:238` `test_capture_flag_defaults_on`)
must be **changed to assert `False`**. That is an edit to an existing A test —
in scope for A, but it is a *behaviour* assertion change and should be reviewed.

If instead you prefer `True`, then isolation must land **first** and A should be
sequenced after it. **This is the decision to make before A is committed.**

---

## 2. MIGRATION SPLIT (proposed, no edits made)

`src/db.py` `_ensure_additive_columns()` holds **one** dict mixing A and B:

```python
additions = {
    "decision_snapshots": [("outcome_status", "VARCHAR(16)")],      # B
    "orders": [
        ("fee_source", "VARCHAR(16)"),        # A (see note)
        ("mid_at_submit", "FLOAT"),           # A
        ("spread_at_submit", "FLOAT"),        # A
        ("quote_ts", "DATETIME"),             # A
    ],
}
```

**Proposed split:**

| Commit | `orders` columns | `decision_snapshots` |
| --- | --- | --- |
| **A** | `fee_source`, `mid_at_submit`, `spread_at_submit`, `quote_ts` | — |
| **B** | — | `outcome_status` |

**`fee_source` belongs to A, not B** — verified: `reconstruct_position_outcome`
(the B core, db.py:474–715) **never reads `fee_source`** (grep count 0). It is
written by `save_order_record` and read by tests/analysis, not by B's arithmetic.

**How to split without breaking either commit:** the dict literal cannot be
staged per-key by Git. Options:

- **(a)** In Commit A, the dict contains **only** the four `orders` columns; in
  Commit B, `outcome_status` is added to the dict. Each commit's dict is valid
  and idempotent on its own. **Recommended** — cleanest, each commit migrates
  exactly its own columns.
- **(b)** Land the whole dict in A. Rejected: it would add `outcome_status`
  before B's code uses it, mixing scopes.

**Note:** A's `OrderRecord` model must also add its four columns
(hunk `@@ -155,6 +177,25 @@`, which **also** contains the `fee_source` +
quote column definitions — A-only). B's `DecisionSnapshot.outcome_status`
(hunk `@@ -75,6 +75,28 @@`) is B-only.

---

## 3. EXACT A-ONLY SCOPE

### `src/config.py` — 1 hunk (A)
| Hunk | Content |
| --- | --- |
| `@@ -423,6 +424,14 @@` | `CAPTURE_SUBMISSION_QUOTE` field — **set `default=False` per §1** |

*(The `import os` hunk and the two `*_STATE_PATH` hunks are **already committed in D**.)*

### `src/exchange.py` — all hunks (A)
| Hunk | Content |
| --- | --- |
| `@@ -11,7 +11,7 @@` | import `CryptoLatestQuoteRequest` |
| `@@ -193,6 +193,7 @@` | `BaseExchange` protocol: `get_latest_quote` |
| `@@ -443,6 +444,46 @@` | `AlpacaExchange.get_latest_quote` implementation |
| `@@ -791,6 +832,18 @@` | pre-submit capture in `create_order` |
| `@@ -803,6 +856,7 @@` | `order_info["submission_quote"]` |
| `@@ -840,6 +894,10 @@` | `setdefault` on the recovered-order path |

**All A.** Confirmed additive: `commission_estimated` is **pre-existing at HEAD**
(3 refs) — A does not introduce it; A only *persists* its provenance.

### `src/db.py` — A hunks
| Hunk | Content | Scope |
| --- | --- | --- |
| `@@ -155,6 +177,25 @@` | `OrderRecord.fee_source` + 3 quote columns | **A** |
| `@@ -235,6 +276,10 @@` | `save_order_record` new kwargs | **A** |
| `@@ -255,6 +300,10 @@` | `save_order_record` assignments | **A** |
| `@@ -494,11 +872,50 @@` | `_ensure_additive_columns` — **A columns only** (§2) | **A** |
| `@@ -719,6 +1136,9 @@` | call `_ensure_additive_columns()` in `init_db` | **A** |

**B hunks in db.py (exclude):** `@@ -75,6 +75,28 @@` (`outcome_status` model),
`@@ -335,6 +384,335 @@` (`reconstruct_position_outcome` + helpers),
`@@ -1059,…` `get_closed_decision_snapshots`, `@@ -1224/1254,…`
`close_decision_snapshot`, `@@ -1267,…` `get_recent_realized_pnl` docstring.

### `src/bot.py` — A hunks
| Hunk | Content | Scope |
| --- | --- | --- |
| `@@ -1348,6 +1404,27 @@` | `_fee_source` derivation + quote extraction | **A** |
| `@@ -1356,7 +1433,11 @@` | pass the new kwargs to `save_order_record` | **A** |

**B/C hunks in bot.py (exclude):** `_record_committee_outcome` (recon = B,
excursion = C), `reconcile_open_snapshots` (B), `process_signal_for_symbol`
(capture-before-clear = C).

### Tests — A
| File | Scope |
| --- | --- |
| `tests/test_submission_quote_capture.py` | **A** — but `:238` must be flipped to `False` per §1 |
| `tests/test_fee_provenance.py` | **A** — 22 `fee_source` refs; all A |

**Not in A:** `test_position_outcome_reconstruction.py` (B/C),
`test_analyze_deployed_trades.py` + research files.

---

## 4. SAFETY REVIEW (as required)

| Concern | Status in the A-as-written code |
| --- | --- |
| **Shared API-key rate limit** | Both clients use the **same** `api_key`; Alpaca limits per key. Quote capture consumes the order path's quota. Isolation does **not** fix this. `get_latest_quote` has **no `@retry`**, so a 429 fails open (returns `None`) — capture cannot retry-loop into a quota hole. |
| **Bounded quote worker pool** | **NOT implemented.** Capture uses the shared default executor. This is the isolation change, deliberately separate. |
| **Caller timeout** | **NOT implemented.** No `asyncio.wait_for` on the capture. |
| **Timeout ≠ thread stop** | Confirmed: `wait_for` raised at 0.5s while the thread ran to 3.0s. Bounds latency, not the worker. |
| **Failure isolation** | **Present.** Capture is wrapped in `try/except Exception` bound outside the submit try; a quote error cannot block an order. |
| **Never fabricates a cost** | **Present.** Missing/partial quote persists **NULL**, never `0.0`; tests assert this. |

**Net:** A-as-written is failure-isolated but **not** worker-isolated and
**not** time-bounded. Hence §1's recommendation of `default=False` until the
isolation change lands.

---

## 5. VALIDATION PLAN (must pass before staging/committing A)

1. `pytest tests/test_submission_quote_capture.py tests/test_fee_provenance.py -q`
2. `pytest tests/ -q` — full suite (expect 600 + A's tests)
3. `uvx ruff check .` — `All checks passed!`
4. **State-file check** after the suite: tracker `44136fa3…`, drift `82a78c0d…`
   byte-identical (D's isolation must still hold).
5. **Migration check on a COPY** of `bot_snapshot.db` (never the original):
   confirm `_ensure_additive_columns` adds the four `orders` columns and is
   idempotent on a second run.
6. **Contamination check:** staged A diff contains **no** `outcome_status`,
   `reconstruct_position_outcome`, `excursion`, `max_fav`, `max_adv`.
7. **Default check:** `settings.CAPTURE_SUBMISSION_QUOTE is False` (per §1).

---

## 6. NOT DONE (held)

No staging, edits, commits, pushes, DB changes, deployment, or orders. Awaiting
the §1 default decision and separate authorization to stage A.

## Safety state

Branch `w1-instrumentation-accounting`; HEAD `6c46fb1`; `main` `7a8ca12`
untouched; **0 staged**; tracker/drift/snapshot/backup unchanged
(`44136fa3…` / `82a78c0d…` / `52e39180…` / `ddb2e987…`).
