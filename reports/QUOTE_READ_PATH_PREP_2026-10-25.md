# PREP — QUOTE READ PATH DESIGN (READ-ONLY)

Date: 2026-10-25. **Read-only.** No edits, staging, commits, DB changes, push, or
deploy. `CAPTURE_SUBMISSION_QUOTE` stays **False**.

Authorized by bare "Authorize prep for quote read path". This designs the
consumer that would make the captured quote useful. It implements nothing.

Companion to `reports/CAPTURE_DEFAULT_DECISION_PREP_2026-10-25.md`, which
established that the capture is **write-only** and that a read path is the
prerequisite for either "enable" option.

---

## 1. The gap, restated precisely

Three columns are written and never read:

| Column | Written at | Type |
| --- | --- | --- |
| `mid_at_submit` | `bot.py:1438` | FLOAT, nullable |
| `spread_at_submit` | `bot.py:1439` | FLOAT, nullable |
| `quote_ts` | `bot.py:1440` | DATETIME(tz), nullable |

(`fee_source`, added by commit A, is likewise write-only — same class of gap.)

Confirmed absent from every reader: `get_recent_order_records`
(`db.py:775`) selects 13 fields and omits all four; the analyzer's
`_ORDER_COLS` (`scripts/analyze_deployed_trades.py:31`) omits them; no
`classify_submission_quote` helper exists.

**Note on a stale memory entry:** memory claimed a `classify_submission_quote`
helper, a `DEFAULT_QUOTE_MAX_AGE_SEC` constant, and quote read-back in
`get_recent_order_records`. **None exist in this tree** (grep: no hits; no
branch contains them). That describes work lost with the wiped sandbox. The
design below re-derives the need rather than assuming it exists.

## 2. Two candidate consumers

| | A. Offline analyzer | B. Runtime reconcile path |
| --- | --- | --- |
| Where | `scripts/analyze_deployed_trades.py` | `db.get_recent_order_records` → `bot.py` |
| Input | a downloaded snapshot (read-only) | the live DB |
| Purpose | measure execution cost post hoc | recover real fill price on restart |
| Risk of change | **none** (offline, stdlib) | touches the live reconcile path |
| Blocked by | nothing | needs care; runs during startup recovery |

**Recommendation: build A first.** It is offline, stdlib-only, already opens the
snapshot `mode=ro&immutable=1`, and cannot affect trading. B is a separate,
higher-stakes change and is **not** proposed here.

## 3. What the read path should compute

The capture gives, per order row: `mid_at_submit`, `spread_at_submit`,
`quote_ts`, plus the existing `filled_avg_price`, `filled_qty`, `side`, `qty`.

### 3.1 Effective slippage vs the contemporaneous mid

For a **buy** (entry), paying above mid is a cost:

```
slippage_bps = (filled_avg_price - mid_at_submit) / mid_at_submit * 1e4
```

For a **sell** (exit), receiving below mid is a cost:

```
slippage_bps = (mid_at_submit - filled_avg_price) / mid_at_submit * 1e4
```

Positive = cost, in both cases. This is the key convention to pin with a test,
because the sign flip between buy and sell is the easiest thing to get wrong.

### 3.2 Half-spread decomposition (the actual W1 payoff)

The half-spread is the *unavoidable* cost of crossing; the rest is timing/impact:

```
half_spread_bps   = (spread_at_submit / mid_at_submit) / 2 * 1e4
excess_slippage_bps = slippage_bps - half_spread_bps
```

- `half_spread_bps` — what a market order should pay to cross a quoted spread.
- `excess_slippage_bps` — adverse move between quote capture and fill, plus
  impact. **This is the number W1 exists to isolate**: it separates "the spread
  costs us" from "the market moved against us while we submitted".

Aggregate per symbol / regime / exit reason: mean and median of both, plus the
share of orders where `excess_slippage_bps > 0`.

### 3.3 Stale-quote exclusion (why `quote_ts` exists)

A quote captured long before the fill is not a valid reference. The read path
must:

- compute `quote_age_sec = fill_time - quote_ts` (prefer `filled_at`, fall back
  to `submitted_at`);
- **exclude** rows where age exceeds a threshold (a `DEFAULT_QUOTE_MAX_AGE_SEC`,
  e.g. 5 s) from slippage statistics, and **count** them separately;
- treat `quote_ts IS NULL` as "unknown age" — excluded, never assumed fresh.

Reporting excluded/unusable counts is mandatory: silently dropping rows would
recreate the trust problem this whole workstream is fixing.

### 3.4 NULL discipline

| State | Meaning | Handling |
| --- | --- | --- |
| `mid_at_submit IS NULL` | capture off or failed | excluded, counted as `no_quote` |
| `quote_ts IS NULL` but mid present | age unknown | excluded, counted as `no_quote_ts` |
| `spread_at_submit <= 0` | bad data | excluded, counted as `bad_spread` |

Never substitute zero. A zero spread would read as "free execution".

## 4. What this does NOT do

- **Does not touch the live order path.** Analyzer-only.
- **Does not change the capture default.** Still `False`; the analyzer reads
  whatever a snapshot contains, including zero rows when capture was off.
- **Does not prove the fill is exchange-accurate.** It compares recorded fill to
  recorded quote; both come from the same ledger.
- **Does not resolve the orphan-order gap** (11 orders with no `decision_id`, 18
  with orphan ids, per the deployed audit). Those rows can still be analyzed
  individually for slippage since the quote is on the order row, but cannot be
  attributed to a decision.

## 5. Concrete change sketch (analyzer only, for a later authorization)

1. Extend `_ORDER_COLS` with `mid_at_submit`, `spread_at_submit`, `quote_ts`,
   `fee_source` (guarded: tolerate their absence on old snapshots — see §6).
2. Add a `_slippage_bps(side, fill, mid)` helper + `_half_spread_bps(spread, mid)`.
3. Add a `slippage` section to `analyze()`: per-symbol and overall mean/median of
   `slippage_bps`, `half_spread_bps`, `excess_slippage_bps`, plus
   excluded-counts by reason.
4. Render the section in `render()`.
5. Tests, hand-computed (the analyzer's existing convention — every test pins a
   hand-derived value).

## 6. Backward-compatibility trap (important)

Old snapshots (and the local `data/bot.db`, verified) **lack these columns
entirely**:

```
local bot.db orders cols: order_id, decision_id, symbol, side, qty, filled_qty,
filled_avg_price, commission, status, type, time_in_force, client_order_id,
submitted_at, filled_at        # no mid_at_submit/spread_at_submit/quote_ts/fee_source
```

`_rows()` builds `SELECT <cols> FROM <table>` verbatim, so adding a column to
`_ORDER_COLS` would make the analyzer **crash on every pre-existing snapshot**.
The read path must introspect `PRAGMA table_info(orders)` and select only the
columns present, reporting the missing ones as "not captured". This is a real
correctness requirement, not a nicety.

## 7. Sequencing

1. (this) read-path design — done.
2. Analyzer read-path implementation + hand-computed tests — separate auth.
3. Verify against a snapshot **that has** quote data (none exists yet — capture
   is off, so step 4 must precede real verification).
4. Then the capture-default decision can be made on evidence: with a consumer in
   place, `True` starts producing *usable* measurements.

So the honest ordering is: **consumer → enable → measure.** Enabling before the
consumer exists collects rows nothing reads.

## Boundary

This prep authorizes **nothing**. `CAPTURE_SUBMISSION_QUOTE` stays **False**. No
push, no deploy. The live reconcile path (consumer B) is explicitly out of scope.

## Safety state

Branch `w1-instrumentation-accounting`, HEAD `8945afd` (6 ahead), `main` `7a8ca12`
unchanged; 0 staged, 0 unstaged.
