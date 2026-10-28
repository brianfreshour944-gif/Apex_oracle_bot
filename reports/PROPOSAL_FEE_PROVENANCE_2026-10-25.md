# PROPOSAL — Item 1: persist fee provenance (additive)

Status: **PROPOSAL ONLY — NOT IMPLEMENTED.** No code written, no schema changed,
nothing staged. Awaiting your review.

Scope: item 1 of the five audit follow-ups, and **only** item 1. Items 2–5
(learner filter, linkage backfill, loss-gate filter) are explicitly out of scope.

---

## 1. Problem being solved

`orders.commission` is a plain `Float` with no provenance (`src/db.py:164-186`).
The exchange adapter cannot observe a fee — Alpaca reports none — so it
*estimates* at `ESTIMATED_TAKER_FEE_BPS` (default 25) and flags
`commission_estimated = True` in the in-memory dict
(`src/exchange.py:924-953`). **That flag is never persisted**, so downstream
cannot distinguish an estimated fee from an observed one, and a stored `0.0`
means "no fee recorded", not "no fee charged" (audit Finding A).

This change makes the provenance durable so future records are self-describing.
It does **not** verify any fee, and it does **not** touch historical rows.

## 2. Proposed change (smallest safe, additive)

### 2.1 Schema — one nullable column on `orders`

```python
# src/db.py, class OrderRecord, after `commission`
    # Provenance of `commission`. NULL = unknown (never assume "verified").
    #   "estimated" = derived from ESTIMATED_TAKER_FEE_BPS by the adapter
    #                 (Alpaca reports no fee), NOT an exchange-observed charge.
    #   "exchange"  = a nonzero commission the adapter observed directly.
    # A stored 0.0 with NULL provenance means "no fee recorded", which is NOT
    # the same as "no fee charged".
    fee_source: Mapped[str | None] = mapped_column(String(16), nullable=True, default=None)
```

Nullable, additive, no default value written to existing rows — they stay `NULL`
(= unknown), which is exactly the honest label for them.

### 2.2 Migration — one entry in the existing additive mechanism

`_ensure_additive_columns()` already exists and is idempotent
(`src/db.py`, guarded by `inspect`). Add the column to its map:

```python
    additions = {
        "decision_snapshots": [("outcome_status", "VARCHAR(16)")],
        "orders":             [("fee_source", "VARCHAR(16)")],   # <-- new
    }
```

No new migration framework; `ALTER TABLE orders ADD COLUMN` only, only when the
column is absent. Existing rows are not rewritten.

### 2.3 Writer — explicit kwarg on `save_order_record`

```python
def save_order_record(
    *,
    order_id: str,
    ...
    commission: float = 0.0,
    fee_source: str | None = None,      # <-- new, default = unknown
    ...
):
    ...
    order = OrderRecord(..., commission=float(commission), fee_source=fee_source, ...)
```

`save_order_record` stays a dumb writer: the caller states provenance explicitly.

### 2.4 Derivation — the one funnel, `bot._persist_order_record`

The adapter already sets `commission_estimated` in the result dict. The funnel
derives `fee_source` from it — **so `src/exchange.py` needs no change at all**:

```python
    commission = float(order_result.get("commission", 0.0) or 0.0)
    if order_result.get("commission_estimated"):
        fee_source = "estimated"      # adapter derived it from the bps rate
    elif commission > 0.0:
        fee_source = "exchange"       # nonzero fee observed directly
    else:
        fee_source = None             # no fee recorded -> unknown, NOT "verified"
    save_order_record(..., commission=commission, fee_source=fee_source, ...)
```

Rationale for deriving here rather than editing the adapter: it is one edit in one
funnel that every ledger write already passes through, it cannot drift from the
adapter's own flag, and it leaves the adapter untouched.

## 3. Files this would touch

| File | Change | Risk |
| --- | --- | --- |
| `src/db.py` | `OrderRecord.fee_source`; `additions["orders"]`; `save_order_record` kwarg | additive, nullable |
| `src/bot.py` | `_persist_order_record` derives + passes `fee_source` | ~6 lines, no behavior change |
| `tests/test_fee_provenance.py` | new tests (below) | new file |

**Not touched:** `src/exchange.py`, `src/risk.py`, `src/committee/`, any consumer,
the accounting patch's four files, the momentum72 artifacts.

## 4. Proposed tests (`tests/test_fee_provenance.py`)

Each pinned to an observable value; DB via the existing `fresh_db` fixture pattern.

| # | Test | Assertion |
| --- | --- | --- |
| 1 | `save_order_record(fee_source="estimated")` round-trips | read-back column == `"estimated"` |
| 2 | `save_order_record(fee_source="exchange")` round-trips | read-back == `"exchange"` |
| 3 | omitted `fee_source` | read-back is **`None`**, never `"exchange"` |
| 4 | `_persist_order_record` with `commission_estimated=True`, commission>0 | stored `"estimated"` |
| 5 | `_persist_order_record` with observed commission>0, no flag | stored `"exchange"` |
| 6 | `_persist_order_record` with commission==0, no flag | stored `None` (unknown) |
| 7 | migration: pre-existing `orders` table without the column | column added; existing rows `NULL`; second call is a no-op (idempotent) |
| 8 | **no behavior change**: `reconstruct_position_outcome` output is identical with and without the column populated | guards against accidental coupling |

Plus: the full suite must stay green, and `tests/test_fee_recording.py` must still
pass unchanged (its assertions are about `commission`, not provenance).

## 5. Explicit boundaries (honored by this proposal)

- **No change to the existing accounting patch** without separate approval.
- **No historical backfill and no relinking** — existing rows keep `NULL`.
- **No learner or loss-gate behavior change** — no consumer reads `fee_source`.
- **No staging, commits, deployment, restarts, or live orders.**
- Nothing implemented until you approve this proposal.

## 6. Points needing your ratification before implementation

1. **Column name / value domain.** I propose `fee_source` ∈
   `{"estimated","exchange", NULL}` rather than a boolean
   `commission_estimated`. A string distinguishes *unknown* from *verified* and
   leaves room for a later third state; a boolean would tempt a default of
   `False` (= "verified") on old rows, which would be a lie. Do you agree?
2. **Meaning of `"exchange"`.** On this deployment the adapter never observes a
   fee, so `"exchange"` is currently unreachable — it is future-proofing for a
   venue that does report fees. Acceptable, or would you prefer to defer the
   value until a real observed fee exists?
3. **`0.0` + `NULL` semantics.** I define a stored `0.0` with `NULL` provenance
   as "no fee recorded" (unknown), **not** "fee-free". Confirm this is the
   intended reading, since it is what the audit's BTC rows would be labelled.
4. **No consumer wiring.** This change only *stores* provenance; nothing acts on
   it yet. Confirm that is the intended boundary (wiring the learner/loss gate is
   items 2 and 4, separately authorized).

## 7. Deliberately NOT proposed here

- Backfilling `fee_source` on the 24 historical rows (item 3 territory).
- Making `reconstruct_position_outcome` require `fee_source == "exchange"` to
  certify a row. That would reclassify all 11 currently-"reconciled" rows as
  uncertified — a **meaning change** for `outcome_status`, and the audit's central
  caution that *reconciled ≠ audited*. It belongs with item 2, after the meaning
  of "reconciled" is agreed.
