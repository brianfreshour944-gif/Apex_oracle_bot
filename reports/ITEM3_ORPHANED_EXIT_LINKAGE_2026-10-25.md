# ITEM 3 — Orphaned-exit linkage investigation & repair proposal (READ-ONLY)

Date: 2026-10-25. **READ-ONLY. No DB writes, no relinking, no backfill, no source
change; nothing staged/committed/deployed.** Simulation was done on `/tmp` copies
of the snapshot only; the original is byte-identical (`52e39180…`, re-verified).

Base: HEAD `7a8ca12` + accounting patch + held item-1 `fee_source`.
Evidence: `bot_snapshot.db` opened `mode=ro&immutable=1`.

---

## 1. Executive summary

All **11 orphaned sells link uniquely** to a closed decision — and the evidence is
much stronger than "same symbol + same second". But the investigation found
something more important: **the missing links are a symptom, not the disease.**
The exit legs were written without `decision_id`, *and* the entry legs of the same
trades were **split across multiple foreign decision IDs** that have no snapshot.
Repairing the exit link alone is **insufficient** — the quantity still fails to
reconcile.

Even after a *complete* relink, **none of the 11 can be certified**, because they
all predate the fee-estimation code: every fee is `0.0`, and the reconstruction's
own fee gate correctly refuses to certify a fee-free round trip.

**Bottom line: the safely recoverable count is 0.** The linkage is provable; the
*certification* is not recoverable from this snapshot. That is a stronger and more
useful result than "11 fixed."

---

## 2. The 11 orphaned sells and their candidates

| Sell `order_id` (8) | Symbol | Sell qty | Linked decision | Snap qty | ratio | Δt (s) | candidates |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `9ce9d339` | BTC/USD | 0.000578549 | `930a65f5fd…` | 0.000580 | 0.997498 | −0.004 | 1 |
| `fadeae03` | BTC/USD | 0.000602487 | `b66773a3fb…` | 0.000604 | 0.997495 | −0.005 | 1 |
| `aedcc08d` | BTC/USD | 0.000590518 | `1d0a22f8f4…` | 0.000592 | 0.997497 | −0.005 | 1 |
| `7e0b5abf` | BTC/USD | 0.000598498 | `ad018313e5…` | 0.000600 | 0.997497 | −0.006 | 1 |
| `f8aab1c4` | BTC/USD | 0.000610468 | `a5c225a28b…` | 0.000612 | 0.997497 | −0.005 | 1 |
| `279878db` | BTC/USD | 0.000596503 | `21f6a8c1f3…` | 0.000598 | 0.997497 | −0.012 | 1 |
| `d5b6f25b` | BTC/USD | 0.000594508 | `6cb5b42b1d…` | 0.000596 | 0.997497 | −0.004 | 1 |
| `519c0c27` | BTC/USD | 0.000605479 | `edad1c2a81…` | 0.000607 | 0.997494 | −0.006 | 1 |
| `e61d83b9` | BTC/USD | 0.000591515 | `45a1989bb4…` | 0.000593 | 0.997496 | −0.007 | 1 |
| `99bc8c32` | BTC/USD | 0.000614460 | `968d419347…` | 0.000616 | 0.997500 | −0.004 | 1 |
| `62f77ded` | ETH/USD | 0.015332572 | `82caeaba02…` | 0.015371 | 0.997500 | −0.005 | 1 |

**Uniqueness holds both directions:** every orphan sell has exactly **one**
candidate decision within 60 s, and **no decision has more than one** candidate
sell (verified by reverse count). The Δt is a consistent **−5 ms** (submitted_at
just before closed_at), and `ratio = sell_qty/dec_qty = 0.9975` is **exactly**
`1 − entry_fee_rate` — the fee-withheld-in-base convention from the item-1 audit.

**Independent corroboration — `client_order_id`:** each sell's
`client_order_id` is `"<SYM>_sell_<qty>_<epoch>"`, and the epoch decodes to the
**same second** as `submitted_at` (e.g. `…_1790911181` → `03:19:41`). This is a
second, independent key beyond symbol+timestamp.

**Classification of the 11 candidates:** all **uniquely supported** by the
timestamp+quantity+client_order_id triple. **None ambiguous, none unresolvable.**

---

## 3. The deeper finding — the disease behind the symptom

Linking the sells is **not enough**. The reason is visible in the BTC ledger:
every BTC entry has **two buys under different decision IDs**:

```
03:16:43  buy 0.000290  dec=930a65f5fd  ← the snapshot's own leg
03:17:42  buy 0.000290  dec=a08597aaa6  ← SAME trade, DIFFERENT id, NO snapshot
03:19:41  sell 0.00057855  dec=NULL      ← 2 × 0.00029 = 0.00058 = snapshot qty
```

- The snapshot's `qty` is **0.00058** = **exactly `2 × 0.00029`**.
- Verified for **10/10 BTC rows**: `snapshot_qty == 2 × linked_buy_qty` (ratios
  0.99832–1.00000; two rows are 1 buy + 1 sell-leg-without-a-buy).
- The second buy carries decision ID `a08597aaa6`, which is **not in
  `decision_snapshots`** — nor are any of the other 9 paired IDs.

So each "one trade" is stored as **two decision IDs' worth of buys**, only one of
which has a snapshot. There are **18 distinct decision IDs in `orders` with no
snapshot row** (10 BTC buys, 3 ETH buys, 4 SOL buys, 1 SOL sell).

**ETH confirms the pattern at a different scale** (`82caeaba02`): the snapshot qty
`0.015371` equals the sum of **four** buys — `0.003994 (own) + 0.003994 + 0.003681
+ 0.003702` = **0.015371 exactly** — three of which sit under foreign IDs.

**Implication:** the ledger's *entry* legs are fragmented across phantom decision
IDs. A repair that only re-attaches the sell would produce `quantity_mismatch`
(proven below). A complete repair must also decide what to do with the foreign-ID
entry legs — which is a **data-model** question, not a simple relink.

---

## 4. Simulation (on `/tmp` copies — never the original)

### 4a. Relink the sell only → still fails
Setting each orphan sell's `decision_id` to its matched decision and re-running
the **real** `reconstruct_position_outcome`:

```
930a65f5fd … 82caeaba02   → unreconciled / quantity_mismatch   (all 11)
buy_qty=0.00029  sell_qty=0.000578549   ← sell is ~2× the linked buy
```

So **exit-only relinking is insufficient** — as the reviewer anticipated ("matching
only by symbol and timestamp is not enough").

### 4b. Relink the sell **and** the paired second buy → quantities pass, fee gate fails
With the full hypothesis (sell + the paired foreign-ID buy both assigned to the
decision):

```
930a65f5fd … 968d419347   → unreconciled / unverified_fee_provenance
82caeaba02                → unreconciled / unverified_fee_provenance  (after 4 buys)
```

The quantity now reconciles, but the reconstruction **correctly refuses to
certify**, because the fee gate requires a nonzero fee on each leg and these
trades have **fee = 0.0**.

### 4c. Why the fee gate blocks all 11
- The **first nonzero commission on any order** is `2026-10-13 02:08:04`.
- The **latest orphan sell** is `2026-10-05 20:58:45`.
- All **11/11** orphan sells have `commission = 0.0`; all their buys too.

The orphaned trades **predate the fee-estimation code** (deployed ~2026-10-13).
Their fees were never recorded, and — per the item-1 qualification — a `0.0` means
"no fee recorded", **not** "fee-free". The reconstruction therefore cannot certify
them, and that is the **correct, fail-closed** behaviour.

---

## 5. Safe-recovery count and learner-sample impact

| Class | Count | Can be certified? |
| --- | ---: | --- |
| Uniquely linked AND quantity-reconcilable AND fee-present | **0** | — |
| Uniquely linked, quantity-reconcilable, **fee absent** | **11** | **No** (fee gate) |
| Ambiguous | 0 | — |
| Unresolvable | 0 | — |

**Safely reconstructable outcomes: 0 of 11.** The linkage is provable; the
*certification* is unrecoverable from this snapshot because the fees were never
recorded.

**Effect on the eligible learner sample:** none. Since relinking (even a complete
one) yields `unreconciled`, filtering to `reconciled` would still keep exactly the
**same 11 of 24** rows as the item-2 analysis found — **no BTC, no
`low_volatility`** — and would **still** drop `sideways` 11→9 (below
`VALIDATION_MIN_TRADES=10`). Relinking does **not** relieve the item-2 bias; only a
**fee** recovery would.

---

## 6. Repair proposal (PROSE / PSEUDODIFF ONLY — not applied)

**Recommendation: do NOT repair, and do NOT relink.** Two independent reasons:

1. **Exit-only relinking does not achieve the goal** — it yields
   `quantity_mismatch` (§4a). A correct repair must also re-assign the fragmented
   foreign-ID entry legs, i.e. **rewrite the ledger's ownership model**, which is a
   structural change far beyond "fix 11 rows".
2. **Even a complete relink cannot certify the rows** — the fee gate blocks them
   (§4b/c). So the repair would buy *no* learner-eligible rows, at the cost of
   mutating historical data.

If, after review, a **classification-only** step is still wanted (marking rows
without altering P&L), the minimal safe option is:

```
# PROPOSED (not applied): a read-only audit VIEW / report, no writes.
# For each orphan sell, emit (sell_order_id, matched_decision_id,
#   match_evidence = {dt_sec, qty_ratio, client_order_id_epoch}, uniqueness=1)
# and leave orders/decision_snapshots UNCHANGED.
```

This gives the reviewer the evidence to decide, **without** touching data.

**If relinking were ever authorized**, it must be gated on **all** of:
- unique match in **both** directions;
- `|sell_qty/dec_qty − 0.9975| ≤ tol`;
- `client_order_id` epoch == `submitted_at` second;
- the resulting reconstruction reports `reconciled` (it will not, today).

Anything less must stay `unreconciled`.

---

## 7. How legacy databases should be handled safely

- **Never write `outcome_status='reconciled'` without a nonzero fee on both legs.**
  The reconstruction already enforces this; keep it.
- **Do not treat `NULL`/legacy as reconciled** (item-2 rule).
- A relink that cannot pass the fee gate should leave the row **`unreconciled`**,
  not "reconciled with a caveat".
- Prefer a **read-only classification artifact** over mutating `orders`.

## 8. Tests (for a future, separately-authorized change)

1. Orphan-sell matcher returns **exactly one** candidate for each of the 11, with
   `dt ≤ 5 ms`, `ratio ≈ 0.9975`, and epoch-consistent `client_order_id`.
2. Reverse uniqueness: no decision matches >1 sell.
3. **Regression guard:** with the sells relinked but fees still 0, the classifier
   must return `unreconciled / unverified_fee_provenance` (do **not** certify).
4. Quantity guard: sell-only relink ⇒ `quantity_mismatch` (documents §4a).
5. Legacy guard: no writes to `orders`/`decision_snapshots` in the report path.
6. **No-risk-change guard:** `get_recent_realized_pnl` unchanged by any of this.

## 9. Unresolved questions

1. Why do paired entries get **foreign decision IDs** with no snapshot? (A
   scale-in / re-entry path writing a fresh id per leg?) This is the real defect
   and is **not** fixed by relinking.
2. Should the 18 phantom-ID entry legs be treated as part of their sibling
   decision, or as separate (unsnapshotted) trades?
3. Since fees are unrecoverable pre-2026-10-13, is the historical window simply
   **unauditable**, and should the learner treat it as such?
4. Do we want a **fee-provenance backfill** (item 1 territory) before any relink,
   so the gate *could* pass? (Would still be an estimate, not verified.)

## 10. Non-goals (explicit)

- No DB writes, relinking, backfill, or historical P&L edits.
- No change to risk controls, the rolling loss gate, Kelly sizing, decay alerts,
  or the validation gate.
- No learner filter (item 2 remains unimplemented).
- No staging, commit, deploy, restart, or orders.

## 11. Recommendation

**Reject the "relink the 11 exits" repair.** It does not reconcile quantities on
its own, and even a complete relink cannot certify the rows because their fees
predate fee recording. The linkage question is **answered** (all 11 uniquely
supported), but the **recovery** question is answered *negatively*: **0 safely
recoverable outcomes**. The productive next step is to understand the **phantom
entry decision IDs** (§9 Q1) — that is the actual data-model defect — and to treat
the pre-2026-10-13 window as unauditable rather than repairing it.
