# Accounting & Learning-Integrity Audit (read-only)

Date: 2026-10-25. Scope: how realized P&L flows from exchange fills → `orders` →
`decision_snapshots` → the learner and the rolling loss gate, and which of those
numbers can be trusted. **Read-only. No files modified, nothing staged, no
deployment, no orders, accounting patch untouched.**

Base: HEAD `7a8ca12`, staged 0, patch `d1131e48…` unchanged.
Evidence source: `/workspace/project/bot_snapshot.db` (sha256 `52e39180…`), opened
`mode=ro&immutable=1`. The real classifier was run against **copies** in `/tmp`
(the original was never opened for write; a pristine copy is byte-identical, §7).

---

## 1. Headline

The bot's stored historical P&L is **not trustworthy as an audited record**, for
three independent reasons, and **13 of 24 closed trades cannot be certified from
the order ledger at all** — yet those rows still feed both the learner and the
rolling loss limit.

| Question | Answer |
| --- | --- |
| Are recorded fees actual exchange charges? | **No** — every fee is an estimate; the estimate flag is not even stored |
| Is the deployed ledger's exit linkage complete? | **No** — 11 of 24 closed trades have no sell leg under their `decision_id` |
| Do stored and reconstructed P&L agree? | **No** — 2 of 11 reconcilable rows disagree materially, one flips sign |
| Do untrusted rows reach the learner / loss gate? | **Yes** — no consumer filters on trust status |

## 2. Data-flow trace (verified by reading the code)

```
exchange fill  ──► orders row (commission = estimated or 0)
                       │
                       ▼
        reconstruct_position_outcome(decision_id)      [src/db.py:~455-700]
          • assembles buys+sells from the ledger
          • fails closed ("unreconciled") if a leg is missing,
            quantities don't tie, or a needed fee is 0
                       │
                       ▼
        close_decision_snapshot(..., outcome_status)    [src/db.py:1617+]
                       │
        ┌──────────────┴───────────────────────────────┐
        ▼                                              ▼
  learner label                                  rolling loss gate
  bot.py:295  label = 1 if realized_pnl>0        get_recent_realized_pnl()
  bot.py:3600 learner replay history             db.py:1670  → bot.py:2159
  (NO outcome_status filter)                     (NO outcome_status filter)
```

Both consumers use `realized_pnl` **regardless of `outcome_status`** — this is
documented in the code itself (`db.py:78-98`, `db.py:1670+`) as deliberate for
the loss gate (fail-safe) but is a data-quality hazard for the learner.

## 3. Finding A — fee provenance is estimated, and the estimate flag is not stored

**Verified.** `orders.commission` is a plain `Float` with no provenance column
(`src/db.py:164-186`). The exchange adapter cannot observe a fee: Alpaca reports
none, so `_apply_estimated_commission` (`src/exchange.py:924-953`) sets
`commission = filled_avg_price × filled_qty × ESTIMATED_TAKER_FEE_BPS/1e4`
(default **25 bps**, `config.py:414`) and marks `commission_estimated = True`.

**But `commission_estimated` is never persisted** — `save_order_record`
(`db.py:1617+`) and `OrderRecord` have no such field. So downstream cannot
distinguish an estimated fee from an observed one, and a recorded `0.0` means
"no fee recorded", **not** "no fee charged".

Snapshot evidence:

| Symbol | Orders | Nonzero-commission | Sum | Date range |
| --- | --- | --- | --- | --- |
| BTC/USD | 30 | **0** | 0.0 | 2026-10-02 only |
| ETH/USD | 27 | 22 | 0.8167 | 2026-10-02 → 10-24 |
| SOL/USD | 27 | 20 | 0.6719 | 2026-10-02 → 10-24 |

All 30 BTC orders are dated **2026-10-02** and carry commission 0; the earliest
nonzero fee on any symbol is **2026-10-13**. That is consistent with the
fee-estimation code being deployed between those dates — i.e. BTC rows were
written by a build that recorded no fee at all. Classification: **BTC fees =
UNKNOWN** (not zero); **ETH/SOL fees = ESTIMATED** (not verified).

## 4. Finding B — 11 of 24 closed trades have no exit leg in the ledger

**Verified, and the exits are not actually missing — their linkage is.**

- 84 filled orders: **11 have NULL `decision_id`**, 18 point at decision IDs that
  are absent from `decision_snapshots`.
- 24 closed decisions: **13 have both legs, 11 have a buy leg only** (0 have
  sell-only or no legs).
- All **11** buy-only decisions match a NULL-`decision_id` sell on
  (symbol, `submitted_at` == `closed_at` within 5 s, quantity within 1%):
  Δt = **0.0 s** on all 11, and the quantity gap is **exactly 0.250%** — which is
  precisely the 0.0025 entry-fee rate that the fee convention predicts
  (§5). So the exit fills exist; the `decision_id` was lost on the exit leg.

Running the **real** `reconstruct_position_outcome` against a copy:

| Outcome | Count | Reasons |
| --- | --- | --- |
| reconciled | 11 | — |
| unreconciled | 13 | `no_linked_exit_fills` ×11, `quantity_mismatch` ×1, `multiple_position_lifecycles` ×1 |

So **more than half the deployed history cannot be certified from the ledger** —
not because the trades didn't happen, but because the exit legs are unlinked.

## 5. Finding C — the fee convention itself is sound (verified)

On the 13 both-leg rows the ratio `sell_qty / buy_qty` is **0.997500** on 12 of
them (one is a multi-lifecycle row, `1fd2749865`, correctly failed closed). That
is exactly `1 − 0.0025`: the entry fee is withheld in the **base** asset, so it is
already inside the reduced proceeds and must not be subtracted twice. This
confirms the convention documented in `db.py` and the discriminator the patch
uses. The odd row is a genuine data problem, not a code bug — the same
`decision_id` was reused across two lifecycles (buy 0.1517 on 10-08, sell 0.7474
on 10-09), which the patch correctly refuses to merge.

## 6. Finding D — stored vs reconstructed P&L disagree, and the learner inherits the error

Comparing the snapshot's stored `realized_pnl` to the real classifier on the 11
reconcilable rows:

| decision | stored | reconstructed | abs Δ | verdict |
| --- | --- | --- | --- | --- |
| 8a4518ec03 | +0.15875621 | +0.15875621 | 0 | match |
| f110556591 | **−0.03745913** | **+0.12588103** | **0.1633** | **MISMATCH (sign flip)** |
| 35e92c4ad1 | **−0.03761484** | **−0.21632586** | **0.1787** | **MISMATCH** |
| c47889fa4d | −1.42479047 | −1.42479298 | 0.0000025 | match |
| (7 others) | — | — | <1e-4 | match |

The two mismatches are exactly the rows with the absurd stored returns
**−804.76%** and **−70218.31%**. Both are dust-partial-exit rows: the stored
value was computed by the *old* path from the final dust leg's quantity against
the whole entry fee. Example (`f110556591`): buy 0.006049 @ 2479.85, then sells
0.002068 + 0.002068 + 0.001896 + **0.00000188** — the old code closed on the
`0.00000188` dust leg, producing a tiny basis and a nonsense percentage.

**Consequence for learning:** `bot.py:295` sets the learner label to
`1.0 if realized_pnl > 0 else 0.0`. For `f110556591`, the **stored** value
(−0.037) labels the trade **0**, while the **reconstructed** value (+0.126) is a
**winner (1)**. The learner and the loss gate both read the *stored* value, so
they are training on and reacting to the corrupted number.

## 7. Read-only guarantee (proven)

- Original `bot_snapshot.db` sha256 `52e39180…`; pristine `/tmp` copy
  `52e39180…` — **byte-identical**.
- The classifier was run with `DATABASE_URL=sqlite:////tmp/snap_audit.db` (a
  copy), because `_ensure_tables()` performs the additive `outcome_status`
  migration. The original was only ever opened `mode=ro&immutable=1`.
- The deployed snapshot has **no `outcome_status` column at all** (schema check),
  so every one of its 24 rows is trust **UNKNOWN**, never "reconciled".

## 8. Classification summary

| Item | Status |
| --- | --- |
| Fee convention (entry in base, exit in quote) | **VERIFIED** from the fills |
| Ledger arithmetic on the 11 both-leg rows | **VERIFIED** (9 exact; 2 corrected) |
| ETH/SOL fee amounts | **ESTIMATED** (25 bps), not exchange-verified |
| BTC fee amounts | **UNKNOWN** (recorded 0, no fee captured) |
| Stored `realized_pnl` for the 2 dust rows | **WRONG** (sign flip on one) |
| 13 of 24 closed rows certifiable from ledger | **NO** |
| `outcome_status` on deployed rows | **UNKNOWN** (column absent) |
| Exit-leg `decision_id` linkage | **LOST** for 11 of 24 trades |

## 9. Does this affect the learner or risk controls?

**Yes.**

- **Learner:** trains on `realized_pnl` / `return_pct` with **no** trust filter
  (`bot.py:295`, `bot.py:3600`, `get_closed_decision_snapshots`). The 2
  sign-flipped rows inject wrong labels; the 13 uncertified rows inject
  unverified ones.
- **Rolling loss gate:** `get_recent_realized_pnl` sums **every** closed row
  regardless of `outcome_status` (`db.py:1670`), by deliberate fail-safe design.
  It can therefore block (or fail to block) an entry on the basis of
  reconstructed, estimated-fee numbers. This is a *trading* effect.
- **Attribution:** reads `realized_pnl`/`return_pct` with no filter
  (`attribution.py:27`).
- **No consumer anywhere filters on `outcome_status`** (verified by grep) — the
  label is a warning, not an exclusion.

## 10. What is fixed vs. what is not

- The **patch fixes future writes** (`reconstruct_position_outcome` + the
  `outcome_status` column). It does **not** rewrite the 24 historical rows: they
  retain their stored (sometimes wrong) values.
- The patch **cannot** make fee provenance verified — `commission_estimated` is
  still not persisted, and the loss-gate/learner filters are explicitly out of
  its scope.

## 11. Issues requiring separate authorization to fix

None of the following were done; each needs its own decision because it changes
either stored data or trading behavior:

1. **Persist fee provenance** — add an `commission_estimated` (or
   `fee_source`) column to `orders` and surface it, so a `0` can be told from an
   estimate. Additive schema change.
2. **Backfill `outcome_status`** on the 24 historical rows by re-running the
   classifier — changes stored data; needs authorization.
3. **Re-link the 11 orphaned exit legs** (match on symbol + `submitted_at` ==
   `closed_at`) — changes stored data; needs authorization.
4. **Filter the learner** to `outcome_status == "reconciled"` — changes training
   inputs.
5. **Reconsider the loss-gate filter** — changes trading behavior (block/no-block);
   explicitly out of the accounting patch's scope.

## 12. Bottom line

The deployed bot's historical P&L is a **best-effort reconstruction with
estimated fees**, not an audited record. On this snapshot: fees are never
verified, 13 of 24 trades cannot be certified from the ledger (11 due to lost
exit linkage), and 2 rows carry a materially wrong stored P&L — one with the
sign flipped — which the learner and loss gate consume anyway. The arithmetic
*convention* is correct; the *data provenance and linkage* are not. This is a
separate problem from the Momentum72 verdict and was not affected by it.
