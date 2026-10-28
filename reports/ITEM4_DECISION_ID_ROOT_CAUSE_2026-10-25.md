# ITEM 4 — Decision-ID creation & snapshot-persistence root cause (READ-ONLY)

Date: 2026-10-25. **READ-ONLY. No source, test, config, or DB change; no
relinking/backfill/repair; nothing staged/committed/deployed.** Evidence:
`bot_snapshot.db` `mode=ro&immutable=1` (`52e39180…`) + git history at HEAD
`7a8ca12`.

Three concepts are kept **strictly separate** throughout: **linkage confidence**
(order↔decision), **accounting confidence** (qty/fees/PnL reconstructable), and
**learner eligibility** (suitable for training).

---

## 1. Executive summary

**Root cause found, and it is a historical code defect, not an ongoing one.**

Every one of the **18 foreign-ID orders** and the **11 NULL-decision exits** is
explained by **two ordering bugs in the entry/exit ledger-keying code that was
live 2026-10-02 → 2026-10-09**. Both were fixed on **2026-10-08/09**
(commits `3384e91`, `59bddac`, `6a91345`), all ancestors of HEAD:

1. **Entry keying (Defect A):** the ledger row was keyed on the **committee's
   fresh `decision_id`** even when `save_decision_snapshot()` **folded** the entry
   into an existing open snapshot — so the buy persisted under an ID that has no
   snapshot row. → the 10 BTC + 3 ETH + 4 SOL **foreign-ID buys**.
2. **Exit keying (Defect B):** the close path persisted the exit with **no
   `decision_id`** at all (`_persist_order_record(..., decision_id=None)`) before
   `_snapshot_decision_id()` existed. → the 11 **NULL-decision sells**.

**Classification:** all 18 foreign IDs = **CONFIRMED** (fold artifact, not
legitimate child-order IDs). 11 NULL exits = **CONFIRMED** (missing exit keying).
The **SOL `1fd2749865`** case = **CONFIRMED same root cause**, different
manifestation (a pre-fold-era duplicate). **No case is ambiguous or unresolved.**

**Prospective impact: none.** The defects are already fixed at HEAD; no
post-2026-10-13 order exhibits either pattern. **Historical impact: the affected
records stay unauditable** — the defect is explainable, but repair is still
rejected (§7), because the fees were never recorded.

---

## 2. Decision-ID creation, propagation, and defaults

| Where | What | Path |
| --- | --- | --- |
| **Generation** | one per committee run: `decision_id = uuid.uuid4().hex` | `committee.py:286` |
| **Snapshot create** | `save_decision_snapshot(decision_id=committee_result.decision_id, …)` | `bot.py:2806-2807` |
| **Snapshot FOLD** | if an open snapshot exists for the symbol, **fold** the new entry in and **return without creating a row for the new id** | `db.py:1332-1357` |
| **Scale-in update** | `update_decision_snapshot_position(snap["decision_id"], …)` | `bot.py:2828-2840` |
| **Entry ledger key** | `ledger_decision_id = committee_result.decision_id`, then **overwritten** by `get_open_snapshot(symbol)["decision_id"]` if one exists | `bot.py:2706-2712` |
| **Exit ledger key** | `_snapshot_decision_id(symbol)` → open snapshot's id, else **None** | `bot.py:1349-1363`, used at `1816`, `2887` |
| **Crash-gap reattach** | `decision_id = order.get("decision_id") or f"reattached_{order_id}"` | `bot.py:1585` |
| **Flatten (risk)** | `get_open_snapshot(symbol)` → id, else **None** | `risk.py:267-272` |

**The critical semantic:** a `decision_id` denotes a **committee decision**, but
the **snapshot row** denotes a **position lifecycle** (it is folded/weighted-
averaged across scale-ins). When the two diverge — a fold — the fresh committee
id is **silently discarded** (`db.py:1354` logs it, but no row is written for it).

---

## 3. Order-to-snapshot persistence (the ordering that produced the defect)

### Defect A — entry row keyed on a folded-away id (pre-`3384e91`)
At the commits active during the data window (`69c643b`, `9f33bf3`, `86c119a`):

```python
# PRE-FIX entry branch
await asyncio.to_thread(
    _persist_order_record, order_result, symbol, signal["action"], client_order_id,
    decision_id=committee_result.decision_id,     # <-- ALWAYS the fresh committee id
)
```

But `save_decision_snapshot()` (added `c6f4342`, 2026-09-27) **folds** a second
entry for the same symbol into the existing open snapshot and returns **without
writing** `committee_result.decision_id`. So on any scale-in / stale-positions
"new entry", the buy row is keyed to an ID **that will never have a snapshot**.

### Defect B — exit row keyed to nothing (pre-`6a91345`)
```python
# PRE-FIX close branch (e.g. 69c643b:2128)
await asyncio.to_thread(_persist_order_record, order_result, symbol, side, client_order_id)
#                                                       ^ decision_id omitted -> NULL
```
`_snapshot_decision_id()` did **not exist** before `6a91345` (2026-10-07); the
close path called `_persist_order_record` with no `decision_id`. → the exit rows
landed with `decision_id = NULL`.

### The fixes at HEAD
- `3384e91` (2026-10-08 23:56): `ledger_decision_id` resolves to the **open
  snapshot** first, falling back to the committee id only for a genuinely fresh
  entry. Entry rows now key on the snapshot.
- `6a91345` (2026-10-07 19:07): introduced `_snapshot_decision_id()` and wired it
  into the trailing-stop and close exits.
- `59bddac` (2026-10-08 23:23): attribute entry fees by `decision_id`; stop-fill
  reconcile coverage.

All three are **ancestors of HEAD** (verified). At HEAD the entry key is
`ledger_decision_id` (4 refs) and exits use `_snapshot_decision_id` (3 refs).

---

## 4. Classification of all 18 foreign-ID orders

| # | ID (10) | Sym | Side | qty | time | Folds into | Classification |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| 1 | `a08597aaa6` | BTC | buy | 0.000290 | 10-02 03:17:42 | `930a65f5fd` | CONFIRMED — fold artifact |
| 2 | `0a939f2ea4` | BTC | buy | 0.000302 | 10-02 03:58:45 | `b66773a3fb` | CONFIRMED |
| 3 | `32e6c57a56` | BTC | buy | 0.000296 | 10-02 06:39:58 | `1d0a22f8f4` | CONFIRMED |
| 4 | `c618fd5f53` | BTC | buy | 0.000300 | 10-02 10:01:15 | `ad018313e5` | CONFIRMED |
| 5 | `63a54d841a` | BTC | buy | 0.000306 | 10-02 10:36:17 | `a5c225a28b` | CONFIRMED |
| 6 | `7f81de1d56` | BTC | buy | 0.000299 | 10-02 10:45:18 | `21f6a8c1f3` | CONFIRMED |
| 7 | `17d221320b` | BTC | buy | 0.000298 | 10-02 10:54:19 | `6cb5b42b1d` | CONFIRMED |
| 8 | `5dd54c0183` | BTC | buy | 0.000303 | 10-02 11:09:20 | `edad1c2a81` | CONFIRMED |
| 9 | `721b25b1a5` | BTC | buy | 0.000296 | 10-02 11:32:22 | `45a1989bb4` | CONFIRMED |
| 10 | `2805488d25` | BTC | buy | 0.000308 | 10-02 11:41:22 | `968d419347` | CONFIRMED |
| 11 | `0a14b6034a` | ETH | buy | 0.003994 | 10-05 12:59:04 | `82caeaba02` | CONFIRMED |
| 12 | `c132dc8d08` | ETH | buy | 0.003681 | 10-05 14:04:11 | `82caeaba02` | CONFIRMED |
| 13 | `2584d78dbe` | ETH | buy | 0.003702 | 10-05 15:09:15 | `82caeaba02` | CONFIRMED |
| 14 | `0f24a2b607` | SOL | buy | 0.164204 | 10-08 16:16:08 | `1fd2749865` | CONFIRMED |
| 15 | `4aa1414a85` | SOL | buy | 0.134175 | 10-08 16:52:11 | `1fd2749865` | CONFIRMED |
| 16 | `c9d336ba07` | SOL | buy | 0.186218 | 10-08 17:56:18 | `1fd2749865` | CONFIRMED |
| 17 | `07d8a40875` | SOL | buy | 0.193491 | 10-08 19:01:23 | `1fd2749865` | CONFIRMED |
| 18 | `773ddfde98` | SOL | **sell** | 0.080375 | 10-09 00:00:26 | `1fd2749865` | CONFIRMED — see §5 |

**Every ID is a real committee decision whose row was never written because its
entry folded into a still-open snapshot.** They are **not** child-order IDs, not
duplicate IDs, not retry artifacts, and not missing writes of a *separate*
decision — they are the **discarded id of a folded scale-in**. Evidence: each is
a `buy` under the symbol's most-recent prior open snapshot, with quantity that
completes the snapshot total (§6).

---

## 5. The SOL `1fd2749865` cross-lifecycle case

**This is the same root cause in a different shape — CONFIRMED.**

```
snapshot 1fd2749865: qty=0.910206, created 10-08 16:05:07, closed 10-09 00:05:25
10-08 16:05:07  buy  0.151743  dec=1fd2749865   ← own leg (row exists)
10-08 16:16:08  buy  0.164204  dec=0f24a2b607   ← folded, no snapshot
10-08 16:52:11  buy  0.134175  dec=4aa1414a85   ← folded, no snapshot
10-08 17:56:18  buy  0.186218  dec=c9d336ba07   ← folded, no snapshot
10-08 19:01:23  buy  0.193491  dec=07d8a40875   ← folded, no snapshot
10-09 00:00:26  sell 0.080375  dec=773ddfde98   ← foreign-id SELL, no snapshot
10-09 00:05:25  sell 0.747381  dec=1fd2749865   ← own leg (row exists)
```

- **Arithmetic:** `snapshot_qty (0.910206) − 5 recorded buys (0.829831) =
  0.080375` = **exactly the foreign sell `773ddfde98`'s qty**. So the snapshot's
  quantity includes a **sixth buy (0.080375)** whose ledger row is missing — the
  scale-in that produced `773ddfde98` folded its *buy* away, yet its *sell*
  (a partial exit of that add) was persisted under the fold's discarded id.
- **Why it is the same cause:** the sell is keyed to the **committee id of the
  entry that folded** — exactly Defect A/B combined. `773ddfde98` appears in
  **no** snapshot and **no** shadow table.
- **Why it looks like "cross-lifecycle":** the *id* spans a buy and a sell, but
  neither has a snapshot, so the id is not a real lifecycle — it is the
  **throwaway id of one folded add**, whose buy row was mis-keyed (A) and whose
  partial-exit sell was keyed to the same throwaway (B).
- **Note:** `773ddfde98` is **not** a protective stop (`type=market`, not
  `stop_limit`) and there is **no partial-exit feature** in the code (verified) —
  so this is a full/partial market exit recorded under the fold id, consistent
  with the pre-`6a91345` "no decision_id plumbing" state.

**Verdict: CONFIRMED — same root cause, distinct manifestation.** The SOL case
therefore does **not** indicate a second defect class; it corroborates Defect A/B.

---

## 6. Confirming arithmetic (independent of the code reading)

- **BTC (10):** `snapshot_qty == 2 × linked_buy_qty` for all 10 (ratios
  0.99832–1.00000).
- **ETH `82caeaba02`:** snapshot `0.015371` = `0.003994×2 + 0.003681 + 0.003702`
  (four buys) — exact.
- **SOL `1fd2749865`:** snapshot `0.910206` = five recorded buys `0.829831` +
  `0.080375` (the foreign sell's qty) — exact.
- **Temporal proof:** every foreign-ID and NULL-decision order is dated
  **≤ 2026-10-09**; **zero** occur on/after **2026-10-13**, and **zero** on/after
  the `3384e91`/`6a91345` fixes.

---

## 7. Root-cause proposal (PROSE ONLY — no change made)

**Prospective (already fixed at HEAD):** nothing further required — `3384e91` and
`6a91345` key both entry and exit rows on the **open snapshot**. The one residual
hygiene gap is **cache coherence**: `save_decision_snapshot`'s fold path updates
the row but does **not** refresh `_open_snapshot_cache` (`db.py:818`), whereas
`close_decision_snapshot` and the update path **do** (`1582`, `1655`). Within a
cycle the stale `None` cache is harmless (the fallback is the committee id), but
it is worth an explicit invalidation for clarity. **Not proposed for change now.**

**Historical (do NOT repair):** the 29 affected orders (18 foreign + 11 NULL)
remain **unauditable**, for the reasons already established:
- Relinking alone yields `quantity_mismatch` (Item 3 §4a).
- A complete relink yields `unverified_fee_provenance` (Item 3 §4b) — the fees
  predate fee recording (first nonzero fee `2026-10-13 02:08`).
So even a perfect decision-ID reconstruction would buy **no** learner-eligible
rows. **Linkage confidence ↑; accounting confidence unchanged (0); learner
eligibility unchanged (0).**

## 8. Regression tests (for a future, separately-authorized change)

1. **Fold keying:** a second entry for an open symbol writes its ledger row with
   the **existing snapshot's** id, and the fresh committee id is never persisted.
2. **Exit keying:** close / trailing-stop / flatten exits always carry
   `_snapshot_decision_id`; never `None` when an open snapshot exists.
3. **SOL shape:** a folded add whose partial exit fires records buy+sell under the
   snapshot id, not the throwaway id.
4. **No-regression:** `get_recent_realized_pnl` and the learner inputs are
   unchanged by any of the above.
5. **Cache coherence (if fixed):** after a fold, `get_open_snapshot` returns the
   folded snapshot within the same cycle.

## 9. Historical consequences

- The pre-2026-10-13 window is **explainable but not certifiable**.
- The 11 `unreconciled` closes stay `unreconciled`; the Item 2 filter would still
  keep only 11/24 rows (no BTC, no `low_volatility`).
- **No** retroactive relink, backfill, or P&L edit is justified.

## 10. Unresolved questions

1. Why did a **scale-in** fire at all in these windows (stale-positions "new
   entry" vs genuine add)? The fold is the mechanism; the *trigger* isn't
   fully pinned from the DB alone (no logs retained).
2. Was `773ddfde98` a **partial** exit? No partial-exit feature exists, so likely
   a full exit of a partially-filled add — but the add's buy row is missing, so
   this can't be fully closed from the snapshot.
3. Should `_open_snapshot_cache` be invalidated on fold (hygiene, §7)?

## 11. Non-goals

- No source/test/config/DB change; no relinking/backfill/repair.
- No learner-behavior or risk-control change.
- No staging, commit, deploy, restart, or orders.

## 12. Verdict

**Root cause: CONFIRMED and already fixed prospectively.** All 18 foreign IDs and
11 NULL exits trace to **entry/exit ledger keying that ignored the snapshot fold**
(pre-`3384e91`/`6a91345`). The **SOL cross-lifecycle case shares this cause**.
**Linkage confidence is high; accounting confidence and learner eligibility are
unchanged (0)** — the affected records remain unauditable because their fees were
never recorded. **Recommendation: no repair.**
