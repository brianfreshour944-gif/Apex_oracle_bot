# ITEM 2 — Learner-label integrity: read-only impact analysis & proposal

Date: 2026-10-25. **READ-ONLY. No source, test, config, migration, or DB change;
nothing staged/committed/deployed/restarted; no orders.** This report ends the
task; implementation awaits separate authorization.

Scope: every learner / model-training path / classifier / adaptive component
that consumes stored trade outcomes or realized P&L. Risk-control consumers are
identified and **explicitly excluded** (§6).

Base: HEAD `7a8ca12` + the pre-existing accounting patch + the held item-1
`fee_source` change. Evidence: `bot_snapshot.db` sha256 `52e39180…`, opened
`mode=ro&immutable=1`.

---

## 1. Executive summary

Nine code paths consume stored outcomes. **None filters on `outcome_status`**
(verified: the only `outcome_status` references in `src/` are the *writers* in
`bot.py`/`risk.py`; no reader inspects it). The learner-label paths read
`realized_pnl`/`return_pct` directly, so a **sign-flipped** reconstructed value
(§ audit Finding D) teaches the opposite lesson, and 13/24 uncertified rows
train on unverified numbers.

**The decisive constraint:** on the deployed snapshot, restricting to
`outcome_status='reconciled'` would keep only **11 of 24 rows (46%)**, and it
**removes 10 of 10 `low_volatility` rows and 10 of 10 BTC rows entirely**. That
is not a neutral data-quality cleanup — it is a large, non-random selection
change that would push several learners *below their own bootstrap/validation
thresholds* and change which regimes they ever learn. This is the central
finding and the reason the proposal is deliberately conservative.

---

## 2. Consumer inventory (exact paths)

| # | Consumer | File:line | Fields used | Filters `outcome_status`? | Can consume NULL/unreconciled/legacy? |
| --- | --- | --- | --- | --- | --- |
| L1 | Adaptive meta-learner — live update | `bot.py:409` → `adaptive_meta.py:274` (`update`) | `net_pnl`, `return_pct`; label at `adaptive_meta.py:289` | **No** | **Yes** |
| L2 | Adaptive meta-learner — bootstrap replay | `bot.py:3606` → `adaptive_meta.py:366` (`rebuild_from_history`) | `net_pnl`, `return_pct` (`bot.py:3611-3621`) | **No** | **Yes** |
| L3 | Strategy learner + performance tracker | `bot.py:433` → `strategy_selector.py:191-226` | `pnl`, `return_pct` (`reward_score = return_pct/10`) | **No** | **Yes** |
| L4 | Transformer live replay buffer | `bot.py:295` | `realized_pnl` → binary label | **No** | **Yes** |
| L5 | Transformer online step | `bot.py:356` | `realized_pnl` → binary label | **No** | **Yes** |
| L6 | Decision-Transformer training data | `decision_transformer.py:991` (`prepare_training_data`), rtg at `:1050` | `realized_pnl / TARGET_RETURN_SCALE` | **No** | **Yes** |
| L7 | Decision-Transformer inference context | `decision_transformer.py:728` (`_get_recent_decisions`), rtg at `:786` | `realized_pnl` | **No** | **Yes** |
| L8 | PPO meta-learner training | `scripts/evolutionary_ppo_trainer.py:368` → `rl_env.py:110` | `realized_pnl` → reward | **No** | **Yes** |
| L9 | OOD discriminator retrain | `bot.py:3433` | `regime`, `features`, `brain_votes` — **not P&L** | No (n/a) | Yes (features only) |
| L10 | Causal attribution (LLM narrative) | `bot.py:445` → `attribution.py:27` | `realized_pnl`, `return_pct` | **No** | **Yes** |

**Note on L9:** the OOD discriminator consumes *features/votes*, not outcomes. It
is listed for completeness but is **not a label consumer** and should **not** be
filtered by `outcome_status` — doing so would reduce its history for no accuracy
benefit and delay its `OOD_BOOTSTRAP_MIN_HISTORY=100` gate.

**Note on L10:** attribution feeds an LLM narrative, not a numeric learner. It is
out of scope for label filtering; flag it only because it currently narrates
uncertified P&L as if factual.

---

## 3. Current behavior & failure modes

### L1/L2 — adaptive meta-learner (the primary concern)
`adaptive_meta.py:289`: `label = 1 if pnl > 0 else (0 if pnl < 0 else -1)`. The
sign of `pnl` is the entire lesson. `_profitable_direction()` (line 262) then
credits/debits each brain's vote against that sign.

**Failure mode:** for a sign-flipped row (audit Finding D: `f110556591` stored
`−0.0375` vs reconstructed `+0.1259`), the learner is told the *wrong* direction
won and adjusts brain weights toward the losing vote. This is the most direct
"teaches the wrong lesson" path.

### L2 — bootstrap replay compounds it
`rebuild_from_history()` replays **every** closed row oldest-first through the
same `update()`. It resets to cold start each boot, so the corrupted labels are
re-applied on every restart, not just once. Its own docstring says counts equal
"the number of closed snapshots actually in the DB" — i.e. unfiltered by design.

### L3 — strategy learner reward
`reward_score = max(-1, min(1, return_pct/10))`. A nonsense `return_pct`
(−804.76%, −70218.31%) saturates to **−1.0**, the strongest possible penalty,
for a trade whose *dollar* loss was ~$0.04. The strategy learner is punished
maximally for a near-flat trade. Same values feed `performance_tracker`, whose
Sharpe/win-rate windows (§6) can raise decay alerts.

### L4/L5 — transformer labels
Both derive `label = 1.0 if realized_pnl > 0 else 0.0`. A sign flip inverts the
label; a genuinely flat trade with a tiny fee becomes a "loss". These write to
`data/live_experiences.jsonl` and the online gradient step.

### L6/L7 — decision transformer returns-to-go
`rtn = realized_pnl / 100.0`. Outlier P&L distorts the returns-to-go signal the
transformer conditions on — both in training and at inference context.

### L8 — PPO reward
`rl_env.py:110`: `reward = ±abs(realized_pnl) * pos_size_mult`. Outliers scale the
reward directly; a sign-flipped row rewards the wrong action.

---

## 4. Reconciled ≠ audited (explicit)

A `"reconciled"` row means the P&L was **rebuilt from the order ledger and the
arithmetic balanced**, with a **nonzero but ESTIMATED** fee on each leg (the
adapter cannot observe Alpaca's fee; `fee_source` item 1 shows "exchange" is
currently unreachable). It is **not** a claim that fees, fills, or the historical
accounting were independently verified against exchange billing.

Therefore filtering to `"reconciled"` **raises the floor but does not make the
labels audited.** Any proposal must state this wherever it says "reconciled",
or it will overstate the guarantee — exactly the reviewer's caution.

---

## 5. Sample-size impact (measured on the real snapshot)

Simulating the classifier's own verdicts (11 reconciled / 13 unreconciled, from
the item-1 audit) against `bot_snapshot.db`:

| Group | All | Reconciled | Dropped |
| --- | ---: | ---: | ---: |
| regime `sideways` | 11 | 9 | 2 |
| regime `low_volatility` | 10 | **0** | **10** |
| regime `neutral` | 2 | 2 | 0 |
| regime `bull` | 1 | 0 | 1 |
| symbol BTC/USD | 10 | **0** | **10** |
| symbol SOL/USD | 8 | 7 | 1 |
| symbol ETH/USD | 6 | 4 | 2 |
| **total** | **24** | **11** | **13** |

- **Retention: 11/24 = 46%.**
- Label distribution: ALL = {win 8, loss 16}; KEPT = {win 2, loss 9}.

**Consequences:**

1. **`low_volatility` and BTC would be erased** from learner history — not
   because those trades were bad, but because their exits lost `decision_id`
   linkage (§audit B). The filter would encode a **data-plumbing accident** as a
   **strategy lesson**.
2. **Threshold collapse.** With `VALIDATION_MIN_TRADES=10`
   (`adaptive_meta.py:40`), `low_volatility` drops 10→0 and can never validate;
   `sideways` drops 11→9 and falls *below* the gate it currently meets. The
   learner would silently stop validating regimes it validates today.
3. **`OOD_BOOTSTRAP_MIN_HISTORY=100`** (`ood_discriminator.py:61`) is unaffected
   here (that consumer uses features, not labels — see L9), but any *future*
   outcome-based gate would feel the same shrinkage.

This is why a naive "filter everything to reconciled" is **not** a safe default.

---

## 6. Risk-control consumers — SEPARATE, not touched

These consume the same P&L but **change trading behavior** if altered. They are
listed to mark the boundary; **no change is proposed here.**

| Consumer | File:line | Why separate |
| --- | --- | --- |
| Rolling loss gate | `bot.py:2159` → `db.py:1670` `get_recent_realized_pnl` | Sums **every** closed row regardless of status (deliberate fail-safe). Filtering it changes whether entries are blocked. |
| Kelly size cap | `risk.py:957` → `performance_tracker.py:251` | Reads win-rate/payoff per regime to **cap position size**. |
| Sharpe/win-rate decay alerts | `performance_tracker.py:206-227` | Could trigger size tightening. |
| Brain-C validation gate | `adaptive_meta.py:453-474` | Gates whether learned weights go live. |

**Key point:** the learner filter (L1–L8) and the risk-control filter are
**different concerns**. The reviewer's boundary is explicit: excluding uncertain
records from *training* must not silently change the loss gate or sizing.
Because L1–L8 and §6 read from **different call sites**, a learner-only filter
*can* be implemented without touching risk — provided it is applied at the
learner call sites, **not** inside `get_closed_decision_snapshots()` (which the
risk paths also use, e.g. `bot.py:3433`). **Filtering the shared reader would be
the dangerous shortcut and is explicitly rejected.**

---

## 7. Legacy databases without `outcome_status`

The deployed snapshot has **no `outcome_status` column** (verified). Handling:

- `get_closed_decision_snapshots` returns `row.outcome_status`, which is
  `None`/absent for legacy rows. A filter must treat **`None` as "unknown",
  never as "reconciled"**.
- Therefore a naive `WHERE outcome_status='reconciled'` excludes **all** legacy
  rows — on the deployed DB that is 24/24, i.e. **the learner would get zero
  history** and the validation gate would never arm.
- Safe handling options (to be chosen in review, none implemented):
  - **(a) Exclude only the known-bad**: drop rows explicitly `'unreconciled'`;
    keep `'reconciled'` **and** `None`. Preserves legacy history; still removes
    the 13 uncertified rows *once the column is populated going forward*.
  - **(b) Allow-list with fallback**: keep `'reconciled'`; for `None`, keep only
    rows whose stored `return_pct` is within a sane band (e.g. |return_pct| ≤
    100) to drop the −804%/−70218% outliers without needing the column.
  - **(c) Exclude `None` too** (strictest) — rejected as a default because it
    zeroes history on the current DB.

Option **(a)** is the recommended starting point: it is the smallest change that
stops the *known* wrong labels without discarding legacy rows wholesale.

---

## 8. Minimal proposed diff (PROSE / PSEUDODIFF ONLY — not applied)

Principle: **filter at the learner call sites, not in the shared reader.**

```
# src/db.py — NEW helper (does NOT change existing readers)
+ def is_learner_usable(row: dict) -> bool:
+     """True if a closed snapshot may train a learner. 'unreconciled' is
+     excluded; NULL/legacy is kept (option (a)). Reconciled is NOT audited."""
+     status = row.get("outcome_status")
+     return status != "unreconciled"      # keep 'reconciled' AND None
```

```
# src/bot.py L2 — bootstrap replay (~3606)
- snaps = get_closed_decision_snapshots(limit=10000)
+ snaps = [s for s in get_closed_decision_snapshots(limit=10000)
+          if db.is_learner_usable(s)]
```

```
# src/committee/decision_transformer.py L6/L7 — prepare_training_data / context
- decisions = get_closed_decision_snapshots(limit=limit)
+ decisions = [d for d in get_closed_decision_snapshots(limit=limit)
+              if is_learner_usable(d)]
```

```
# scripts/evolutionary_ppo_trainer.py L8 (~368)
- live_snapshots = get_closed_decision_snapshots(limit=2000)
+ live_snapshots = [s for s in get_closed_decision_snapshots(limit=2000)
+                   if is_learner_usable(s)]
```

```
# src/bot.py L1/L4/L5 — these build the outcome dict from the LIVE recon
# result, which already carries outcome_status. Gate the learner.update /
# buffer-append on it:
- report = learner.update(...)
+ if outcome_status != "unreconciled":
+     report = learner.update(...)
```

**Explicitly NOT changed:** `get_closed_decision_snapshots()` itself;
`get_recent_realized_pnl()`; `risk.py`; `performance_tracker` windows;
`adaptive_meta.py` internals.

**Open design question (needs review):** for L1/L4/L5 the label is computed
inline from the just-closed trade. Excluding an `unreconciled` live trade is
slightly different from excluding a historical row — the live trade still
happened, and skipping it just means that one trade doesn't train. Confirm
whether live `unreconciled` outcomes should be skipped too, or only historical.

---

## 9. Test plan (to accompany a future implementation)

**Focused**
1. `is_learner_usable`: `reconciled`→True, `None`→True, `unreconciled`→False.
2. Bootstrap replay with a mixed set excludes `unreconciled`, keeps `None`.
3. `prepare_training_data` / PPO loader skip `unreconciled`.
4. Live path: an `unreconciled` close does **not** call `learner.update`.
5. **No-risk-change guard:** `get_recent_realized_pnl` returns the **same**
   value with and without the filter helper in play.
6. **Legacy guard:** a DB with no `outcome_status` column yields the **full**
   history under option (a) (no accidental zeroing).

**Regression**
- Existing `test_adaptive_bootstrap.py`, `test_adaptive_meta.py`,
  `test_strategy_selector_seeding.py`, `test_rl_meta.py`, `test_ood_runtime.py`
  must stay green (they exercise the consumers being touched).

**Edge cases**
- All rows `unreconciled` → learner gets empty history → must not crash, must
  not reset an already-learned state file (mirror the existing fail-safe at
  `bot.py:3595`).
- A sign-flipped row (`f110556591`) is excluded and therefore cannot invert a
  brain's weight.
- `return_pct` outlier excluded by option (b) if chosen.

---

## 10. Potential effects on sample size & learner behavior

- **Retention 46%** on the current DB; **`low_volatility` and BTC lose all
  history**.
- `sideways` falls **below `VALIDATION_MIN_TRADES=10`**; `low_volatility` → 0.
  **Learners would validate fewer regimes**, changing which brains get adaptive
  weight — a behavioral change, even though it is not a risk-control change.
- Because the dropped rows correlate with a *linkage bug* (BTC exits), the
  filter would **bias the learner by symbol and regime**, not at random.
- **Mitigation to discuss:** re-linking the 11 orphaned exits (item 3/4) would
  convert many `unreconciled` rows to `reconciled` **before** any filter is
  enabled — recovering history *and* correctness together. **Recommended
  sequencing: fix linkage first, then filter.**

---

## 11. Unresolved questions

1. Live vs historical: should `unreconciled` **live** outcomes be skipped (§8)?
2. Option (a) vs (b) for legacy `None` rows — keep-all or sane-band fallback?
3. Is a 46% retention acceptable, or should the filter be **deferred until
   linkage is repaired** (my recommendation)?
4. Should `performance_tracker` (L3's second consumer, and a §6 sizing input)
   be filtered at all, given it is also a risk input? — **out of scope here.**
5. Do we want the learner to *record* that it skipped rows (observability), so a
   silently shrinking sample is visible?

## 12. Non-goals (explicit)

- No change to the rolling loss gate, entry blocking, risk limits, position
  management, or execution.
- No backfill of `outcome_status`/`fee_source`; no order relinking; no
  historical P&L edits.
- No change to `get_closed_decision_snapshots()`'s shared contract.
- No implementation, staging, commit, deploy, restart, or orders.

## 13. Recommendation

**Do not implement a learner filter yet.** Two findings make it premature:
(a) a naive filter erases 10/10 BTC and 10/10 `low_volatility` rows and pushes
`sideways` below its validation threshold; (b) those losses stem from a **linkage
bug**, not from bad strategy. The higher-value, lower-risk order is: **repair the
11 orphaned exits (item 3/4) → reclassify → then filter**, which yields *both*
more correct labels *and* preserved history. If a filter is still wanted before
then, option (a) applied **only at learner call sites** is the minimal safe step.
