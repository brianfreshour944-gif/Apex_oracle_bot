# COMMIT PLAN — W1 INSTRUMENTATION + ACCOUNTING + TEST ISOLATION

Date: 2026-10-25. **Planning only. Nothing committed, staged, deployed, or
restarted.** Base `7a8ca12`, 0 staged.

Purpose: give the reviewed work a coherent revision without touching `main`.
This document maps **every change to a scope** so a reviewer can tell exactly
what is being accepted, per the caution against bundling accounting changes into
one unquestioned approval.

---

## 1. What is uncommitted, and why it matters

HEAD is `7a8ca12`. Everything below exists **only in the working tree** — a
deploy from the repository today would contain none of it.

`git diff --stat` vs HEAD:

| File | +/− | Protected? |
| --- | --- | --- |
| `src/bot.py` | +131/−76 | — |
| `src/db.py` | +449/−0 | **yes** (schema) |
| `src/exchange.py` | +60/−1 | **yes** |
| `src/risk.py` | +52/−28 | **yes** |
| `src/config.py` | +25/−0 | — |
| `src/feature_drift_monitor.py` | +5/−4 | — |
| `src/performance_tracker.py` | +5/−4 | — |
| `tests/conftest.py` | +12/−0 | — |
| `tests/test_reconciliation_and_db_recovery.py` | +8/−1 | pre-existing |
| **new:** `tests/test_feature_drift_isolation.py` | +72 | — |
| **new:** `tests/test_performance_tracker_isolation.py` | — | — |
| **new:** `tests/test_submission_quote_capture.py` | — | — |
| **new:** `tests/test_analyze_deployed_trades.py` | — | — |
| **new:** `scripts/analyze_deployed_trades.py` | — | — |
| **new:** `reports/` (assessments, fixes, backup) | — | — |

## 2. Four distinct scopes (not three)

Your list named three. The diff contains a **fourth** independent concern:

| Scope | Name | Nature |
| --- | --- | --- |
| **A** | W1 submission-quote instrumentation | Measurement only — additive, never alters an order |
| **B** | Accounting reconstruction | **Behaviour-affecting** — changes realized P&L on close paths |
| **C** | MFE/MAE excursion capture | **Behaviour-affecting** — new MFE/MAE values; no exit-timing change |
| **D** | Test isolation (tracker + drift) | Test-only — no runtime behaviour change |

Scope **C** was not in the three you listed, yet it is real: every historical row
recorded MFE/MAE as `0.0` because the caller cleared the trailing peaks *before*
the recorder read them. That is a data-quality fix, distinct from B.

## 3. Scope → hunk map (the bundling problem, precisely)

The changes are **entangled at the file level**, so they cannot be split by file:

**`src/config.py`** — mixes A and D:
- `CAPTURE_SUBMISSION_QUOTE` field → **A**
- `PERFORMANCE_TRACKER_STATE_PATH`, `FEATURE_DRIFT_STATE_PATH` + exports → **D**
- `import os` → **D**

**`src/db.py`** — mixes A, B, D:
- **ONE** `additions` dict in `_ensure_additive_columns()` mixes:
  - `orders.mid_at_submit`, `orders.spread_at_submit`, `orders.quote_ts` → **A**
  - `orders.fee_source` → **A/B** (fee provenance, used by both)
  - `decision_snapshots.outcome_status` → **B**
- `reconstruct_position_outcome()` (db.py:474–715) → **B**
- `get_recent_realized_pnl()` docstring LIMITATION note → **B** (documentation only)

> **The migration dict is the sharpest entanglement:** the three W1 columns and
> the accounting column are added by a single dict literal. Git cannot stage
> them separately without a manual edit. This is the concrete reason a
> file-level split is impossible.

**`src/bot.py`** — mixes A, B, C:
- `_record_committee_outcome()` — reconstruction block → **B**; excursion
  peak/trough params + reads → **C**
- `_persist_order_record()` — `fee_source` → **A/B**; `mid_at_submit` /
  `spread_at_submit` / `quote_ts` → **A**
- `reconcile_open_snapshots()` → **B**
- `process_signal_for_symbol()` — capture-before-clear → **C**

**`src/exchange.py`** → **A** only (`get_latest_quote` + protocol method + capture
before submit).

**`src/risk.py`** → **B** only (`_close_open_snapshot` reconstruction + the two
flatten call sites).

**`src/performance_tracker.py`**, **`src/feature_drift_monitor.py`**,
**`tests/conftest.py`** → **D** only.

## 4. Options

**Option 1 — single branch, scoped commits (recommended).**
Create one branch; commit A, B, C, D as **four separate commits**, ordered so each
compiles and its tests pass. The shared migration dict is split by editing the
dict once so A's columns land in A's commit and B's column in B's commit. Gives
four reviewable diffs on one branch without four separate branches.

**Option 2 — single squashed commit.** Simplest, but collapses the accounting
change into the same revision as the measurement change. Against your stated
caution.

**Option 3 — four branches.** Maximum isolation, but B and C both touch
`_record_committee_outcome()` and cannot be applied independently without
conflicts. Not practical.

## 5. Recommended commit sequence (Option 1)

| # | Commit | Scope | Files / hunks |
| --- | --- | --- | --- |
| 1 | `test: isolate performance-tracker and feature-drift state files` | **D** | config (paths + `os`), performance_tracker.py, feature_drift_monitor.py, conftest.py, 2 new tests |
| 2 | `feat(exchange): capture submission NBBO quote before order` | **A** | exchange.py, config (`CAPTURE_SUBMISSION_QUOTE`), db (3 quote cols + `fee_source`), bot (`_persist_order_record`), new quote test |
| 3 | `fix(accounting): reconstruct position outcome from order ledger` | **B** | db (`reconstruct_position_outcome`, `outcome_status` col), bot (`_record_committee_outcome` recon block, `reconcile_open_snapshots`), risk (`_close_open_snapshot`) |
| 4 | `fix(accounting): record MFE/MAE from close-time excursions` | **C** | bot (excursion params, capture-before-clear) |

Order rationale: D first (independent, unblocks clean suite runs); then A
(measurement, low risk); then B (accounting — the change to scrutinise hardest);
then C (smallest, depends on B's recorder signature).

## 6. Pre-commit checklist (each must be shown, not asserted)

1. Full suite green on the branch tip: `pytest tests/` (expect 600 passed).
2. `uvx ruff check .` → `All checks passed!`
3. Real state files byte-identical across a suite run (tracker `44136fa3…`,
   drift `82a78c0d…`).
4. `git diff --check` clean (no whitespace errors).
5. The **exact staged diff** shown per commit before it is made.
6. Migration verified additive on a **copy** of `bot_snapshot.db`.

## 7. Explicitly out of scope

- Deployment, restart, service changes — separate authorization.
- Resolving `CAPTURE_SUBMISSION_QUOTE=True` — a **decision**, not a commit
  (see §8).
- `get_recent_realized_pnl` filtering on `outcome_status` — changes trading
  behaviour; needs its own review.
- Any change to `main`.

## 8. Open decisions the commit does NOT settle

1. **`CAPTURE_SUBMISSION_QUOTE` default.** Currently `True` — the capture path
   runs on **every** order by default (an extra API round-trip). Your item 2.
   Must be decided before deployment; the commit can carry either default.
2. **Database protection.** `_ensure_additive_columns()` mutates the deployed
   schema on first boot. Back up the deployed DB and verify the backup **before**
   the migration is allowed to run. Your item 3.
3. **Paper-only verification in the deployment environment** — confirm from the
   host's actual config, not the repo default.

## 9. Safety state

| Item | Value |
| --- | --- |
| Branch created | **No** |
| Commits made | **0** |
| Staged | **0** |
| Tracker / drift / snapshot | `44136fa3…` / `82a78c0d…` / `52e39180…` — unchanged |
| Deployment / restart / orders | **none** |
