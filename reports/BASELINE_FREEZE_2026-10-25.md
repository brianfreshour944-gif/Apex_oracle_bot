# STEP 1 — BASELINE FREEZE / VERIFICATION (read-only)

Date: 2026-10-25. **READ-ONLY. No code change, no staging, no commit, no deploy.**
This document only records and verifies the current state so later work can be
diffed against a fixed reference.

---

## 1. Repository identity

| Item | Value |
| --- | --- |
| HEAD | `7a8ca12d1b4cac00372053e59f97440ff2d48f13` |
| Branch | `main` |
| Staged files | **0** |

## 2. Working-tree state (tracked, unstaged)

| File | sha256 (full) |
| --- | --- |
| `src/db.py` | `9353575ed170e871f4acf69ff86e3346649cb1b4eab2bd00c5eebcadced970a6` |
| `src/bot.py` | `972e82cb681f177b7edf195e313f27de3a500fdca8300644374726977b4d087e` |
| `src/risk.py` | `3733d1d6a64e8ba6a861ab3766e60d123ff0352f50f2370351e572b46164db4e` |
| `tests/test_reconciliation_and_db_recovery.py` | `f2466cd701e43db1b3b72ccaeef728121d164f4384b16c53dfa6bb192d8bf6ae` |

These four files carry the **pre-existing accounting patch** plus the **held
item-1 `fee_source`** addition. **`src/` changes are limited to `db.py`, `bot.py`,
`risk.py`** — no other source file differs from HEAD.

## 3. Untracked (non-ignored) artifacts

```
scripts/analyze_deployed_trades.py   tests/test_analyze_deployed_trades.py
scripts/edge_study.py                tests/test_edge_study.py
scripts/fetch_ohlcv.py
scripts/momentum72_study.py          tests/test_momentum72_study.py
tests/test_fee_provenance.py         tests/test_position_outcome_reconstruction.py
reports/
```

`reports/` is gitignored but present on disk (all investigation reports).

## 4. Momentum72 artifacts (preserved)

| File | sha256 (first 16) |
| --- | --- |
| `scripts/momentum72_study.py` | `24f39a9d7da70a02` |
| `tests/test_momentum72_study.py` | `5dec4b927e8f283e` |

## 5. Database snapshot (immutable reference)

| Item | Value |
| --- | --- |
| Path | `/workspace/project/bot_snapshot.db` |
| Size | 2,105,344 bytes |
| sha256 | `52e39180d5d9e9e10eb0d689277cae0cb0c3c480843d321a0732886223cfca1b` |

**Unchanged from every prior investigation** (`52e39180…` throughout Items 2–4).
No relinking, backfill, or repair has been applied.

## 6. Accounting-patch integrity check

The recorded patch (`reports/UNCOMMITTED_PATCH_2026-10-25.md`, rev 13,
sha256 `349f095f9b51fba6…`) was compared against the live working-tree diff:

- The recorded patch's **main block accounts for all current changes except 24
  added lines**.
- Those **24 lines are exactly the `fee_source` additions** (model column,
  `save_order_record` kwarg, `_ensure_additive_columns` entry, and the
  `_persist_order_record` derivation).

**Conclusion:** the working tree = **recorded accounting patch + item-1
`fee_source`**, and nothing else. The patch's own hunks are intact.

## 7. Verification performed (read-only)

- `git rev-parse HEAD` / branch — §1
- `git diff --cached` → 0 staged — §1
- `git diff --name-only` (tracked) — §2
- `git status --porcelain` (untracked) — §3
- `sha256sum` on all four patched files + Momentum72 artifacts + snapshot — §2/§4/§5
- Working-tree diff vs recorded patch, isolating the `fee_source` delta — §6

## 8. What this freeze does and does not do

- **Does:** fixes a reference (HEAD + checksums + snapshot hash) so any future
  change is provably attributable.
- **Does not:** change any file, stage anything, commit, deploy, restart, or place
  an order. No learner or risk-control behavior is touched.

## 9. Standing constraints (unchanged)

- Historical records: **untouched** (repair rejected).
- Item 1 `fee_source`: **held, not deployed**.
- Items 2–5: **no implementation authorization**.
- Learner and risk controls: **unchanged**.
- No staging, commit, deployment, restart, or orders.

---

## 10. Status

**Baseline frozen and verified.** Step 1 complete. Step 2 (learner-eligibility
design proposal) is **not started** — it requires explicit authorization.
