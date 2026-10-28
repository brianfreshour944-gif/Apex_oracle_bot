# UNTRACKED FILE INVENTORY & COMMIT D SCOPE — READ-ONLY

Date: 2026-10-25. **Read-only. No file changed, nothing staged.** Resolves the
scope ambiguity before Commit D, per the recommended order.

---

## 1. Full untracked inventory

Beyond `reports/` (documentation), there are **8 Python files** untracked, not 7:

| File | Lines | Serves scope | Classification |
| --- | --- | --- | --- |
| `tests/test_performance_tracker_isolation.py` | 69 | **D** | Required regression test |
| `tests/test_feature_drift_isolation.py` | 63 | **D** | Required regression test |
| `tests/test_submission_quote_capture.py` | 329 | **A** | Required regression test |
| `tests/test_position_outcome_reconstruction.py` | 768 | **B + C** | Required regression test |
| `tests/test_fee_provenance.py` | 196 | **A/B** | Required regression test |
| `scripts/analyze_deployed_trades.py` | — | — | Research harness |
| `tests/test_analyze_deployed_trades.py` | 542 | — | Research-harness test |
| `scripts/edge_study.py` | — | — | Research harness |
| `tests/test_edge_study.py` | 55 | — | Research-harness test |
| `scripts/fetch_ohlcv.py` | — | — | Research harness |
| `scripts/momentum72_study.py` | — | — | Research harness (FROZEN) |
| `tests/test_momentum72_study.py` | 274 | — | Research-harness test |

## 2. The four buckets

**Bucket 1 — Required by a commit scope (5 files):**
`test_performance_tracker_isolation.py` (D),
`test_feature_drift_isolation.py` (D),
`test_submission_quote_capture.py` (A),
`test_position_outcome_reconstruction.py` (**B and C**),
`test_fee_provenance.py` (A/B).

**Bucket 2 — Research harness (6 files):** `analyze_deployed_trades.py` +
its test, `edge_study.py` + its test, `fetch_ohlcv.py`,
`momentum72_study.py` + its test.

**Bucket 3 — Documentation (whole `reports/` dir, ~35 files + backup + JSON).**

**Bucket 4 — Nothing unrelated found.** Every untracked file traces to either a
commit scope or the research line.

## 3. NEW ENTANGLEMENT — B and C share one test file

`tests/test_position_outcome_reconstruction.py` contains **both**:

- `test_mfe_mae_recorded_from_caller_supplied_excursions` (line 703) — **Scope C**
- `test_recorder_clear_timing_matches_head` (line 730) — **Scope C**
- the reconstruction tests around them — **Scope B**

So the B/C entanglement is **not only** in `bot.py`; it is also in the **test
file**. Commits B and C cannot each own this file whole. Options:

- **(i)** Split the file: B's tests in Commit B, C's excursion tests in Commit C.
- **(ii)** Put the whole file in Commit B and let C add only its `bot.py` change
  (but then C's test lands before C's code).
- **(iii)** Merge B and C into one commit — **rejected**; they fix different
  defects and you explicitly want them separate.

Recommend **(i)**: split the test file so each commit carries its own tests. This
is the same class of manual split as the shared migration dict.

## 4. `reports/backup/` — handle deliberately

Contains `performance_tracker_state.ddb2e987….json` (5816 bytes) — the **verified
backup** of the contaminated tracker, the only copy outside `data/` (which is
gitignored). If it is committed it becomes durable; if not, it lives only here.
**Recommendation:** do not bundle it into a commit silently. Decide explicitly
whether the backup is preserved as a tracked artifact or kept as a local file.
Not a code change either way.

## 5. Commit D — exact file list (proposed)

Only the tracker + drift test-isolation scope and what it directly requires:

| Action | File |
| --- | --- |
| stage | `src/config.py` — **only** the two `*_STATE_PATH` fields + `import os` + exports |
| stage | `src/performance_tracker.py` |
| stage | `src/feature_drift_monitor.py` |
| stage | `tests/conftest.py` |
| stage | `tests/test_performance_tracker_isolation.py` |
| stage | `tests/test_feature_drift_isolation.py` |

**Excluded from D:** `CAPTURE_SUBMISSION_QUOTE` (A), all quote/accounting columns
and functions (A/B), `bot.py`, `risk.py`, `exchange.py`, and every research file.

> **Note:** `src/config.py` is shared between D and A (the `CAPTURE_SUBMISSION_QUOTE`
> field). Commit D must stage **only the D hunks** of `config.py` — a partial-file
> stage (`git add -p`), not the whole file. This is the config-level twin of the
> migration-dict split.

## 6. Research files — recommendation

Leave Bucket 2 **untracked for now**. They are not part of any of the four
scopes; they belong to the research line you have paused ("hasn't demonstrated a
reliable positive net edge"). Committing them would mix paused research into the
implementation history — exactly what you cautioned against. Preserve them by
other means (they are already on disk; a future, explicitly-scoped research
branch can adopt them).

## 7. What this changes in the commit plan

The plan now has **three** partial-file splits, not one:

1. `src/db.py` — one migration dict mixing A and B columns.
2. `src/bot.py` — one function mixing B and C concerns.
3. `tests/test_position_outcome_reconstruction.py` — one file mixing B and C tests.
4. (plus) `src/config.py` — D and A hunks in one file.

Each needs a deliberate `git add -p`-style split, shown in the staged diff before
its commit.

## Safety state

No branch change beyond the existing `w1-instrumentation-accounting`; **0
commits**, **0 staged**, no file modified. Tracker `44136fa3…`, drift
`82a78c0d…`, snapshot `52e39180…` unchanged.
