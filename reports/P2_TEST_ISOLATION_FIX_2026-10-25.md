# P2 — TEST-STATE CONTAMINATION FIX (IMPLEMENTED, NOT DEPLOYED)

Date: 2026-10-25. Authorized scope: **P2 test-isolation fix only**. No
deployment, no orders, no commit. Base `7a8ca12`, 0 staged.

---

## 1. Root cause

`src/performance_tracker.py` hardcoded its state path as a module constant:

```python
_STATE_PATH = os.path.join("data", "performance_tracker_state.json")   # relative!
```

`PerformanceTracker.__init__(state_path=_STATE_PATH)` defaulted to it, and the
process-wide singleton `get_performance_tracker()` used the default. So any test
calling `record_strategy_outcome()` wrote through the singleton into the **real
runtime file**.

`tests/conftest.py` already isolated `DATABASE_URL` and `ADAPTIVE_STATE_PATH`,
but **not** the tracker path — an omission, since the tracker had no
configurable path to isolate. `tests/test_strategy_selector_seeding.py:43`
records a fixture outcome (`pnl=5.0, return_pct=2.0`), so **every full-suite run
appended one synthetic trade** to the real file.

## 2. The fix (follows the existing ADAPTIVE_STATE_PATH precedent)

| File | Change |
| --- | --- |
| `src/config.py` | New `PERFORMANCE_TRACKER_STATE_PATH` field (default `data/performance_tracker_state.json`) + module export; `import os` |
| `src/performance_tracker.py` | Default resolves from `settings.PERFORMANCE_TRACKER_STATE_PATH` **at construction time**; dead `_STATE_PATH` constant removed |
| `tests/conftest.py` | Redirects `PERFORMANCE_TRACKER_STATE_PATH` to the temp dir |
| `tests/test_performance_tracker_isolation.py` | 4 regression tests (new) |

Two design points:

- **Resolved at construction, not import time.** The original failure mode was
  partly an import-time constant. Reading settings in `__init__` means an
  env-var override is always honoured regardless of import order.
- **Backwards compatible.** `state_path` is still an explicit override
  (`PerformanceTracker(state_path=...)`); only the default changed.

`_STATE_PATH` was removed rather than left in place: after the change it was
dead, and a stale constant naming a real file is exactly the trap this fix
closes.

## 3. Verification

| Check | Result |
| --- | --- |
| Isolation regression tests | **4 passed** |
| Full suite | **596 passed** (592 + 4 new) |
| Ruff | **All checks passed!** |
| **Real tracker file after a full suite run** | `ddb2e98765a566b2` — **byte-identical, 48 trades** |
| Real tracker after a **second** full run | `ddb2e98765a566b2` — unchanged |
| Production default (no conftest env) | `data/performance_tracker_state.json` — **unchanged** |
| Snapshot | `52e39180…` untouched |
| Staged | **0** |

The decisive check is the fourth row. Before the fix, each full-suite run
appended one synthetic trade (48 → 49 → 50 → 51). After the fix, the file is
**byte-identical across repeated runs**. The isolation is proven by the real
file not changing, not merely by a green test.

## 4. Scope boundary — what this fix does NOT do

- **It does not clean the existing file.** The 48 synthetic records already in
  `data/performance_tracker_state.json` are **still there**. Removing or
  rebuilding that state is a **separate** decision, as you specified — I did not
  touch it beyond restoring the three entries I had added during W1.
- **It does not fix the alert consequence.** The synthetic data still holds
  win-rate/profit-factor above threshold and fires a spurious sharpe alert (see
  the W1 validation addendum). Isolating the *writer* stops future corruption;
  it does not repair the *current* state.
- **It does not touch fee provenance or W1.** Those are unchanged.

## 5. Limitations

- The isolation guarantee is enforced by `conftest.py`, which applies to the
  pytest suite. A test run outside pytest (a bare script) would still use the
  real default — that is intended, since the default must stay correct for
  production.
- The regression test asserts the real file is unchanged *within a test run*;
  the stronger evidence is the before/after hash across full-suite runs,
  recorded above.

## 6. Safety state

| Item | Value |
| --- | --- |
| Staged | **0** |
| Snapshot | `52e39180d5d9e9e1` — untouched |
| Tracker state | `ddb2e98765a566b2` — 48 trades, unchanged |
| Orders / deployment / restart / commit | **none** |

## 7. Recommended next steps (not authorized)

1. Separately decide the tracker's recovery — whether to reset or rebuild the 48
   contaminated records (needs its own authorization; do not delete until then).
2. Then consider W1 deployment for paper-only measurement.
3. Accumulate usable fills; assess real execution costs.
4. Only then W2/P3.
