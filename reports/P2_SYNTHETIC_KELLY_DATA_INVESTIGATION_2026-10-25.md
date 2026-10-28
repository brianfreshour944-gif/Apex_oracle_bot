# P2 — SYNTHETIC KELLY-SIZING DATA: READ-ONLY INVESTIGATION

Date: 2026-10-25. **READ-ONLY investigation.** Authorized scope: determine whether
the 48 synthetic trades in `data/performance_tracker_state.json` enter the running
sizing path. No code change, no repair, no deletion.

Base: HEAD `7a8ca12` (branch `main`), 0 staged.

> **Note on process (disclosed):** reproducing the write required actually running
> the offending test. That run incremented the real state file 48 → 49. I restored
> it byte-identically (`ddb2e987…`, 48 trades) and re-verified. The restore is
> shown in §7. No other file was touched.

---

## VERDICT

**The synthetic data is a TEST-FIXTURE LEAK, not a fabrication in production —
but the leak is REAL and it writes to the production state file.**

- The 48 rows are written by `tests/test_strategy_selector_seeding.py:43` —
  a **test**, not the bot.
- `tests/conftest.py` isolates the DB, adaptive state, live buffer, and model
  store — **but NOT `performance_tracker_state.json`**. So every test run appends
  to the real file.
- **REPRODUCED:** running that one test file incremented the **real** file
  48 → 49.
- **Kelly is NOT currently affected:** `KELLY_SIZING_ENABLED=False`, and even if
  enabled, `get_regime_win_payoff` returns **`None`** on this data (the
  `payoff_ratio = inf` finiteness gate). So no position is mis-sized **today**.
- **But there is a live secondary effect:** the fake 48-trade, 100%-win-rate
  `trend_following|bull` block **satisfies `MIN_TRADES_FOR_STATS`** and enters the
  decay monitor, **suppressing a win-rate decay alert** and polluting
  `/metrics` and weekly reports.

Severity: **MEDIUM** — real production-state contamination with a confirmed
behavioral effect (alert suppression), but no mis-sizing while Kelly stays off.
It is a **safety-adjacent defect**, exactly as you suspected, though its current
blast radius is smaller than "fabricated trades set position size."

---

## 1. What the file is and where it lives

| Property | Value |
| --- | --- |
| Path | `data/performance_tracker_state.json` |
| Constant | `_STATE_PATH = os.path.join("data", "performance_tracker_state.json")` (`performance_tracker.py:21`) |
| Configurable? | **No** — a hardcoded module constant. Contrast `ADAPTIVE_STATE_PATH`, which *is* a settings field. |
| Tracked by git? | **No** — untracked and gitignored (`.gitignore:62` `data/`) |
| Contents | One key: `trend_following|bull`, **48 trades**, all `pnl=5.0`, `return_pct=2.0` |

The uniform values are the tell: 48 identical `pnl=5.0/ret=2.0` rows are not
market outcomes.

## 2. Where the 48 rows come from — CONFIRMED

`tests/test_strategy_selector_seeding.py:43`:

```python
sel.record_strategy_outcome("bull", "trend_following", "buy", pnl=5.0, return_pct=2.0)
```

This is the **only** generator of `pnl=5.0/return_pct=2.0` anywhere in the repo
(verified by grep). The chain:

```
test → sel.record_strategy_outcome(...)
     → performance_tracker.record_trade_outcome(...)
     → get_performance_tracker()          # process-wide singleton, default path
     → PerformanceTracker(state_path=_STATE_PATH)   # REAL file
     → record_trade(...) → _save_state()  # writes data/performance_tracker_state.json
```

`record_trade_outcome` (`performance_tracker.py:339-340`) uses
`get_performance_tracker()`, whose singleton is built with the **default**
`_STATE_PATH`. The test does not redirect it.

## 3. Why the test writes the real file — the isolation gap

`tests/conftest.py` redirects **four** paths to a temp dir:

```python
os.environ["DATABASE_URL"]        = .../bot.db
os.environ["ADAPTIVE_STATE_PATH"] = .../adaptive_meta_state.json
os.environ["APEX_LIVE_BUFFER_PATH"] = .../live_experiences.jsonl
os.environ["APEX_MODEL_STORE_DIR"]  = .../models
```

`performance_tracker_state.json` is **not among them**, and it cannot be: it is a
hardcoded constant with no env/settings override (§1). The conftest docstring
itself records a *prior* instance of this same class of bug — a test that "wrote
one fake losing … trade into the real DB and fed it to the adaptive learner on
every run." That fix closed the DB and adaptive paths; the tracker path was
**missed**.

## 4. REPRODUCED — the leak is real

I ran the offending test in isolation, with a pre/post snapshot:

```
before:  48 trades
$ .venv/bin/python -m pytest tests/test_strategy_selector_seeding.py -q
9 passed
after:   49 trades          <-- the REAL production file was written
```

Full-suite run: `578 passed`, real file 48 → **49** (exactly one appended row —
only this test file records an outcome). Both runs were followed by a
byte-identical restore (§7).

**The mechanism is confirmed end-to-end, not inferred.** I also reproduced it
directly against a `/tmp` copy (`record_trade` appended one row), which is the
safe form of the same proof.

## 5. Does it reach the Kelly sizing path? — NO, today (three independent gates)

Even though the file is read by the Kelly path, **no mis-sizing occurs**, for
three reasons that stack:

1. **`KELLY_SIZING_ENABLED = False`** (config default; no `.env` override).
   The call site is guarded: `if getattr(settings, "KELLY_SIZING_ENABLED", False):`
   (`risk.py:957`). With the flag off, `get_kelly_size_cap` is **never called**.
2. **`KELLY_MIN_TRADES = 30`.** Even if enabled, `get_kelly_size_cap` returns
   `None` unless `stats["n"] >= 30` (`risk.py:1556`). The fake block has 48 ≥ 30,
   so this gate alone would **not** block it.
3. **The finiteness gate — which DOES block it.** `get_regime_win_payoff`
   (`performance_tracker.py:270-271`) returns `None` when
   `payoff_ratio <= 0 or not np.isfinite(payoff_ratio)`. The fake block has **no
   losing trades**, so `payoff_ratio = inf` → **`None`**.

**REPRODUCED:** `get_regime_win_payoff('bull', 30) == None` and `== None` for 90d.

So the fake data **cannot** set a Kelly cap — not because the data is clean, but
because an all-wins fixture happens to trip a degenerate-statistic guard. That is
**luck, not design**: a fixture with even one loss would produce a finite
`payoff_ratio` and, with Kelly enabled, would produce a cap.

## 6. The REAL effect today — decay-alert suppression

The tracker feeds more than Kelly. `get_performance_report`
(`performance_tracker.py:165-227`) computes rolling windows for every pair with
`len(trades) >= MIN_TRADES_FOR_STATS` (10). The fake 48-trade block clears that.

**REPRODUCED report:**

```
trend_following|bull: windows 30d/90d/180d all present, n=48,
                      win_rate=1.00, sharpe=0.00, profit_factor=inf
alerts: [sharpe_decay]   <-- fires (sharpe 0.0 < 0.3)
```

- **win-rate decay alert: SUPPRESSED.** The 30d win-rate is 1.00, far above
  `WIN_RATE_DECAY_THRESHOLD = 0.48`, so no `win_rate_decay` alert fires — even
  though the block is fabricated. A real degradation in `trend_following|bull`
  would have to overcome 48 fake wins before the alert would sound.
- **Consumers:** `get_decay_alerts()` → `alerting.py:496-497` (operator alerts),
  and `performance_server.py` exposes the full report at `/metrics` and
  `/performance`.

**This is the concrete harm:** a monitoring signal that exists to warn of decay
is blinded for that (strategy, regime) pair by fabricated data. It does not
change position size, but it degrades a safety alert — which is why this belongs
ahead of the trend experiment, as you placed it.

## 7. Restore (disclosed, verified)

Running the test to prove the leak modified the file. I restored it:

```
pre-test backup  /tmp/pt_before.json   sha256 ddb2e98765a566b2…
after restore    data/performance_tracker_state.json  sha256 ddb2e98765a566b2…
diff: IDENTICAL;  trades: 48
```

Final state: real file at its original 48 trades and original hash; 0 staged; no
src/tests change; snapshot `52e39180…` unchanged; no stray temp files in `data/`.

## 8. Findings

| ID | Severity | Claim | Evidence |
| --- | --- | --- | --- |
| P2-1 | **MEDIUM** | A test writes the REAL production tracker state file every run | REPRODUCED: 48→49 on one test file and on the full suite; `test_strategy_selector_seeding.py:43` |
| P2-2 | **MEDIUM** | Fabricated data suppresses the win-rate decay alert for `trend_following|bull` | REPRODUCED: 30d win_rate=1.00 ≥ 0.48 → no alert; consumers `alerting.py:496`, `performance_server.py` |
| P2-3 | LOW | `conftest.py` does not isolate the tracker path (unlike DB/adaptive/buffer/model) | `conftest.py:41-44`; `_STATE_PATH` hardcoded, no env override |
| P2-4 | LOW | Kelly is **not** affected today, but only via a degenerate-statistic guard | `KELLY_SIZING_ENABLED=False`; `payoff_ratio=inf` → `None` (`performance_tracker.py:270`) |
| P2-5 | LOW | The guard is luck, not design: a fixture with one loss would yield a finite payoff ratio | reasoning from `_compute_extended_stats`; not separately reproduced |

## 9. Direct answers to the four questions you posed

1. **Test fixture or running path?** Both — it originates as a **test fixture**
   (§2) but **enters the production file** because of the isolation gap (§3),
   and is **read by the running path** (§5, §6).
2. **When is the file loaded?** At `PerformanceTracker.__init__` →
   `_load_state()` (`performance_tracker.py:99-102`), constructed once per
   process by the `get_performance_tracker()` singleton (`:323-327`).
3. **Is it overwritten?** Not wholesale — `record_trade` **appends** and rewrites
   the whole JSON (`_save_state`, `:115-121`), capped at the last 500 trades per
   pair (`:151-153`). Real trades **coexist** with the fake ones; they do not
   replace them.
4. **Does real data replace it?** **No.** Real closed trades append alongside the
   48 fake rows. The fake rows persist until (a) the 500-cap eventually evicts
   them by age, or (b) the file is deleted. Nothing purges them today.

## 10. Unresolved questions

1. **Was the file ever written by a real bot run?** All 48 rows are uniform
   fixture values and dated 2026-10-24..26, matching test-run times — consistent
   with **test-only** origin, but I cannot prove no real trade was ever recorded
   without the bot's logs.
2. **Should the fix isolate the path, or redirect it?** `_STATE_PATH` is a
   hardcoded constant; the clean fix is likely a settings field like
   `ADAPTIVE_STATE_PATH` plus a `conftest` override. **Not proposed for change
   here** — this is read-only.
3. **Does the 500-cap create a real-data eviction risk?** If tests keep appending,
   they consume cap slots; over many runs, fake rows could push real rows out.
   Bounded by the cap, but worth noting.

## 11. Recommendation (no implementation authorized)

1. **Treat this as a real defect** — production state is being written by the test
   suite. It is MEDIUM, not CRITICAL, because Kelly is off and blocked by a guard.
2. **Do not "fix" by deleting the file.** That removes the evidence and does not
   close the leak; the next test run recreates it.
3. **The correct fix is isolation** (a settings-overridable path + a `conftest`
   redirect), mirroring the existing `ADAPTIVE_STATE_PATH` pattern — for a
   **separately authorized** change, with a regression test asserting the test
   suite does not touch the real file.
4. **Keep `KELLY_SIZING_ENABLED=False`** until the data is clean. Note that
   enabling Kelly would not currently mis-size (the `inf` guard blocks it), but
   the guard is incidental, not a control (§5).
5. **Proceed to P1 (measured paper costs)** next, per your ordering.

## 12. Safety boundary

- 0 staged; no source/test/config change; snapshot `52e39180…` byte-identical.
- Real tracker state restored to its original hash `ddb2e987…` (48 trades).
- No repair, deletion, or code change performed.
- Nothing committed, deployed, restarted, or ordered.
- Only new artifact: this report.
