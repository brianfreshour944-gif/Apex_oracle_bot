# MOMENTUM72 — PREREQUISITE FAILURE: harness does not implement the frozen spec

Date: 2026-10-25
Author: OpenHands. **No validation data was fetched. The strategy was NOT run on
2021-01-01..2025-09-30.** This report records a pre-execution conformance failure
and stops, per instruction 4 ("if any essential parameter is unspecified or
inconsistent, stop and report it rather than choosing a value").

Base commit: 7a8ca12 (branch main), 0 staged. Accounting patch sha256
`d1131e48…` unchanged. Protected paths byte-identical to HEAD. Nothing staged,
committed, deployed, or restarted. No file modified except this report.

---

## 1. Prerequisites checked (all PASS except the harness)

| Prerequisite | Result |
| --- | --- |
| `MOMENTUM72_SPEC_2026-10-25.md` sha256 | **PASS** — `08316c38…a21af854` matches |
| Git HEAD / branch | `7a8ca12d1b4cac00372053e59f97440ff2d48f13` / `main` |
| Staged files | 0 |
| Accounting patch sha256 | `d1131e48b4c58e4f21db4f30aeca196c116b2a7cff2e37ac5cd8907cdfbfd4a9` (unchanged) |
| Protected paths (`exchange.py`, `strategies.py`, `committee/`, `docker-compose.yml`) | UNCHANGED |
| Spec parameters explicit & internally consistent | **PASS** (see §2) |
| Harness implements the spec | **FAIL** (see §3) |

## 2. The frozen spec IS explicit and consistent (no ambiguity to resolve)

Every essential parameter is stated: momentum `log(c_i/c_{i-72})`, vol = sample
stdev (ddof=1) of the 72 log returns ending at *i*, `w_i = min(1.0,
0.005497/sd_i)` when long else 0, exposure cap 1.0, cost `|Δw|×28bps`,
turnover-based, 56bps round trip. Success criteria and verdict rule are explicit.
**Return accounting is stated unambiguously in §3:**

> strategy return for bar *i+1* = `w_i * logret_{i+1} − |w_i − w_{i-1}| * COST_ONE_WAY`

with execution "at the close of bar *i* (the first instant the information is
available)", and a separately reported *sensitivity* "to execution delayed by one
bar". So the primary convention and the delayed variant are both defined. There
is no gap here for the implementer to fill by judgement.

## 3. The harness does NOT match the spec (stop condition)

`scripts/momentum72_study.py` `run()` (lines 90–101) computes, for bar *i*:

```python
src = i - delay          # delay=0 -> src=i
tgt = w[src]
r_gross = prev * lr[i]   # prev = w[src-1] = w[i-1] when delay=0
```

i.e. the **primary path earns `w_{i-1} × logret_i`** — one bar later than the
spec's `w_i × logret_{i+1}`. Verified on synthetic series (no validation data):

| Check (synthetic) | Spec | Harness `run(delay=0)` |
| --- | --- | --- |
| Jump test `[100×72, 101, 110]` — only return is bar 73 | `+0.085360` | **`0.000000`** |
| Small series, lookback 2 | spec primary `0.047856` | `0.038100` = **spec 1-bar-delayed** |

The harness's `delay=1` reproduces the spec's *2-bar* figure, so **no harness
path reproduces the spec's primary convention**:

```
spec primary s0 = 0.047856 | spec 1-bar s1 = 0.038100 | spec 2-bar s2 = 0.028438
harness delay=0 = 0.038100  -> equals spec 1-BAR-DELAYED
harness delay=1 = 0.028438  -> equals spec 2-bar
```

This is a genuine off-by-one, not a labelling choice: the harness's own
`weights()` docstring says "w[i] is the target held over the move from close[i]
to close[i+1]", which is the **spec's** convention — but `run()` consumes
`w[i-1]` on bar *i*. The docstring and the loop disagree.

**Impact is material, not cosmetic.** On the already-inspected *development*
period (no validation data used), the two conventions differ by:

| Symbol | spec gross | harness gross | delta |
| --- | --- | --- | --- |
| BTC/USD | −3452 bps | −5512 bps | **−2060 bps** |
| ETH/USD | −3703 bps | −7674 bps | **−3971 bps** |
| SOL/USD | −6721 bps | −6786 bps | −65 bps |

The headline would be computed under the *sensitivity* convention, not the
primary — and the pass criterion (#2, gross ≥56 bps/trade) is evaluated on the
primary. Running the frozen experiment with this harness would answer a different
question than the one pre-registered.

## 4. Two further defects found (secondary, but must be fixed before any run)

- **Per-trade net is closing-bar-only (spec §5 criterion 3).** `run()` line 116
  appends `r_net` of the *single closing bar* as the trade's net, not the sum of
  the trade's bars. Synthetic check: closes `[100,100,100,120,119,100]` produce
  `trades_gross=-0.007417` but `trades_net=0.000000`, while the full round-trip net
  is −0.007417. The moving-block bootstrap (criterion 3) and the per-trade mean
  (criterion 2) both consume `trades_net`, so this would corrupt the p-value and
  the pass/fail decision.
- **Unclosed final trade recorded as `0.0`** (line 122), not its real net — biases
  the final trade toward zero in both the mean and the bootstrap.

(A fourth, cosmetic-but-confusing point: `information_coefficient()` uses
`w[i]×lr[i]` — the *spec* convention — so the reported IC and the backtested
return are computed under **different** timing conventions.)

## 5. Why this is a stop, not a fix-and-continue

The instruction says: *"Do not change the frozen specification or use validation
outcomes to select parameters"* and *"if any essential parameter is unspecified
or inconsistent, stop and report it rather than choosing a value."* Here the
**spec is fine**; the **harness is non-conformant**. Fixing the harness is a code
change that would itself need to be verified against the spec and re-frozen
before validation — it is not a documentation edit, and doing it now would be
proceeding on an unvalidated instrument. So I stopped before fetching data.

## 6. State (unchanged, safe)

- **No validation data fetched.** `data/ohlcv/` still holds only the
  2025-10-01..2026-10-24 development files; no `data/ohlcv_dev2021_2025/` exists.
- The strategy was **not** run on 2021-01-01..2025-09-30 or any validation period.
- `reports/MOMENTUM72_RESULTS_2026-10-25.md` was **not** created (nothing to report).
- Harness tests: `tests/test_momentum72_study.py` → **12 passed** — but they do
  **not** pin the return-accounting convention end-to-end (they only assert
  `gross_log > net_log` and warmup/weights properties), which is why the
  off-by-one slipped through. That is itself a gap to close.
- 0 staged; patch `d1131e48…`; protected files unchanged; nothing committed.

## 7. Recommended next step (for the operator to authorize)

1. Authorize a **harness fix only** (no spec change), aligning `run()` with the
   spec's §3 formula: primary earns `w_i × logret_{i+1}`, the delayed variant
   earns `w_{i-1} × logret_{i+1}`; accumulate **full round-trip** per-trade net;
   record the final open trade's real net.
2. Add regression tests that pin these against hand-computed values (the current
   12 tests do not).
3. Re-verify the spec hash is unchanged, re-freeze, then run on the untouched
   window.
4. Only then fetch the 2021-01-01..2025-09-30 bars and execute.

No parameter, period, or criterion was chosen or tuned in producing this report.

## Reproduce (synthetic data only; no validation data)

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0,'scripts')
import momentum72_study as m
c=[100.0]*72+[101.0,110.0]
w=m.weights(c); lr=m.log_returns(c)
print("spec ", sum(w[i]*lr[i] for i in range(len(c)-1)))   # 0.085360
print("harness", m.run(c)["gross_log"])                    # 0.000000
PY
```
