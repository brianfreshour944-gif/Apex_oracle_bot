# MOMENTUM72 — EXPERIMENT CLOSED (rejected, archived)

Status: **CLOSED — hypothesis rejected under its registered rules.**
Closed: 2026-10-25. No retuning. Not deployed. No further momentum variation run.

This is the archival close-out. It changes no result and adjusts no threshold. It
records the frozen artifacts and their hashes, restates the verdict, and lists what
is explicitly **out of scope** for this experiment.

Base commit: 7a8ca12 (branch main), 0 staged.

---

## 1. Verdict (final, unchanged)

**FAIL.** Per spec §6, "if gross mean < 56 bps/trade on validation → FAIL (reject
hypothesis)." That decisive condition failed on every symbol; criteria 3 (p<0.05)
and 4 (net > buy-and-hold) also failed.

| Symbol | Gross bps/trade | Required | Net bps/trade | IC | p (boot) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| BTC/USD | +12.59 | 56 | −39.28 | 0.006 | 1.000 | FAIL |
| ETH/USD | +12.58 | 56 | −35.61 | 0.005 | 1.000 | FAIL |
| SOL/USD | +33.17 | 56 | −1.88 | 0.007 | 0.583 | FAIL |
| Pooled BTC+ETH (primary) | +12.58 | 56 | −37.47 | — | 1.0000 | FAIL |

The single most informative finding: **frequent exposure changes overwhelmed a
weak signal.** The volatility-scaling resized almost every bar (11k–16k exposure
changes per symbol), producing 29k–59k bps of modeled cost drag; gross per trade
(~13–33 bps) was far below the 56 bps cost bar. The signal also carried almost no
information (IC ≈ 0.005–0.007).

## 2. Preserved artifacts (frozen, with exact sha256)

| Artifact | sha256 |
| --- | --- |
| `reports/MOMENTUM72_SPEC_2026-10-25.md` (frozen spec — historical record) | `08316c38e11f8e20e6fcb795c5fa4daa220c919364209ad808b2a865a21af854` |
| `reports/MOMENTUM72_VALIDATION_MANIFEST_2026-10-25.md` | `d6746317679dd4062b8d85e9512f13e0a997a9d653262ef9979cf32ac3d070aa` |
| `reports/MOMENTUM72_RESULTS_2026-10-25.md` | 1e3bf8ae5d1778249027e3ad4469625aa442ea9cdb06b10cd07fb6c98eb307a2 |
| `reports/MOMENTUM72_HARNESS_REFREEZE_2026-10-25.md` | `f67ddf6ff5c8b2d684aebfb2028f6ccbfd1f2c6cc3f13646cbf0ce53790f712f` |
| `reports/MOMENTUM72_PREREQ_FAILURE_2026-10-25.md` | `b318d63e9d36611012845d0529d2c87f3acb5e0d94242cdb95e72e09fa2aa20d` |
| `scripts/momentum72_study.py` (re-frozen instrument) | `24f39a9d7da70a02b5f8c7c63a0e5aa0efa0dfe2df51e0722fd5e12e1c35f752` |
| `tests/test_momentum72_study.py` | `5dec4b927e8f283efee9bf28b1d201f8fdcc7e20fa2f10c1aa90354b9db19634` |

Validation cache (gitignored, `data/ohlcv_dev2021_2025/`):

| File | sha256 |
| --- | --- |
| BTC_USD_1Hour.json (41,597 bars) | `71a838b7533e887aa241f3e280d2914d479f5334d62ae3c2ddacd2af5ac789a7` |
| ETH_USD_1Hour.json (41,596 bars) | `073fd25a7295e08f2db797d5323058a4d5fc7e5dbcf9fed801a50ed7c4f85ddc` |
| SOL_USD_1Hour.json (31,521 bars) | `b415e525fe17432f0f4560e06c95a92687952b7f29a21e018872e27321e2ddf7` |

Development cache (unchanged, `data/ohlcv/`): BTC `00f8a605…`, ETH `476bfe65…`,
SOL `f4877008…`.

## 3. Explicitly out of scope for this experiment (kept separate)

- **Historical P&L / fee provenance.** The deployed-snapshot audit's open issues
  (`orders.commission` has no provenance; the learner and rolling loss gate consume
  stored `realized_pnl` regardless of `outcome_status`) are **not** addressed here
  and are not implied by this verdict.
- **Real-world costs.** These results are under the experiment's **modeled** 56 bps
  round trip. They do not establish the actual commissions and slippage a paper
  account would see. The modeled-cost verdict is a statement about the strategy's
  gross signal vs. an assumed cost, not a measurement of real fills.
- **Production behavior.** No risk limit, committee behavior, execution logic,
  accounting patch, or deployment config was or will be changed on the basis of
  this experiment. Live trading remains frozen.

## 4. What this experiment does and does not establish

**Establishes:** the frozen 72h volatility-scaled momentum hypothesis, on
2021-01-01..2025-09-30, failed its pre-registered bar — gross too small vs. cost,
net negative, IC ≈ 0. It also demonstrates, concretely, that frequent exposure
changes can overwhelm a weak signal.

**Does not establish:** that no momentum strategy can work; that the true edge is
exactly zero (a bootstrap p of 1.000 is a property of the specified test — see
RESULTS §5); or anything about regimes other than the one tested.

**Independence note:** because this window has now been used, it is no longer
"untouched." Any future hypothesis must use a *different* independent period, and
retuning this one on the same window would destroy that independence.

## 5. Close-out checksums (post-edit)

Recorded after the RESULTS §5 wording refinement (no numbers or verdict changed):

- `reports/MOMENTUM72_RESULTS_2026-10-25.md` — `1e3bf8ae5d1778249027e3ad4469625aa442ea9cdb06b10cd07fb6c98eb307a2` (wording refinement only; numbers and verdict unchanged).

## 6. Safety boundary (confirmed at close)

- 0 staged; nothing committed, deployed, restarted, or ordered.
- Spec `08316c38…` unchanged; instrument `24f39a9d…` unchanged; accounting patch
  `d1131e48…` unchanged; protected files byte-identical to HEAD.
- No retuning; no alternative variants; no new momentum run.
