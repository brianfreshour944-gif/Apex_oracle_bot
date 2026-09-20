# Apex Oracle Bot — Adversarial Audit

**Date:** 2026-09-20
**Method:** Static code audit (4 parallel deep-reads of the full `src/` tree, `scripts/`, tests, and existing docs) + git history. **No `.env` present in this clone, no `data/bot.db`, and Python deps (polars/torch/etc.) are not installed here — I could not run the bot, a backtest, or the test suite.** Every claim below is either a direct code citation or explicitly marked as untestable from a static read. Nothing was changed; this is read-only per your instruction.
**Stance:** For every strength claimed by the code/docs, I've tried to argue the skeptical case against it. For every weakness found, I've also asked "does fixing this actually matter, or is it a distraction?" For every enhancement idea, I've applied the same skepticism — per your framework's own Complexity Budget section, a fix that can't show measurable OOS value doesn't earn its keep.

---

## 0. The one finding that should reframe how you read everything else

Across all four sub-audits, the same shape of problem kept recurring: **the codebase contains real, well-built validation/safety machinery that is defined, sometimes even unit-tested, and then never actually called from the live path.** This isn't a vague impression — it's a specific, repeated, verifiable pattern:

| Module/feature | What it claims to do | What actually happens |
|---|---|---|
| `src/feature_ablation.py` | Ablation study identifying which features matter | Returns hardcoded recommendation strings; never imported anywhere else; docstring admits it's "a framework," not an implementation |
| `src/feature_drift_monitor.py` | Detects feature distribution drift (PSI/KS — this part is real) | Drift *is* computed and logged, but `log_weekly_report()`/`get_feature_drift_alerts()` have **zero callers** anywhere in the repo. Drift is recorded into a state file and never surfaces to a human or to trading logic. |
| `RiskManager.check_correlation_concentration()` | Caps correlated exposure | Defined, never called from anywhere. A cruder same-regime-fraction proxy (`_MAX_SAME_REGIME_FRACTION`) is used instead, by the code's own admission (`risk.py:37-44`). |
| `KELLY_FRACTION` config | Kelly-criterion sizing | Defined in config, read only by two audit/dump scripts, never read by `risk.py` or `committee.py`. Not implemented. |
| `src/walkforward.py` | Real multi-window, multi-asset walk-forward validator | Well-built and broader than what's actually used — but it is **never called** from the model-promotion path. The actual promotion gate (`scripts/retrain_transformer.py`) uses a narrower single-symbol/single-90-day-window check instead, exactly as `KNOWN_ISSUES.md` already flags. |
| `max_adverse_pct` (MAE) in `db.py`/`bot.py` | Tracks max adverse excursion per trade | Hardcoded to `0.0` forever (`bot.py:157`). Only MFE (favorable excursion) is actually computed. |
| `get_decay_alerts()` (Sharpe/win-rate/PF decay) | Surfaces performance decay for monitoring | Computed, logged via `logger.warning`, **zero callers** — never read by anything that acts on it. |
| `README`'s "shadow mode by default" claim for adaptive ML | Implies `ADAPTIVE_ML_ENABLED=false` out of the box | **`src/config.py` field default is `True`.** `.env.example` says `false`, README says shadow-by-default, but the actual Pydantic field default that applies when no `.env` exists is `True`. This repo, cloned exactly as you gave it to me, would run in **live-adaptive mode**, not shadow mode, the moment it started — the opposite of three separate places telling an operator otherwise. |

**Adversarial take on this pattern itself:** it's tempting to read a long list of well-named modules (`ood_discriminator`, `shadow_arena`, `walkforward`, `feature_ablation`, `circuit_breaker`, `fitness_evaluation`) as evidence of a mature, rigorously-validated system. A skeptical reviewer's first move should be exactly what I had the agents do: for every safety/validation module, ask "is this on the path that actually executes before an order goes out, or is it a parallel universe that logs to a file nobody reads?" Several of the ones above are the latter. That doesn't mean the system is unsafe — the things that ARE wired in (OOD veto, sentinel hard veto, drawdown killswitch, exposure/position-count reservation, circuit breaker, gap-risk throttle) are real and do gate live orders. But the count of "looks like a safety feature, isn't one in practice" items is large enough that I'd treat every unverified claim in the README with suspicion until you've grepped for its actual call sites, the way these agents did.

**Recommended test (cheap, no code change):** `grep -rn "def log_weekly_report\|def get_decay_alerts\|def check_correlation_concentration\|WalkForwardValidator\|feature_ablation" --include="*.py"` and manually confirm which hits are definitions vs. actual callers outside tests. This single 10-minute audit habit would have caught most of the "theater" items above without needing 4 sub-agents.

---

## 1. Strategy & Trading Edge

### Core edge, per brain
- **Momentum/regime brain** (`momentum_brain.py`): votes purely off the *regime label string* (bull/bear/trending/sideways), not a computed momentum value. Docstring claims "structural market regime transitions"; there's no actual momentum calculation in this file — just hardcoded confidence per regime name (0.85/0.75/0.60/0.45).
- **Quant brain** (`quant_brain.py`): pure RSI thresholds (<25 buy, >75 sell, softer bands 25-40/60-75). README describes it as "z-score of price relative to Bollinger Bands" — **that is not what the code does.**
- **Transformer brain**: a GQA transformer with MC-dropout epistemic uncertainty, `sigmoid(logit) > 0.55` → buy / `< 0.45` → sell. Sophisticated machinery, but the actual decision rule collapsing out of it is a threshold on a learned probability — same shape as any other Brain, just with a much heavier compute path behind it.
- **Sentinel**: pure risk veto (flash crash / halted / crash regime / high ATR / high volatility regime), can't vote directionally.
- **LLM brain**: sentiment score threshold ±0.4 with confidence >0.5, plus its own independent veto on security/regulation events.

### Documented rationale for why the edge should exist
**None found, anywhere** — not in code comments, not in docstrings, not in the README — for *why* RSI extremes, Hurst-based regime classification, or the transformer's learned pattern should continue to produce alpha after being discovered by other market participants. Every description found is mechanical ("buys when RSI < 30") or statistical ("Hurst > 0.5 = trending"), never economic. Section 20's honesty-test question — "what part of the evidence would a skeptical quant researcher attack first?" — the answer is this: **there is no articulated reason any of these five signals should have positive expected value net of costs, only a description of what they compute.** RSI mean-reversion and momentum-regime signals are among the most heavily-arbitraged, well-known patterns in retail-accessible crypto trading; five-year-old RSI(2) strategies on BTC are well documented to have decayed toward noise as more participants run them.

**Adversarial take:** I'd push back on myself here too — "no articulated rationale in the code" isn't proof there's no edge; a lot of profitable systematic trading is empirically-discovered and doesn't need a textbook citation to work. But it does mean **you currently have no way to distinguish "this signal exploits a real, persistent inefficiency" from "this signal is curve-fit noise that happened to backtest well"** without actually running edge verification (cross-asset, cross-regime, cross-parameter stability — section "Edge Verification" below). And you don't currently have any tooling that runs that verification (see next point).

### Edge verification — not traceable to the current live defaults
`sweep_params.py` only tunes `MAX_HOLD_HOURS`/`STOP_LOSS_PCT`/`PROFIT_TARGET_PCT`. `bayesian_optimizer.py` has four parameter spaces (meta-learner, committee, transformer, risk) — **none of them include the Hurst thresholds, RSI thresholds, ATR volatility ceiling, or the regime position-size multipliers (1.5x/0.8x/0.5x) that are actually live.** There is no output artifact anywhere in the repo linking the live-default values in `.env.example`/`config.py` to any backtest or optimization run. **These numbers cannot currently be shown to come from evidence rather than intuition — I looked for the trail and it doesn't exist in this repo.**

There's also a live bug that makes this worse than it looks: `risk.py:661-666` branches on `regime == "mean_reverting"` for the 0.8x position-size multiplier, but the actual regime classifier in `strategies.py` only ever emits `"sideways"` for that state (confirmed by both the strategy agent and cross-checked against `regime_utils.py`'s vocabulary). **The mean-reversion risk multiplier is dead code on the live/backtest path as currently wired** — mean-reverting trades are being sized as if regime multiplier is 1.0 (the `else` fallthrough), not the intended 0.8x. This is a genuine, currently-live discrepancy between intent and behavior, not a style nit.

There's a second, separate discrepancy: `HIGH_VOLATILITY_PCT` defaults to `12.0` in `config.py`, but the README states `5.0%` (`README.md:93`). Either the README is stale or the shipped default drifted from what was intended/documented — either way, an operator reading the README has the wrong mental model of when the bot stands aside for volatility.

### Required Output (per your framework)
- **Edge confidence: Low.** Not because the signals are necessarily bad, but because there is zero in-repo evidence connecting the live-default thresholds to any verification process, no economic rationale documented, and at least one of the five regime-conditional risk multipliers is provably unreachable given the current regime vocabulary.
- **Primary evidence:** None — no cross-asset/cross-regime/cross-parameter stability study, no backtest artifact tying current defaults to a result, exists in this repo.
- **Biggest weakness:** No falsifiable, documented reason any brain's signal should retain edge after discovery, combined with tuning tools that don't cover the parameters actually in production.
- **Recommended test:** Before anything else, run `bayesian_optimizer.py`'s risk/committee spaces *and* extend it (or `sweep_params.py`) to cover Hurst/RSI/ATR regime thresholds across at least 2 assets and 2 non-overlapping time windows, and diff the result against the current `.env.example` defaults. If the live defaults aren't near a local optimum found independently across those windows, that's a strong signal they were hand-picked, not derived.

**Enhancement idea (with adversarial pushback attached):** Add an economic-rationale docstring requirement to each brain — one paragraph per brain, in code, stating the hypothesized mechanism and the condition under which it should decay. *Adversarial pushback on my own suggestion:* this is cheap to add and easy to approve, but a comment doesn't create evidence — it just documents an assumption. Don't let writing the rationale substitute for testing whether the rationale holds (section 20's "Improvement" question — the single smallest change with the largest expected improvement here is the cross-regime/cross-asset test above, not documentation).

---

## 2. Data Integrity & Information Quality

### Features (the 11 "institutional" features actually used)
`z_return`, `parkinson_vol`, `garman_klass_vol`, `kyle_lambda`, `signed_flow`, `vwap_z`, `vol_of_vol`, `amihud_z`, `trade_size_proxy`, `roll_autocorr`, `range_position_z` (`feature_engineering.py:48-60`). All rolling windows are trailing (`.rolling(center=True)` — zero hits repo-wide), so no classic centered-window leakage. Good.

### Leakage — confirmed, real, currently in the training pipeline
`scripts/train_foundation_model.py:116-136`: `StandardScaler().fit_transform()` is called on the **full concatenated dataset across all symbols/timeframes**, and only *afterward* is it split 80/20 chronologically into train/val. This is the textbook version of the leakage pattern your framework explicitly calls out: **the scaler that features get normalized through has already seen the validation set's distribution before training happens.** This is a real, current, verifiable defect in the foundation-model training path — not a retracted false positive like the two bugs in the old `AUDIT_REPORT.md`.

**Adversarial take:** how much does this actually matter? For a `StandardScaler` (mean/std normalization) specifically, the leakage is milder than it would be for, say, a leaked label — it shifts feature scale slightly based on validation-period statistics, which inflates validation metrics somewhat but is less catastrophic than genuine label leakage. Still, it means **your reported validation Sharpe/accuracy for the foundation model is optimistically biased by an unknown, unmeasured amount**, and the size of that bias will vary with how different the validation period's volatility regime is from the training period's. Given crypto's regime-swings, this is not a "probably fine" leakage — it should be fixed and the model re-validated with the fix before you trust any validation number that came out of this path.

### The replay-dataset fix is not actually fixed
`KNOWN_ISSUES.md` says the corrupted `historical_experiences.jsonl` needs regenerating via `scripts/generate_replay_dataset.py`. Git shows it *was* regenerated (commit `f30eb2f`, 2026-08-16, 6,312 records) — but that same commit introduced `REPLAY_FAST_MODE` defaulting to **on**, which populates `tensor_state` from `np.random.RandomState(seed).randn(128)` — **seeded random noise, not real feature vectors** — unless an operator explicitly sets `REPLAY_FAST_MODE=0`. The commit message doesn't mention setting that flag. **The dataset that KNOWN_ISSUES.md's "action needed" asked you to regenerate has very likely been regenerated with synthetic noise instead of real corrected features.** This is worse than the original bug in one sense: the original at least had *real, if wrongly-scaled* data; the current file may be closer to pure noise dressed up as training data. Anything trained on this file (`scripts/retrain_transformer.py` consumes it directly) inherits this.

**Recommended test (cheap, no install needed beyond what's already there):** `python scan_learning.py` against the current `data/historical_experiences.jsonl` and check whether `tensor_state` values look like `N(0,1)` random noise (mean ~0, std ~1, no autocorrelation with `label`) versus real feature statistics. This is a 2-minute check that would confirm or rule out my reading of the git history.

### Data quality checks
No duplicate-timestamp, gap, or missing-period validation found for OHLCV bars anywhere (`data_fetcher.py` only filters `close > 0` and backfills `vwap`). `feature_engineering.py`'s `_sanitize()` silently coerces NaN/inf to a fill value — **this masks bad data rather than rejecting or flagging it.** `scripts/verify_data_integrity.py` (the check FOUNDATION_FREEZE.md shows failing) validates *fill/order records against Alpaca*, not the bar/feature data itself — it's currently failing for the mundane reason that there are zero real trades in the DB to check, not because of a data-quality defect per se.

**Adversarial take on "is this a real problem":** for a bot polling a single reputable exchange's bar API, missing-period/duplicate-timestamp bugs are a lower-probability failure mode than, say, a stale-bar-not-yet-closed bug (see below) — I wouldn't over-invest here. But silently coercing NaN/inf via `_sanitize()` is a real risk: if `garman_klass_vol` or `kyle_lambda` produces inf (e.g. `log(H/L)` on a zero-range bar, or divide-by-zero on `volume`), the current behavior is to fill and continue trading on a fabricated value rather than stand aside. That's the kind of bug that looks fine for months and then produces one very bad trade on a data anomaly.

### One I found myself, not from the agents: is the "current bar" actually closed?
No agent found any `is_closed`/"forming bar" filter in `data_fetcher.py` or the feature pipeline. If the last bar Alpaca returns for a still-in-progress interval is fed into `add_features()` and then into the committee, several of the 11 features (which use the *current* bar's OHLC, not a shifted one — see the "point-in-time?" column the agent built) would be computed on a bar that's still mutating. This wouldn't be "look-ahead" in the backtest-correctness sense, but in live trading it's a live-vs-backtest parity risk: the backtest presumably iterates over already-closed historical bars, while live trading might be evaluating on a partially-formed one, meaning live signal quality is silently worse than whatever backtest validated it.

**Recommended test:** confirm explicitly (grep for how `fetch_bars`'s time range end is set, and whether it's `now` or `now - 1 bar interval`) whether the last row is guaranteed closed. This is a fact-check, not a fix, and should be resolved before trusting any backtest number as representative of live behavior.

### Required Output
- **Data-quality score: Medium-Low.** Real, currently-active leakage in the foundation-model scaler; the replay dataset's actual corruption status is unresolved and possibly worse than before; NaN/inf silently masked rather than surfaced.
- **Potential leakage detected:** Yes — confirmed in `train_foundation_model.py`'s scaler-fit ordering.
- **Most valuable features:** Not determinable from this repo — `feature_ablation.py` is not a real ablation and no comparative retraining evidence exists anywhere.
- **Redundant features:** Not determinable for the same reason — the correlation-matrix code in `feature_ablation.py` computes real correlations but never acts on them or feeds a report anyone reads.
- **Features requiring investigation:** `kyle_lambda`, `amihud_z`, `garman_klass_vol` — all divide by volume or log(H/L), all vulnerable to inf/NaN on degenerate bars (zero volume, zero range), all silently sanitized rather than flagged.

---

## 3. Signal Generation & Decision Quality

Entry/exit logic per committee mechanics: 5 brains vote concurrently, combined via a static per-regime weight matrix (`REGIME_WEIGHT_MATRIX`, hardcoded dict, no config field) into a weighted-average score, gated by `DEFAULT_SCORE_THRESHOLD` (0.15, relaxed to `max(0.15, threshold*0.75)` in sideways/low-vol/neutral regimes). Sentinel hard-veto (confidence forced to 0.95) overrides everything on flash-crash/halted/crash-regime.

### Signal strength / confidence calibration — the calibration code exists and is never run
Real Expected-Calibration-Error code exists (`bayesian_transformer.py::compute_ece`, temperature scaling in `train_temperature_scaling`) — this is exactly the "predicted probability vs actual win rate" check your framework asks for. But `USE_BAYESIAN_TRANSFORMER` isn't even a defined config field (defaults `False` via `getattr` fallback), and grep shows **no scheduled job or call site anywhere invokes this calibration.** So the confidence scores driving position sizing (`calculate_confidence_size_multiplier`, 0.40x–1.75x range) have never been checked against realized outcomes in this codebase. You are sizing bigger on higher-confidence signals with no evidence those confidence values are honest.

**Adversarial take:** this is one of the highest-leverage gaps in the whole audit. Position sizing scales 0.4x-1.75x directly off a confidence score whose calibration has never been measured. If the score is systematically overconfident (a very common failure mode for softmax/sigmoid outputs without temperature scaling), you are structurally oversizing your worst trades. This is cheap to check — the ECE code already exists — and expensive not to know.

### No percentile-based signal filtering
Confirmed no code path filters trades to only the top 10%/20% strongest signals — filtering is via fixed/adaptive scalar thresholds only. **Enhancement candidate:** add a mode that logs (not yet acts on) the outcome-by-confidence-decile table your framework's calibration section describes, using existing DB fields (`confidence`, `realized_pnl` are both already columns per the outcome-tracker agent's findings) — this requires no new instrumentation, just a query against data you already have once real trades accumulate. *Adversarial pushback on this idea:* with likely single/low-double-digit real trade counts so far (see Track Record below), decile buckets will be statistically meaningless for a long time — don't over-build this before you have the sample size to use it.

---

## 4. Market Regime Detection

Regime matrix (Regime × Trades × Win Rate × Avg P&L × Profit Factor × Max DD) **cannot be filled in from this audit** — there is no `data/bot.db` in this clone and I have no way to query real trade history. This table needs to be generated by you, from your actual deployment's database, via something like `scripts/track_record_status.py` or a direct query. I'd treat any regime-performance claims as unknown until that table exists with real numbers.

What I can confirm from code: regime detection happens *before* trading (Hurst/ATR computed live each cycle, feeding both the strategy selector and the committee — it's not post-hoc). Whether disabling trading in historically-unfavorable regimes would help is unanswerable without the matrix above.

---

## 5. Risk Management

### Position sizing
Real, layered: base % → regime multiplier → confidence weighting (clamped 0.5-1.5x) → drawdown taper (tapers to 0.25x floor as equity approaches `MAX_DRAWDOWN_STOP`) → gap-risk multiplier → ATR/percent-stop-distance-based sizing (size is genuinely inversely proportional to ATR, confirming the framework's "is size the same when vol doubles" question with a **no, it's not** — good) → hard `$MAX_SINGLE_TRADE_USD` cap → L2 order-book impact reduction → edge-vs-cost rejection (`TX_COST_MIN_EDGE_BPS`, trades below net-of-cost edge are rejected outright, not just downsized — genuinely good).

### The portfolio-cap discrepancy I found directly
`.env.example` documents `MAX_PORTFOLIO_VALUE=$500` as "Maximum cumulative portfolio exposure," with no mention anywhere in that file of `MAX_PORTFOLIO_PCT`. But `config.py` defaults `MAX_PORTFOLIO_PCT=0.5`, and `_get_max_portfolio_cap()` uses `ACCOUNT_BASE * MAX_PORTFOLIO_PCT` whenever that field is set — which it always is by default. **The effective live cap is $5,000 (10x the documented $500), silently.** An operator who read only `.env.example` and thought they'd capped exposure at $500 would be wrong by an order of magnitude. This is the second confirmed instance (after the `ADAPTIVE_ML_ENABLED` default mismatch) of the shipped example file / README describing more conservative behavior than the actual code default provides. That's a pattern worth taking seriously on its own: **twice now, the safety-relevant default in code is more permissive than what the documentation tells an operator to expect.**

**Adversarial take:** is $5,000 vs $500 actually dangerous given a $10k `ACCOUNT_BASE`? Not obviously reckless in isolation (50% exposure is a defensible risk posture) — but "not obviously reckless" isn't the point. The point is an operator's mental model of their own risk configuration is wrong, silently, and would only surface the first time actual exposure exceeded $500 and they went looking for why. Fix candidates: either update `.env.example` to show `MAX_PORTFOLIO_PCT` explicitly next to `MAX_PORTFOLIO_VALUE` with a comment on which wins, or (better, since it's cheap) log a one-time startup warning when the effective cap differs from the static `MAX_PORTFOLIO_VALUE` value by more than some margin.

### Caps that are defined but not enforced
`check_correlation_concentration()` — defined, never called. `KELLY_FRACTION` — defined, never read by sizing code. A cruder same-regime-fraction cap (`_MAX_SAME_REGIME_FRACTION=0.67`) stands in for real correlation control, and the code's own comment (`risk.py:37-44`) already admits this. **This is a self-documented gap, not a hidden one — worth noting in your favor: whoever wrote that comment already knew.**

### MAE is fake
`max_adverse_pct` is hardcoded `0.0` forever; only MFE is real. This directly blocks answering your framework's own exit-analysis questions ("how far did price move against the position before recovering" is literally unanswerable from current data). **Enhancement candidate, fairly cheap:** track a running min-price-since-entry alongside the existing max-price-since-entry (`_trailing_peaks`) — the machinery for the favorable side already exists and just needs a mirrored adverse-side counterpart. *Adversarial pushback:* small correctness fix, not a strategy improvement by itself — don't let "we now have MAE" feel like progress on edge; it's a measurement fix that lets you *evaluate* stop placement later, nothing more.

### Required Output for Risk of Ruin
Real Monte-Carlo risk-of-ruin exists (`backtest.py::run_monte_carlo_analysis`, 1000 shuffled-sequence sims, `risk_of_ruin_pct` = fraction with >20% max DD) and is actually used to gate evolutionary parameter search (`evolutionary_ppo_trainer.py` rejects candidates with `risk_of_ruin_pct > 5.0`). This is one of the more genuinely solid pieces of the whole system — real, wired in, gating something. Caveat: it assumes historical trade-return *distribution* stays representative, which per your framework's own instruction ("do not assume historical win rate will remain constant") is exactly the assumption most likely to break for a 2-month-old bot with a small, possibly-not-yet-representative trade sample.

---

## 6. Stop-Loss & Exit Logic

**Live and backtest diverge**, confirmed: live trading calls `risk.py`'s regime-scaled trailing stop (high-vol widens 1.5x, low-vol/sideways tightens to 0.6-0.8x) *before* the strategy's own signal generation even runs, effectively short-circuiting `strategies.py`'s separate fixed-percentage trailing stop. Backtest never calls the regime-scaled version at all — it only exercises the fixed-percentage one. **This means your backtest results for trailing-stop behavior are not representative of what live trading actually does**, in either direction — you can't validate the regime-scaled trailing stop logic through the backtest path as currently wired, and you can't fully trust the backtest's trailing-stop-driven P&L as a preview of live behavior.

**Adversarial take:** which one is "more correct" isn't obvious — regime-scaling the trailing stop is a reasonable idea (give more room in high vol, less in low vol), but you have literally never backtested it, because the backtest path doesn't exercise that code. Every dollar of P&L attributable to the regime-scaled trailing stop in live trading is currently un-validated. This is a concrete, fixable **test-coverage gap masquerading as a feature**, and I'd rank it above most of the "enhancement" ideas in this report because it directly undermines your ability to trust backtest numbers for exit logic specifically.

**Recommended test:** wire `backtest.py` to call the same `risk_manager.check_trailing_stop()` path live trading uses (or, if that's intentionally a live-only feature, document why explicitly, and build a *separate* backtest specifically for the regime-scaled trailing-stop logic before trusting its live P&L contribution).

---

## 7. Execution & Trading Costs

All 4 live order-creation call sites use `type="market"`. Limit-order support with `post_only` exists in `exchange.py` and is **entirely unused** — every live order is a market order regardless of urgency or liquidity. Backtest fill assumption is instant fill at the current bar's close, no partial fills, no latency, no rejected orders modeled.

`run_backtest()`'s `fee_pct`/`slippage_pct` parameters are **declared and never used** — the actual costs applied come from a separate path (`risk.get_transaction_costs`, 10bps one-way default). This is dead/misleading function signature, not a functional bug (the real cost model is applied elsewhere), but it's exactly the kind of thing that misleads a future reader of `run_backtest()`'s signature into thinking they can vary costs by passing those args, when they can't.

**No cost-stress-test found** (2x/3x fees) anywhere. This is one of the cheaper, higher-value tests your framework recommends and it doesn't exist yet. **Enhancement candidate:** parametrize `TX_COST_*` overrides through `run_backtest()` (they already flow from `risk.get_transaction_costs`, which reads `settings`) and run the existing backtest 3x at 1x/2x/3x cost multipliers. *Adversarial pushback on doing this now:* with likely minimal real trade history, this test would run against historical/simulated bars rather than validate live cost assumptions — useful as a first-pass robustness check, but don't treat a backtest-based cost-stress-test as proof live economics survive 2x costs; live slippage on market orders in thin crypto pairs can easily exceed what a bar-close-fill backtest models.

---

## 8-10. Backtesting Integrity, Walk-Forward, Monte Carlo

- **Walk-forward**: `src/walkforward.py` is a genuinely well-built rolling-window validator with purge/embargo — better than what I initially expected — but it's dead code from the model-promotion path's perspective (never called there). The path that's actually wired in (`retrain_transformer.py`) is the narrow single-symbol/90-day one `KNOWN_ISSUES.md` already flags. **You have the better tool sitting unused next to the worse one that's actually running.** This should be one of your cheapest, highest-value fixes: swap the promotion gate to call the existing `WalkForwardValidator` instead of building anything new.
- **Monte Carlo**: real, wired into evolutionary parameter search. Good. Caveat above about representativeness stands.
- **Backtest fee/slippage stress test**: not found (see §7).

---

## 11. Model & AI Contribution

**Ablation**: fake (see §0). **You cannot currently answer "does removing brain X hurt or help OOS performance"** for any of the five brains — the infrastructure exists (`fitness_evaluation.py`'s components, the correlation-matrix code) but nothing runs a true remove-and-reevaluate comparison. Given the framework's own instruction — "a component should not remain merely because it sounds sophisticated" — the LLM brain and the transformer brain (the two most compute/dependency-heavy) are exactly the ones I'd want ablated first, and currently can't be, without you building the actual comparative-retrain harness `feature_ablation.py` only pretends to be.

**Adversarial take, playing devil's advocate on the whole committee architecture:** five brains combined by a hand-set weight matrix, with no measured per-brain OOS contribution, is a design that *looks* like it should be more robust than a single model (diversification intuition) but has no evidence behind that intuition in this repo. It's equally plausible that 2 of the 5 brains are pure noise-adders that the weighted-average is diluting real signal with. The honest answer to "does every model contribute measurable predictive value" (your own §11 question) is: **unknown, and currently unknowable without doing the ablation work for real.**

---

## 12. Adaptive / Self-Learning Behavior — the critical safety-default bug

This is the single most important finding across all four agents: **`ADAPTIVE_ML_ENABLED` defaults to `True` in `src/config.py`**, contradicting `.env.example` (`false`) and the README's explicit "Default = paper-only shadow mode" claim. Since this clone has no `.env` file, the *actual* behavior on a fresh checkout is: adaptive weights compute and, once the gate (`ADAPTIVE_MIN_TRADES_BEFORE_LIVE`, now apparently defaulting to 30 per the config the agent read — note this differs from the `.env.example`-documented 50, another drift between doc and code) is satisfied, **drive live decisions** — not shadow-log them.

**Adversarial take, taken seriously:** does this matter in practice? If an operator always copies `.env.example` to `.env` before running (as the README instructs, step 2), they'd get the safe `false` value and never hit this. The exposure is specifically: anyone who deploys via a mechanism that doesn't materialize `.env` from the example (a container orchestrator setting env vars directly from a different source, a CI/CD pipeline that only sets a subset of vars, or simply forgetting to set `ADAPTIVE_ML_ENABLED` explicitly while setting other vars) gets the opposite of the documented default. Given this repo shows signs of being deployed via Coolify (commit "trigger fresh deploy to verify Coolify tracks latest commit"), **I'd verify directly on your actual deployment's environment right now whether `ADAPTIVE_ML_ENABLED` is explicitly set**, rather than trusting that it inherited the safe default. This is a 30-second check with a real "could be trading live on unvalidated learned weights" downside if wrong.

Beyond the default bug: PPO's live-gate trade-count floor (`PPO_MIN_TRADES_BEFORE_LIVE=10`) is much lower than the adaptive learner's (30), and both count off the *same* shared sample counter. `KNOWN_ISSUES.md` already flags that PPO has zero real training data as of its last update and will start influencing live decisions once that counter clears — worth knowing that counter only needs to hit 10, not 30, for PPO specifically to go live, which is a lower bar than the README's framing suggests.

**Outcome-tracking mismatch risk**: trade-to-snapshot matching is symbol+"most recent open"-based, not a true `decision_id`/order-id join end-to-end. The code shows awareness of this class of bug (a documented "ghost snapshot" reconciliation pass exists), but the core lookup pattern remains exposed to exactly the race it's patching around — two simultaneously-open snapshots for the same symbol (scale-in, or two decision cycles racing before the first close lands) could get outcome-misattributed, which would silently poison the very training signal (`AdaptiveMetaLearner`, PPO, outcome tracker) this whole adaptive layer depends on. **This is a case where a subtle attribution bug could quietly corrupt the thing meant to make the system safer over time** — worth prioritizing over most of the enhancement ideas in this report.

---

## 13. Performance Metrics

Real and reasonably complete: Sharpe, Sortino, win rate, avg win/loss, payoff ratio, profit factor, expectancy, turnover, decay-alert thresholds. **No max-drawdown or Calmar computation found in `performance_tracker.py`** specifically (drawdown IS tracked elsewhere for the killswitch, just not surfaced in this report). Handles <10 trades gracefully (explicit zeros, no NaN/crash, no fabricated numbers) — genuinely good, matches your framework's concern about metrics lying with small samples.

**Track record, as best I can establish without your live DB:** git history shows the project started 2026-07-16 and has 291 commits, with a visible cluster of "fix: critical production bugs," "fix: audit findings," and "fix: crash-recovery" commits concentrated in late August, plus `KNOWN_ISSUES.md` confirming zero real closed trades as of 2026-07-26 and `FOUNDATION_FREEZE.md` (2026-08-20) showing Track Record still FAILING. **Read plainly: this is a ~2-month-old system that has not yet accumulated the 30-trade minimum its own `track_record_status.py` gate requires, as of the last recorded check.** Any edge-confidence claim beyond "Low" would currently be unsupportable by your own system's stated gates. I'd treat this as the load-bearing fact under this entire audit: most of the "is the edge real" questions in §1 aren't answerable yet not because the code is missing the tools, but because **the track record genuinely doesn't exist yet in sufficient volume.**

---

## 14. Benchmark Against Simpler Alternatives

Not found anywhere in the repo: no buy-and-hold benchmark, no simple-MA-crossover comparison, nothing that answers "does this 5-brain system beat doing nothing, or beat a 10-line strategy." `run_benchmark_comparison()` exists in `backtest.py` (consumes the Monte Carlo output) but the agents' notes don't indicate it benchmarks against a *simpler strategy* — worth you confirming directly what it actually compares against, since this is exactly the kind of humility check your own framework's §14 and the final "Simplicity" question call for, and I didn't get a direct answer on it from the sub-agents.

**Enhancement candidate:** add a literal buy-and-hold-BTC baseline column to every backtest/walk-forward report. Cheap, directly answers "what's the smallest system that produces most of the performance," and I can't construct an adversarial objection to doing this — it's pure information with no downside.

---

## 15-16. Operational Reliability & Monitoring

The *previous* audit (`AUDIT_REPORT.md`, 2025-08-22) already found 9 race conditions and 5 check-then-act patterns in `bot.py`/`risk.py`/`exchange.py` — I did not re-verify whether those are fixed (out of scope for this pass; you should re-run that audit's specific line citations against current `bot.py` line numbers, since the file has clearly moved a lot given 291 commits since then).

Real, live-wired alerts confirmed: exposure saturation, repeated vetoes, drawdown-approaching-killswitch, model load failure, trade churn, circuit-breaker trip, data-integrity failure, killswitch activation, adaptive weight-change. Genuinely solid Prometheus + Telegram coverage.

**Missing, confirmed by direct grep (not found anywhere in `src/`):** duplicate-order alert, position-desync alert (reconciliation logic exists for ghost snapshots, but no alert fires when it happens), stale-price alert, dedicated model-confidence-change alert (only weight-change is alerted, not confidence).

**Adversarial take on prioritizing these:** of the four missing alerts, I'd rank stale-price highest — a stale price feeding position sizing/stop-loss checks is a silent, hard-to-notice failure mode that could produce a bad fill or a stop that never fires, whereas duplicate-order and position-desync at least tend to be self-revealing quickly (you'd notice unexpected fills or balances). Not zero priority, just not first.

---

## 17. Human Oversight

**No runtime human-approval gate exists anywhere in `src/`.** The only "approval" language found is in `.clinerules` — an instruction file for an IDE coding assistant (telling *it* not to run live trades without asking), not a runtime control in the bot itself. Per your framework's own distinction between "automatically allowed" and "requires human approval" — model deployment, leverage changes, exchange changes, trading-universe changes, and disabling safety controls **all currently happen through code deploys, not through any in-bot approval workflow.** Given the deployment cadence visible in git history (291 commits in ~2 months, many same-day), this is worth being deliberate about: your actual human-oversight mechanism right now is "whatever gets reviewed before a `git push` to the branch Coolify deploys from," which is a reasonable model for a solo/small-team project but is worth naming explicitly rather than assuming the code has a gate it doesn't.

---

## 18. Complexity Budget

Applying your own framework's question — "what measurable problem does this solve, and can its incremental contribution be demonstrated OOS" — to the biggest complexity items found:
- **LLM brain** (sentiment + security/regulation veto): adds an external API dependency (Groq), a whole separate veto path, and — per §11 — has never been ablated to show it improves OOS results over the other 4 brains alone. Adversarial candidate for removal-and-test.
- **PPO meta-learner**: per `KNOWN_ISSUES.md`, zero real training data as of last update, will begin influencing live decisions once only 10 trades accumulate (a low bar), stacked on top of the adaptive learner, decision transformer, and hierarchical skills as one of *four* fallback decision sources tried in sequence. Four different "who's actually deciding" paths, gated by four different trade-count thresholds, is a lot of surface area for a ~2-month-old system with (per §13) not yet enough real trades to validate even one of them properly.
- **Population-based training / genetic evolution** (`population_trainer.py`, `evolution_cull.py`, `genetic_researcher.py`): exploits every 10 decisions off a rolling window of the last 50 returns, with a minimum-sample floor of only 20 for exploitation — Sharpe computed on n=20-50 is a noisy statistic, and this is a second, independent hyperparameter-search loop layered on top of everything else.

**Adversarial take:** none of this is necessarily wrong to have *built* — it's reasonable to build out a research platform ambitiously. But your framework's own final question — "if I removed every component that cannot demonstrate measurable OOS value, what would remain?" — currently has a stark answer for this repo: **the committee's static weighted vote plus the risk/execution/killswitch layer**, because that's the only part with enough live history and enforcement to have been exercised at all. Everything adaptive/learned (adaptive meta-learner, PPO, decision transformer, hierarchical skills, population trainer) is either still gated in shadow/insufficient-sample mode or, per §12, one config-default bug away from being live without having proven anything yet.

---

## 19. Kill Criteria

**No expectancy/profit-factor/consecutive-loss-based kill switch exists.** The only thing that actually halts live trading is the equity-based drawdown/daily-loss killswitch (`MAX_DRAWDOWN_STOP`/`DAILY_LOSS_LIMIT` → `liquidate_all`). The decay-alert thresholds that *could* serve this purpose (`SHARPE_DECAY_THRESHOLD`, `WIN_RATE_DECAY_THRESHOLD`, `PROFIT_FACTOR_DECAY_THRESHOLD`) are computed and logged but — consistent with the §0 pattern — **have zero callers that take any action.** This is a real, fixable gap: you already compute the numbers your framework's kill-criteria section asks for; nothing currently reads them and pauses trading.

**Enhancement candidate:** wire `get_decay_alerts()`'s existing output into the same alerting path that already fires for drawdown/exposure (`src/alerting.py`'s `run_monitoring_cycle()`), starting with alert-only (Telegram) and only escalate to an automatic risk-reduction/pause action once you trust it isn't noisy on small samples. *Adversarial pushback on my own suggestion:* per §13, you likely don't have 30+ real trades yet, so Sharpe/win-rate decay alerts computed on a tiny sample will be false-alarm-prone. Sequence matters — get the alert wired in now (cheap), but don't let it auto-pause trading until sample size actually supports the statistic, or you'll train yourself to ignore it.

---

## 20. The Most Important Questions — direct answers

- **Edge**: Five brains vote on RSI/regime-label/transformer-probability/sentiment thresholds; no documented reason any should retain edge post-discovery, and no in-repo evidence trail connecting live-default thresholds to any verification process.
- **Evidence**: None currently exists outside the live system's own (still sub-30-trade, per last recorded check) track record.
- **Robustness**: Partially testable (walk-forward and Monte Carlo tooling exist and are well-built) but not actually exercised against the parameters that matter (Hurst/RSI/ATR thresholds untouched by any tuner) and not run across multiple assets/regimes for the model-promotion path specifically.
- **Simplicity**: Unknown — no benchmark against a simpler baseline exists to answer this.
- **Failure**: Equity-based killswitch is real and will stop catastrophic drawdown; slower, "death by a thousand small edge-decay trades" failure has no automatic kill switch, only unread log lines.
- **Adaptation**: The mechanism to recognize invalidated assumptions (decay alerts, drift monitor) is built but not connected to any action.
- **Complexity**: A large fraction of the adaptive/learned layer (§18) cannot yet demonstrate OOS value because the track record is too young to validate it, by the system's own stated gates.
- **Honesty test**: A skeptical quant reviewer's first attack would be exactly §0 and §12 — "show me the safety default that's actually active in production right now, not what the README says it is" — and I could not fully answer that for `ADAPTIVE_ML_ENABLED` without you checking your actual deployment's environment directly.
- **Final question** ("what would remain if every OOS-unproven component were removed"): the committee's static weighted vote, the 5 brains as currently weighted, and the risk/execution/killswitch layer. Everything self-learning is currently unproven by the system's own gates.

---

## What I'd do first, in order, if these were approved (none are — awaiting your go-ahead)

1. **Verify your actual deployment's `ADAPTIVE_ML_ENABLED` value directly** (not a code change — just a fact-check on your live environment). This is the one item with a plausible "already trading on unvalidated weights right now" downside.
2. Confirm whether `data/historical_experiences.jsonl` is real or `REPLAY_FAST_MODE` noise (`scan_learning.py`, 2 minutes, no code change).
3. Fix the `train_foundation_model.py` scaler-fit-before-split leakage — small, contained change, directly affects any validation number you currently trust from that path.
4. Fix the `risk.py` `"mean_reverting"` vs `"sideways"` regime-label mismatch so the 0.8x multiplier is actually reachable — or confirm it's intentionally dead and remove it.
5. Reconcile `MAX_PORTFOLIO_VALUE`/`MAX_PORTFOLIO_PCT` documentation vs. actual effective cap in `.env.example`.
6. Wire `get_decay_alerts()` into existing alerting (alert-only, not auto-pause yet, per the sample-size caveat above).
7. Everything else in this document, roughly in the order it appears, weighted by how directly it touches money movement vs. observability/research tooling.

*No code was modified in the production of this report.*
