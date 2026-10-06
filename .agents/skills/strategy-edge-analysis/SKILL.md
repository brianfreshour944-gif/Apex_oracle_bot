---
name: strategy-edge-analysis
description: This skill should be used when the user asks to "check if the strategy has edge", "is this overfit", "analyze backtest validity", "check for look-ahead bias", or mentions walk-forward, out-of-sample, parameter sensitivity, or curve fitting in a Python trading bot.
triggers:
- strategy edge
- overfitting
- overfit
- look-ahead bias
- walk-forward
- out-of-sample
- curve fitting
- backtest validity
---

# Strategy Edge and Overfitting Analysis

Determine whether a trading strategy's measured performance reflects a real
edge or an artifact of fitting, bias, or metric misuse. A strategy that is
merely bug-free can still have no edge. This skill checks the second question.

Read-only. Do not trade, place or cancel orders, contact a live exchange, or
write state. Portable across repositories: discover the layout first.

## Step 0: discover the layout

```bash
# backtest and validation entry points
ls *backtest*.py src/*backtest*.py 2>/dev/null
grep -rliE "walk_forward|walkforward|out_of_sample|oos|train_test" --include="*.py" .
# parameter search and optimization
grep -rliE "optuna|hyperopt|grid_search|sweep|bayesian_opt" --include="*.py" .
# feature ablation and importance
grep -rliE "ablation|feature_importance|permutation" --include="*.py" .
# metric computation
grep -rliE "sharpe|sortino|calmar|profit_factor" --include="*.py" .
```

Record: the backtest entry point, any walk-forward harness, the parameter
search, and the metrics module. Reuse existing harnesses rather than writing
new ones. If a harness is missing, note it as a gap.

## Scope limits

Analyze only the strategy, backtest, and validation paths, or at most 20
files. Exclude `data/`, `models/`, `.venv/`, `*.jsonl`, `*.db`, `*.pth`,
`*.pkl`, and notebooks unless a finding requires one. Use bounded reads.

## What to check

**Look-ahead bias**
- Signal computed from a bar's close but filled at that same bar's price.
- Indicators using a centered window, `shift(-n)`, or future rows.
- Resampling that pulls later data into an earlier bucket.
- Normalization or scaling fit on the full dataset before splitting.
- Features that only exist after the decision point (e.g. end-of-day values
  used intraday).

**Overfitting and parameter sensitivity**
- Parameter count versus sample size. Many free parameters on little data is
  the primary overfitting signal.
- Performance collapse under small parameter perturbation. If a result only
  works at one precise value, it is fit, not edge.
- A large gap between in-sample and out-of-sample results.
- Optimization run on the same period used to report performance.
- Selection across many strategies or parameter sets without correction for
  multiple testing.

**Validation integrity**
- Walk-forward uses rolling refit, not a single split.
- Out-of-sample data was never touched during development.
- Embargo or purge gap between train and test to avoid leakage.
- The number of trials is recorded, so multiple-comparison correction is
  possible (deflated Sharpe, or similar).

**Bias in the data path**
- Survivorship bias: delisted or failed instruments excluded.
- Point-in-time correctness: fundamentals or constituents as known at the
  time, not restated.
- Corporate actions, splits, and dividends handled without introducing
  phantom returns.

**Cost realism**
- Slippage and commissions included and sized realistically.
- Fill assumptions (always filled at mid, no partial fills, unlimited
  liquidity) that inflate returns.
- Borrow or funding costs for short positions.

**Metric misuse**
- Sharpe computed on autocorrelated returns without adjustment.
- Annualization factor wrong for the bar frequency.
- Risk-free rate ignored or misapplied.
- Returns computed on equity without accounting for deposits or withdrawals.
- Profit factor or win rate reported without the loss distribution.

## Method

1. Identify the exact command that produced the reported performance and
   confirm it is reproducible. If not, that is the first finding.
2. Compare in-sample and out-of-sample results where both exist.
3. Where the harness allows, run a bounded parameter perturbation and report
   how fast performance degrades. Keep runs small and read-only.
4. Check the code for the look-ahead patterns above, citing file and line.
5. Count the effective number of trials if recorded.
6. State plainly whether the evidence supports an edge, is inconclusive, or
   indicates fitting.

## Output

Use the repository's finding format if defined (for example a shared contract
in `AGENTS.md`). Otherwise:

```text
Finding ID: EDGE-<short-id>
Claim: <one falsifiable sentence>
Severity: CRITICAL | HIGH | MEDIUM | LOW
Category: look-ahead | overfitting | validation | data-bias | cost | metric
Evidence: <file:line and observed output>
Reproduction: <exact safe command>
Expected observation: <required result>
Actual observation: <observed result, or NOT_REPRODUCED>
Confidence: <0.0-1.0>
```

End with a verdict:

```text
EDGE SUMMARY
Strategy/scope: <name and files>
In-sample result: <metric and value, or UNKNOWN>
Out-of-sample result: <metric and value, or UNKNOWN>
Effective trials: <count or UNKNOWN>
Look-ahead findings: <IDs or NONE>
Overfitting findings: <IDs or NONE>
Commands executed: <read-only commands>
Not covered: <limitations>
EDGE VERDICT: SUPPORTED | INCONCLUSIVE | LIKELY_FITTED
```

`SUPPORTED` requires out-of-sample evidence and no unaddressed look-ahead
finding. `LIKELY_FITTED` when in-sample and out-of-sample diverge materially,
parameters are fragile, or a look-ahead path is confirmed. Otherwise
`INCONCLUSIVE`.

## Rules

- A high backtest return is not evidence of edge. Require out-of-sample proof.
- Never report a number you did not reproduce or clearly mark it as claimed.
- Do not optimize, tune, or change parameters. This is analysis only.
- Never edit files, trade, or mutate data.
- Keep runs bounded; a context or time limit is a stop condition, not a pass.
