---
name: financial-analysis
description: This skill should be used when the user asks to "analyze the financials", "check the risk logic", "review PnL or position sizing", "audit execution costs", or mentions drawdown, exposure, slippage, or backtest/live parity in a trading bot.
triggers:
- financial analysis
- analyze the financials
- pnl
- position sizing
- risk logic
- drawdown
- slippage
- backtest parity
---

# Financial Analysis

Read-only review of the money paths in a Python trading bot: sizing, risk
limits, PnL accounting, execution cost, and backtest/live parity. Produce
evidence-backed findings. Do not trade, place or cancel orders, contact a
live exchange, or write database state.

This skill is portable across repositories. Discover the layout before
analyzing; never assume file names.

## Step 0: discover the layout

Run these searches and record what you find before analyzing anything:

```bash
# risk and sizing
grep -rliE "position_size|position_sizing|risk_per_trade|max_drawdown|kelly" --include="*.py" .
# PnL, metrics, attribution
grep -rliE "realized_pnl|unrealized|sharpe|sortino|attribution" --include="*.py" .
# execution and costs
grep -rliE "slippage|commission|execution|fill_price|order" --include="*.py" .
# backtest entry points
ls *backtest*.py src/*backtest*.py 2>/dev/null
```

Then determine:

- The test runner: read `pyproject.toml`, `pytest.ini`, `setup.cfg`, or `tox.ini`.
- The risk module, the metrics/PnL module, and the execution module.
- Any existing audit harnesses (`audit_*.py`, `scripts/audit_*.py`,
  `measure_*.py`, `verify_*.py`) and reuse them instead of writing new ones.

If a category has no matching module, say so and skip it. Do not invent paths.

## Scope limits

- Analyze only the paths in the task, or at most 20 source/test files.
- Exclude `data/`, `models/`, `.venv/`, `.git/`, notebooks, `*.jsonl`, `*.db`,
  `*.pth`, `*.pkl`, and generated reports unless a finding requires one.
- Use bounded line reads for large files.
- If scope is too broad, stop and ask for a narrower one.

## What to check

**Position sizing and risk limits**
- Sizing can exceed the configured max position or account equity.
- Risk-per-trade is computed from stale or unvalidated price.
- Stop-loss and take-profit can invert or be unreachable.
- Drawdown or daily-loss limits are checked but not enforced on the live path.
- Correlation or concentration limits are missing where multiple positions open.

**PnL and accounting**
- Fees and slippage omitted from realized PnL.
- Long and short PnL signs handled inconsistently.
- Partial fills or partial exits mis-accounted.
- Rounding or float drift accumulating across many trades.

**Execution cost**
- Market orders sized without checking liquidity or spread.
- Limit price set from a stale quote.
- Retry logic that can double-submit an order.

**Backtest/live parity**
- Indicators computed differently in backtest and live (lookback, warmup,
  resampling, or fill assumptions).
- Survivorship or look-ahead bias in the backtest data path.
- Same parameters produce materially different results between the two paths.

## Output

Use the repository's existing finding format if one is defined (for example a
shared contract in `AGENTS.md`). Otherwise use:

```text
Finding ID: FIN-<short-id>
Claim: <one falsifiable sentence>
Severity: CRITICAL | HIGH | MEDIUM | LOW
Area: risk | pnl | execution | parity
Evidence: <file:line and observed output>
Reproduction: <exact safe command or read-only query>
Expected observation: <required result>
Actual observation: <observed result, or NOT_REPRODUCED>
Confidence: <0.0-1.0>
```

End with:

```text
FINANCIAL SUMMARY
Scope inspected: <paths>
Modules found: risk=<path> metrics=<path> execution=<path>
Commands executed: <read-only commands>
Not covered: <limitations>
Overall risk: <summary>
```

## Rules

- Never place or cancel an order, contact a live endpoint, or mutate real data.
- A finding needs a reproduction or must be labeled unproven.
- Static reading alone does not prove runtime behavior; state the limitation.
- Never include credentials, tokens, or account identifiers in output.
- Do not edit files. Report findings; fixing is a separate role.
