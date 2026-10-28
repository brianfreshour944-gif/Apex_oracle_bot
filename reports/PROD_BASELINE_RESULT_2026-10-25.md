# PRODUCTION BASELINE RESULT — 2026-10-25 (NEW DATA CONFIRMED)

Read-only run executed by the operator on the host. Script hash verified in the
container before execution (`591cab03…3f8ec`). Nothing was changed, restarted, or
deployed.

## Result: DIFFERS — new trading data exists, and it is more loss

| Metric | 2026-10-25 baseline | Production now | Delta |
| --- | --- | --- | --- |
| Closed trades | 24 | **34** | **+10** |
| Orders | 84 | **110** | **+26** |
| Filled | 84 | **110** | **+26** |
| Net P&L | −2.1053 | **−3.8688** | **−1.7635** |

Integrity check: `ok`. The ~95 KB file-size difference flagged earlier was real
new data, not SQLite page noise.

## What the ten new trades did

- New net P&L: **−$1.7635** over 10 trades
- Average per new trade: **−$0.1763**
- Old 24-trade average: −$0.0877
- **The new trades lost at roughly 2.0× the previous rate.**

The losing streak is not flattening as the sample grows. Cumulative average per
trade went from −$0.0877 to −$0.1138.

## Order accounting: 2.6 orders per trade

26 new orders for 10 new closed trades. If each trade is one entry plus one exit,
20 would be expected. **Six orders are unaccounted for.**

Possible explanations (not distinguished by this run):
- partial exits (a position closed in more than one fill) — this would fit the
  known dust-partial-exit accounting anomalies;
- entries that never produced a closed decision row;
- cancelled/rejected orders. Note `orders == filled == 110`, so no non-filled
  status is recorded at all — every row is `filled`, which means cancelled orders
  are either absent or mislabelled.

This is a concrete lead for the accounting investigation.

## Schema evidence: the deployed code is not the instrumented branch

Observed `orders` columns in production:

```
order_id, decision_id, symbol, side, qty, filled_qty, filled_avg_price,
commission, status, type, time_in_force, client_order_id, submitted_at, filled_at
```

`mid_at_submit`, `spread_at_submit`, `quote_ts`, `fee_source` are **absent**.

- Those columns exist at `HEAD` (w1-instrumentation-accounting) — 5 references in `src/db.py`.
- They are **absent at `main`** (7a8ca12) — 0 references.
- `_ensure_additive_columns()` in `src/db.py:875` adds them automatically on startup,
  idempotently, if the running code defines them.

So the deployed code does **not** define those columns. That is **consistent with
`main` (7a8ca12) being deployed**, and rules out any `w1` commit that includes the
quote instrumentation.

**Caveat — this is inference, not proof of the branch.** A deployed commit could be
any revision predating those columns. It does not establish the exact commit, and it
should not be reported as "main is deployed" without the Coolify Source panel. What
it does establish: **quote capture is not active in production**, so no measured
spread/mid data exists for slippage analysis.

## What this does and does not establish

**Establishes:**
- 10 new closed trades since 2026-10-25, netting −$1.7635.
- The loss rate per trade roughly doubled.
- Production records no quote instrumentation.
- Order-to-trade ratio is 2.6, with 6 orders unexplained.

**Does not establish:**
- Whether the P&L figures are net of real fees. All `commission` values are recorded
  but their provenance (exchange charge vs model estimate) is still unknown, and this
  run did not trace it.
- Whether `realized_pnl` is correctly computed — the earlier `return_pct` blowups
  (−804.76% ETH, −70218.31% SOL) from dust partial exits remain unreconciled.
- Any trading edge. 34 trades is still a small sample.

## Recommended next step

Copy the snapshot out and run the analyzer:

```bash
docker cp <container>:/tmp/bot_baseline.db /tmp/bot_baseline.db
```

Then, on a machine with the analyzer (it is on w1, not main — copy it the same way):

```bash
python scripts/analyze_deployed_trades.py /tmp/bot_baseline.db --json out.json
```

The immediate questions: which symbols and exit reasons the 10 new trades came from,
and what the 6 extra orders are.

## Not authorized / not done

No merge, no cherry-pick, no push, no redeploy, no restart, no config change. No
change to `CAPTURE_SUBMISSION_QUOTE`. No trading-behavior change.
