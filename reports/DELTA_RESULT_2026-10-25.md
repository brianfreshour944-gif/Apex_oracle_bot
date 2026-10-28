# DELTA RESULT — THE 10 NEW TRADES (2026-10-25)

Read-only run by the operator on the host against `/tmp/bot_baseline.db`
(hash-verified `delta_since_baseline.py`). Nothing changed, restarted, or deployed.

## The headline

**All 10 new trades are SOL/USD. All 10 lost. None won.**

| Metric | Value |
| --- | --- |
| Trades | 10 |
| Wins | **0** |
| Net P&L | **−$1.7635** |
| Gross profit | **$0.0000** |
| Profit factor | **0.000** |
| Symbols | **SOL/USD only (10/10)** |

A 0/10 run. If each trade were a coin flip, the probability of ten straight
losses is ~0.1%. This is not noise — but the cause is visible in the data, and it
is not (yet) proof that the strategy has no edge.

## The dominant fact: 9 of 10 exited in 0.50 hours

| Exit reason | Trades | Net |
| --- | --- | --- |
| `[TREND_FOLLOWING] Macro trend flipped bearish` | **9** | −$1.5881 |
| `[BREAKOUT] Breakout failed / retraced` | 1 | −$0.1754 |

**Every trade held exactly 0.50h** (1800 s). That is a suspiciously round number
for ten independent trades.

The exit fires from `src/execution_strategies.py:41`:

```python
if qty > 0 and htf_trend == "bearish":
    return {"action": "close", "reason": "Trend Following: Macro trend flipped bearish"}
```

So the pattern is: enter SOL long → the higher-timeframe trend reads bearish on
the next evaluation → close at a small loss. Ten times in a row, all within 30
minutes.

**Regimes:** 8 `neutral`, 2 `bear`. The bot was buying into a neutral/bearish
higher-timeframe trend and being stopped out by its own trend filter almost
immediately.

This is the most actionable finding so far: the entry and the exit filter
disagree. The committee (`quant` voted **buy 10/10**, `transformer` **buy 10/10**)
opened longs, and the trend-following exit closed them on the very next check.

## Vote splits — the committee was not divided

| Brain | Votes (10 trades) |
| --- | --- |
| quant | **buy 10** |
| transformer | **buy 10** |
| sentinel | hold 10 |
| llm | hold 9, sell 1 |
| momentum | stand_aside 8, sell 2 |

`quant` and `transformer` voted buy unanimously; `sentinel` held on all ten. The
entry decision was **not** the result of a split committee — it was a consistent
signal. The failure is in what happened after entry.

## Fees are 67% of the loss

| | |
| --- | --- |
| Net P&L | −$1.7635 |
| Commission on those orders | **$1.1861** |
| **Fees as share of loss** | **67.3%** |
| Gross before fees | −$0.5774 |

**Two thirds of the recorded loss is fees.** The gross loss is only −$0.58.

At the configured 25 bps per side, a round trip is 50 bps. Observed fee per trade
is $0.1186, implying a notional of roughly **$23.72** — a very small position.

### Critical caveat on those fees

`src/config.py:418` and `src/exchange.py:1030` both state plainly:

> Alpaca's paper/adapter reports commission 0.0 (alpaca-py>=0.43 Order has no
> .commission attribute), so the real fee is estimated from this rate…

`_apply_estimated_commission()` computes `filled_avg_price * filled_qty * 25bps`
and flags it `commission_estimated: True`. **These are modelled fees, not
exchange-reported charges.**

So the 67% figure is "share of loss attributable to *modelled* fees." If the true
Alpaca crypto fee differs, so does this number. It is a strong lead, not a
certified cost.

## A material code difference: production is not the audited branch

Production's `orders` table has **no** `mid_at_submit`, `spread_at_submit`,
`quote_ts`, or `fee_source` columns — those exist at `HEAD` (w1) and are added
automatically by `_ensure_additive_columns()` **if the running code defines
them**. They are absent, so the deployed code does not define them.

More importantly, the two branches compute P&L **differently**:

**`main` (7a8ca12)** — `src/bot.py:169`:
```python
realized_pnl = (exit_price - entry_price) * qty
realized_pnl -= exit_fee + entry_fee          # single exit leg
```
`reconstruct_position_outcome` does **not exist** at `main`.

**`HEAD`/w1** — `src/db.py:474` `reconstruct_position_outcome()` rebuilds from
**every** linked fill:
```
realized_pnl = cash_in_on_sells - cash_out_on_buys - entry_fees_in_quote
```
Its docstring records the exact bug this fixes:

> The previous implementation closed the snapshot on the FIRST exit fill and
> computed P&L from that one leg's quantity, which charged the whole position's
> entry fee against a dust-sized gross and could even flip the sign of the
> result (2026-10-25 accounting audit).

**The production numbers are therefore computed by the older, known-buggy
single-leg method.** The `return_pct` blowups (−804.76% ETH, −70218.31% SOL)
already traced to a dust-residual denominator are consistent with exactly this.

I verified the withholding convention empirically on the local snapshot:
`sell_qty / buy_qty == 0.9975` exactly (= 1 − 25 bps) on the fully-closed rows —
confirming the entry fee is withheld in the base asset, which the w1 code handles
and the `main` code does not.

## What this does and does not establish

**Establishes:**
- 10/10 losses, all SOL, all ~0.50h, 9 via the trend-flip exit.
- 67% of the loss is modelled fees.
- The committee voted buy 10/10 — entry was not a split decision.
- Production computes P&L by the single-leg method the w1 branch replaced.

**Does not establish:**
- That the strategy has no edge. A 0/10 run on one symbol is a strong signal, not
  proof, and the sample is tiny.
- The true cost, since fees are modelled, not observed.
- Whether the ~0.50h exits are a genuine trend reversal or an over-sensitive
  filter. **This is the next thing to test.**
- The exact deployed commit (still unverified; see the branch/commit caveat).

## Recommended next step

Do **not** change strategy yet. Two cheap, read-only checks:

1. **Why exactly 0.50h?** Nine identical holding times suggests the trend filter
   is evaluated on a fixed cadence and flipped on the first check. Trace
   `htf_trend` at those ten timestamps. If the filter reads bearish immediately
   after a buy, entry and exit are contradictory.
2. **Is the ~$23 notional intended?** If positions are that small, 50 bps of
   round-trip cost dominates any plausible edge — the bot may be structurally
   unable to profit regardless of signal quality.

Both are answerable from the data already on the host. Neither requires a code
change, a restart, or a deploy.

## Not authorized / not done

No merge, no cherry-pick, no push, no redeploy, no restart, no config change. No
change to `CAPTURE_SUBMISSION_QUOTE`. No trading-behavior change.
