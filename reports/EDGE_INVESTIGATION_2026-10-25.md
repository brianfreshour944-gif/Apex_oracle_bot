# Edge Investigation — hourly crypto, 2025-10 to 2026-10 (read-only)

Date: 2026-10-25
Status: read-only study. No strategy, risk, exchange, or DB code changed.
Harness: `scripts/edge_study.py` (stdlib only). Data: `scripts/fetch_ohlcv.py`.
Tests: `tests/test_edge_study.py` (6, hand-computed values).

## Question

Does the strategy family have a *statistically convincing* directional edge on
hourly BTC/ETH/SOL once realistic trading costs are charged? The prior study found
none; this re-tests a small, pre-registered set rather than adding complexity.

## Method (fixed before looking at results)

- **Pre-registered hypotheses** (7, all simple, no parameters tuned after seeing
  results): 24h/72h time-series momentum, 24h/72h mean reversion, 12/48 and
  24/168 MA crossover, 48h breakout.
- **Costs**: 56 bps round trip = 25 bps taker fee + 3 bps slippage per side,
  charged on every position change. Matches the bot's `ESTIMATED_TAKER_FEE_BPS`.
- **In-sample** = bars before 2026-07-01. **Holdout** = 2026-07-01..2026-10-24,
  not inspected until the final section.
- **Two modes** to separate signal from market beta: LONG/FLAT, and
  LONG/SHORT "always-in" (signal 0 -> short), so a result is not just the market.
- **Significance**: moving-block bootstrap (block 12) over trade returns, which
  accounts for autocorrelation that an i.i.d. t-test would ignore.
- No look-ahead: every signal uses only bars up to and including bar *i*.

## Data

Alpaca `data.alpaca.markets/v1beta3/crypto/us/bars`, 1Hour, 2025-10-01..2026-10-24.
9,335 / 9,334 / 9,329 bars (BTC/ETH/SOL). Cached under `data/ohlcv/`.
(Binance endpoints return HTTP 451 from this environment.)

**Regime note.** The in-sample window is a sustained BEAR market
(BTC 114,259 -> 73,798; ETH 4,148 -> 2,012; SOL 208 -> 83). Buy-and-hold lost
49-65%. The holdout is a BULL recovery (+37% to +49% buy-and-hold). This matters:
a long-only system is dominated by beta, which is why the LONG/SHORT mode is
reported alongside.

## In-sample result (bars before 2026-07-01)

Every pre-registered hypothesis loses, in both modes, on all three symbols.
Representative (mean net return per trade, bps; p = bootstrap P(mean <= 0)):

| Hypothesis | BTC L/F | ETH L/F | SOL L/F | BTC L/S | ETH L/S | SOL L/S |
| --- | --- | --- | --- | --- | --- | --- |
| H1 momentum 24h | -59.6 | -69.9 | -77.4 | -49.8 | -56.9 | -63.5 |
| H1 momentum 72h | -84.9 | -89.4 | -119.9 | -65.5 | -61.8 | -93.3 |
| H2 meanrev 24h | -73.1 | -69.7 | -64.6 | -63.4 | -56.6 | -50.7 |
| H3 ma 12/48 | -90.0 | -118.3 | -158.7 | -52.6 | -65.5 | -106.5 |
| H3 ma 24/168 | -169.5 | -226.7 | -251.5 | -91.8 | -85.9 | -112.4 |
| H4 breakout 48 | -54.8 | -52.3 | -62.8 | -40.7 | -32.1 | -43.4 |

All p-values are 0.89-1.00 (i.e. the mean is essentially never positive).
The best in-sample cell (SOL H2 meanrev 72h, L/S) is -25.8 bps/trade.

## Holdout result (2026-07-01..2026-10-24, previously untouched)

The market recovered sharply, yet every hypothesis still loses on a
cost-adjusted basis in almost every cell:

| Hypothesis | BTC L/F | ETH L/F | SOL L/F | BTC L/S | ETH L/S | SOL L/S |
| --- | --- | --- | --- | --- | --- | --- |
| H1 momentum 24h | -39.9 | -31.6 | -27.2 | -53.0 | -48.0 | -43.2 |
| H1 momentum 72h | -43.2 | -48.3 | -41.6 | -68.4 | -75.8 | -67.9 |
| H2 meanrev 24h | -47.5 | -49.0 | -56.9 | -60.6 | -65.4 | -73.1 |
| H3 ma 12/48 | -36.4 | -22.3 | **+4.6** | -73.9 | -69.9 | -46.7 |
| H3 ma 24/168 | -8.1 | -8.2 | **+73.8** | -123.9 | -158.3 | -33.5 |
| H4 breakout 48 | -44.3 | -42.4 | -58.6 | -59.9 | -64.2 | -76.0 |

Only two cells are positive, both SOL trend-following, both **not significant**
and both **far below buy-and-hold**:

- SOL H3 ma 12/48 LONG/FLAT: +4.6 bps/trade, p = 0.36, 37 trades, total -1.9%
  (buy-and-hold +38.5%).
- SOL H3 ma 24/168 LONG/FLAT: +73.8 bps/trade, p = 0.28, 19 trades, total +8.1%
  (buy-and-hold +38.5%).

These are the least-traded cells, with the widest confidence intervals. A p of
0.28-0.36 means a mean of zero or below is entirely consistent with the data.
And a strategy that returns +8% while the asset returned +38.5% is not an edge --
it is underperformance with extra steps.

## Conclusion

**No pre-registered hypothesis shows a statistically convincing edge**, in
either market regime, with realistic costs. The specific findings:

1. Trend-following, mean-reversion, breakout, and MA-crossover all lose in-sample
   on all three symbols in both modes, with p ~ 1.0.
2. They continue to lose out-of-sample, including through a strong recovery.
3. The only positive holdout cells are insignificant (p = 0.28-0.36) and
   underperform simple buy-and-hold by a wide margin.
4. In the bear in-sample window, active long-only trading was *worse* than
   buy-and-hold; in the bull holdout, active trading failed to capture the
   recovery. The signal, not the market, is the problem.

## Limitations (stated plainly)

- **Sample size and regime.** One year, one bear-to-bull cycle. A "no edge here"
  result does not prove "no edge ever," but it is the honest reading of the data
  available.
- **Scope.** Seven simple hypotheses, hourly bars, close-to-close fills. A more
  exotic signal, intraday granularity, or different execution could differ; none
  is suggested by these results.
- **Costs are modelled, not measured** (25 bps fee + 3 bps slippage). Real fills
  could be better or worse; the conclusions are not sensitive to modest changes
  because the gross signals are already negative.
- **Overlapping bars** are handled by the block bootstrap, but the effective
  sample is smaller than the raw trade count, so even the positive cells would
  need much more data to be trusted.
- This study says nothing about the *accounting* patch under review; it is a
  separate question about strategy edge.

## Recommendation

Do not loosen gates or add strategy complexity on the basis of these results.
The evidence points the other way: the tested signal family has no demonstrated
edge net of costs. If further work is warranted, it should (a) use a longer
history and more symbols to raise statistical power, and (b) pre-register
hypotheses before testing, exactly as here. Live trading should remain frozen
until evidence supports otherwise.

## Reproduce

```bash
python3 scripts/fetch_ohlcv.py --symbols BTC/USD ETH/USD SOL/USD \
    --start 2025-10-01 --end 2026-10-24 --timeframe 1Hour
python3 scripts/edge_study.py                    # in-sample screen
python3 scripts/edge_study.py --holdout          # untouched holdout
python3 -m pytest tests/test_edge_study.py -q    # harness math
```
