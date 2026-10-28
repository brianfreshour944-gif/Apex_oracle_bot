# Validation-data manifest (recorded BEFORE any strategy run)

Window: 2021-01-01T06:00:00Z .. 2025-09-30T23:00:00Z (spec §3 conventions)
Cache: `data/ohlcv_dev2021_2025/` (separate from the development cache; `data/` is
gitignored). Fetched via `scripts/fetch_ohlcv.py::fetch_symbol` — the project's
existing fetcher, data-only, no orders.

## Files, checksums, integrity

| File | Bars | First bar | Last bar | Dups | Non-monotonic | Bad OHLC | Non-positive px | sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC_USD_1Hour.json | 41,597 | 2021-01-01T06:00Z | 2025-09-30T23:00Z | 0 | 0 | 0 | 0 | `71a838b7533e887aa241f3e280d2914d479f5334d62ae3c2ddacd2af5ac789a7` |
| ETH_USD_1Hour.json | 41,596 | 2021-01-01T06:00Z | 2025-09-30T23:00Z | 0 | 0 | 0 | 0 | `073fd25a7295e08f2db797d5323058a4d5fc7e5dbcf9fed801a50ed7c4f85ddc` |
| SOL_USD_1Hour.json | 31,521 | 2021-01-01T06:00Z | 2025-09-30T23:00Z | 0 | 0 | 0 | 0 | `b415e525fe17432f0f4560e06c95a92687952b7f29a21e018872e27321e2ddf7` |

All three pass: monotonic timestamps, zero duplicates, valid OHLC
(`h ≥ max(o,c)`, `l ≤ min(o,c)`, `h ≥ l`), no non-positive prices, exact window
boundaries (first 2021-01-01T06:00Z, last 2025-09-30T23:00Z).

## Gaps

| Symbol | Gap sizes > 1 h | Total missing hours |
| --- | --- | --- |
| BTC_USD | {2:1, 3:1, 4:1, 8:1} | 13 |
| ETH_USD | {3:2, 4:1, 8:1} | 14 |
| SOL_USD | {2:1, 4:1, 6:1, 8:1, 62:1, **10013:1**} | **10,089** |

BTC/ETH have only trivial exchange gaps (≤8 h). **SOL has one enormous hole:
2023-07-06T11:00Z → 2024-08-26T16:00Z (10,013 h ≈ 417 days), plus a 62 h gap on
2023-06-23..26.**

## SOL coverage is materially worse — verified against the source

I did not assume the hole was a fetch artifact. Read-only `limit`-capped probes of
the Alpaca bars endpoint **inside** the gap (2023-08-01, 2023-10-01, 2023-12-01,
2024-02-01, 2024-05-01, 2024-08-01) all return **EMPTY**, while the bars
immediately before (2023-07-05) and after (2024-08-26T16:00Z) exist. So Alpaca
genuinely has no SOL/USD hourly bars in that window; the hole is real, not a
paging error. SOL is missing ~24% of the 4.74-year window.

## Pre-outcome decision (made before any strategy run)

Spec §1: *"If any symbol's validation coverage is materially worse than the
others, it is documented (before outcomes) and excluded; the primary result is
reported per symbol and pooled."*

Applying that rule as written:

- **SOL's coverage IS materially worse** (10,013 h hole vs ≤14 h for BTC/ETH).
- **SOL is documented here, before outcomes, and excluded from the primary
  pooled aggregate.**
- The **primary pooled result = BTC + ETH** (the two symbols with comparable
  coverage).
- SOL is still **reported per symbol** (the spec says report per symbol), with its
  gap disclosed, and a **secondary pooled-all-three** figure is shown for
  completeness.
- No strategy outcome was inspected to make this decision; it rests solely on the
  data-integrity fact above.

The §1 wording ("excluded … and pooled") admits a second reading (exclude only
from consideration, pool all three). Both poolings are reported so the operator
can apply either; the BTC+ETH pooling is treated as primary because it is the
reading under which "excluded" has operative effect.

## Reproduce

```bash
python3 - <<'PY'
import sys, json; sys.path.insert(0,"scripts")
from pathlib import Path
from fetch_ohlcv import fetch_symbol
OUT = Path("data/ohlcv_dev2021_2025"); OUT.mkdir(parents=True, exist_ok=True)
for sym in ("BTC/USD","ETH/USD","SOL/USD"):
    bars = fetch_symbol(sym, "2021-01-01", "2025-09-30", "1Hour")
    (OUT / f"{sym.replace('/','_')}_1Hour.json").write_text(
        json.dumps({"symbol": sym, "timeframe": "1Hour", "bars": bars}))
PY
```
