#!/usr/bin/env python3
"""Report trades NEW since the 2026-10-24 baseline snapshot. Read-only.

Splits decision_snapshots at CUTOFF -- the newest closed_at in the local
24-trade snapshot -- and reports the newer closed trades: overall, by symbol,
by exit reason, by regime, plus their orders and vote splits.

Usage:
    python delta_since_baseline.py /tmp/bot_baseline.db
    python delta_since_baseline.py /tmp/bot_baseline.db --json out.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3

# Newest closed_at in the local 2026-10-24 snapshot (24 closed trades).
# Anything strictly newer is new production activity.
CUTOFF = "2026-10-24 04:21:08.380985"
EXPECTED_OLD = 24


def _ro(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _pf(pnls: list[float]) -> float | None:
    gp = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    return None if gl <= 0 else gp / gl


def _max_dd_pct(equity: list[float]) -> float:
    peak = None
    worst = 0.0
    for e in equity:
        peak = e if peak is None else max(peak, e)
        if peak:
            worst = min(worst, (e - peak) / peak)
    return worst * 100.0


def _summ(rows: list[dict]) -> dict:
    pnls = [r["realized_pnl"] or 0.0 for r in rows]
    eq, run = [], 0.0
    for p in pnls:
        run += p
        eq.append(run)
    wins = sum(1 for p in pnls if p > 0)
    return {
        "trades": len(rows),
        "wins": wins,
        "losses": sum(1 for p in pnls if p < 0),
        "win_rate": (wins / len(rows) * 100) if rows else 0.0,
        "net": round(sum(pnls), 4),
        "gross_profit": round(sum(p for p in pnls if p > 0), 4),
        "gross_loss": round(sum(p for p in pnls if p < 0), 4),
        "pf": _pf(pnls),
        "max_dd_pct": round(_max_dd_pct(eq), 2) if rows else 0.0,
    }


def _bucket(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r[key] or "(none)", []).append(r)
    return out


def split_at_cutoff(rows: list[dict], cutoff: str) -> tuple[list[dict], list[dict]]:
    """Split closed trades into (at/before cutoff, strictly after).

    The boundary row itself counts as old, so re-running against an unchanged
    database reports zero new trades rather than re-reporting the last one.
    """
    old = [r for r in rows if (r["closed_at"] or "") <= cutoff]
    new = [r for r in rows if (r["closed_at"] or "") > cutoff]
    return old, new


def _fmt_pf(pf: float | None) -> str:
    return "n/a" if pf is None else f"{pf:.3f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("db", help="path to the snapshot copy (opened read-only)")
    ap.add_argument("--json", metavar="PATH", help="also write results as JSON")
    ap.add_argument("--cutoff", default=CUTOFF, help=f"split timestamp (default {CUTOFF})")
    args = ap.parse_args(argv)

    conn = _ro(args.db)
    try:
        cur = conn.execute(
            "SELECT decision_id, symbol, regime, exit_reason, realized_pnl, "
            "return_pct, holding_period_sec, closed_at, votes_json, entry_price, qty "
            "FROM decision_snapshots WHERE lower(status)='closed' ORDER BY closed_at"
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]

        cur = conn.execute(
            "SELECT order_id, decision_id, symbol, side, qty, filled_avg_price, "
            "commission, status, filled_at FROM orders"
        )
        ocols = [d[0] for d in cur.description]
        orders = [dict(zip(ocols, r, strict=False)) for r in cur.fetchall()]

        all_ids = {r[0] for r in conn.execute("SELECT decision_id FROM decision_snapshots")}
    finally:
        conn.close()

    old, new = split_at_cutoff(rows, args.cutoff)
    new_ids = {r["decision_id"] for r in new}

    print(f"snapshot     : {args.db}")
    print(f"cutoff       : {args.cutoff}")
    print(f"closed total : {len(rows)}")
    print(f"  at/before  : {len(old)}   (expected {EXPECTED_OLD})")
    print(f"  NEW        : {len(new)}")
    if len(old) != EXPECTED_OLD:
        print(f"  WARNING: expected {EXPECTED_OLD} older trades, found {len(old)};")
        print("           the cutoff may not align with this database.")

    if old and new:
        print(f"\nboundary     : last old closed_at={old[-1]['closed_at']}")
        print(f"               first new closed_at={new[0]['closed_at']}")

    if not new:
        print("\n=> No trades newer than the cutoff. Nothing new since 2026-10-24.")
        return 0

    s = _summ(new)
    print("\n=== NEW TRADES: OVERALL ===")
    print(f"  trades={s['trades']}  wins={s['wins']}  losses={s['losses']}  "
          f"win_rate={s['win_rate']:.1f}%")
    print(f"  net=${s['net']:+.4f}  gross_profit=${s['gross_profit']:+.4f}  "
          f"gross_loss=${s['gross_loss']:+.4f}")
    print(f"  PF={_fmt_pf(s['pf'])}  maxDD={s['max_dd_pct']:.2f}%")

    print("\n=== NEW TRADES: BY SYMBOL ===")
    for k, grp in sorted(_bucket(new, "symbol").items()):
        g = _summ(grp)
        print(f"  {k:10s} trades={g['trades']:3d}  win_rate={g['win_rate']:5.1f}%  "
              f"net=${g['net']:+.4f}  PF={_fmt_pf(g['pf'])}")

    print("\n=== NEW TRADES: BY EXIT REASON ===")
    for k, grp in sorted(_bucket(new, "exit_reason").items()):
        g = _summ(grp)
        print(f"  {k[:52]:54s} trades={g['trades']:3d}  net=${g['net']:+.4f}")

    print("\n=== NEW TRADES: BY REGIME ===")
    for k, grp in sorted(_bucket(new, "regime").items()):
        g = _summ(grp)
        print(f"  {k:16s} trades={g['trades']:3d}  net=${g['net']:+.4f}  PF={_fmt_pf(g['pf'])}")

    print("\n=== NEW TRADES: EACH ===")
    for r in new:
        hp = r["holding_period_sec"]
        hp_s = f"{hp/3600:.2f}h" if isinstance(hp, (int, float)) else "n/a"
        print(f"  {r['decision_id'][:12]} {r['symbol'] or '?':8s} "
              f"{r['regime'] or '?':15s} "
              f"pnl=${(r['realized_pnl'] or 0):+.4f} hold={hp_s:>7s} "
              f"exit={(r['exit_reason'] or '?')[:34]}")

    print("\n=== NEW TRADES: RETURN_PCT ANOMALIES (|return_pct| > 100) ===")
    flagged = [r for r in new if r["return_pct"] is not None and abs(r["return_pct"]) > 100]
    if not flagged:
        print("  none")
    for r in flagged:
        print(f"  {r['decision_id'][:12]} {r['symbol']:8s} pnl=${(r['realized_pnl'] or 0):+.4f} "
              f"return_pct={r['return_pct']:.2f}%  <- bad denominator, not a real loss")

    new_orders = [o for o in orders if o["decision_id"] in new_ids]
    comm = sum(o["commission"] or 0.0 for o in new_orders)
    print("\n=== NEW TRADES: ORDERS ===")
    print(f"  orders linked to new decisions : {len(new_orders)}")
    print(f"  orders per new trade           : {len(new_orders)/len(new):.2f}")
    print(f"  commission on those orders     : ${comm:.4f}  (provenance: unknown)")
    dist: dict = {}
    for o in new_orders:
        dist[o["decision_id"]] = dist.get(o["decision_id"], 0) + 1
    print(f"  orders-per-decision            : {sorted(dist.values(), reverse=True)}")
    print("  NOTE: multiple orders per decision are partial exits (pre-existing).")
    print(f"  orders with no decision_id     : {sum(1 for o in orders if not o['decision_id'])}")
    print(f"  orders referencing unknown id  : {sum(1 for o in orders if o['decision_id'] and o['decision_id'] not in all_ids)}")

    print("\n=== NEW TRADES: VOTE SPLITS ===")
    tally: dict = {}
    for r in new:
        if not r["votes_json"]:
            continue
        try:
            v = json.loads(r["votes_json"])
        except (TypeError, ValueError):
            continue
        if not isinstance(v, dict):
            continue
        for brain, action in v.items():
            tally.setdefault(brain, {}).setdefault(action, 0)
            tally[brain][action] += 1
    for brain in sorted(tally):
        print(f"  {brain:12s} {tally[brain]}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {"cutoff": args.cutoff, "closed_total": len(rows),
                 "old": _summ(old), "new": _summ(new),
                 "new_by_symbol": {k: _summ(v) for k, v in _bucket(new, "symbol").items()},
                 "new_by_exit": {k: _summ(v) for k, v in _bucket(new, "exit_reason").items()},
                 "new_by_regime": {k: _summ(v) for k, v in _bucket(new, "regime").items()}},
                fh, indent=2, default=str,
            )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
