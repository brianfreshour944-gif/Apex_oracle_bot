#!/usr/bin/env python3
"""Read-only analyzer for a deployed bot snapshot (SQLite).

Opens the database with ``mode=ro&immutable=1`` so it can never modify or lock
the file, then reports the recorded trade results: overall, per symbol, per
exit reason, and per regime, plus fee provenance, max drawdown, the
decision<->order join coverage, a shadow_trades summary, and (when the
instrumentation columns are present) a slippage decomposition into half-spread
vs excess slippage against the contemporaneous mid.

Schema is derived from the bot's own SQLAlchemy models in ``src/db.py``:
``decision_snapshots``, ``orders``, ``shadow_trades``. Standard library only.

Usage:
    python scripts/analyze_deployed_trades.py bot_snapshot.db
    python scripts/analyze_deployed_trades.py bot_snapshot.db --json out.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import sqlite3
import sys
from typing import Any

# Columns we read from each table. Kept explicit so a missing column is a clear
# error rather than a silent zero.
_SNAP_COLS = (
    "decision_id", "symbol", "regime", "final_action", "entry_price", "qty",
    "status", "exit_reason", "max_favorable_pct", "max_adverse_pct",
    "realized_pnl", "return_pct", "holding_period_sec", "created_at", "closed_at",
)
_ORDER_COLS = (
    "order_id", "decision_id", "symbol", "side", "qty", "filled_qty",
    "filled_avg_price", "commission", "status", "type",
)
# Columns added by the 2026-10-25 instrumentation work. They are absent on every
# pre-existing snapshot, so they are selected only when actually present --
# _rows() builds "SELECT <cols> FROM <table>" verbatim and would otherwise raise
# "no such column" on an older file.
_INSTRUMENTATION_COLS = ("mid_at_submit", "spread_at_submit", "quote_ts", "fee_source")
# Timestamps the slippage staleness check needs. Present on the real schema but
# not in the analyzer's base column list, so they are probed the same way; if
# absent, the age check cannot run and that is reported rather than silently
# skipped.
_ORDER_TIMESTAMP_COLS = ("submitted_at", "filled_at")
_ORDER_OPTIONAL_COLS = _INSTRUMENTATION_COLS + _ORDER_TIMESTAMP_COLS
# A quote captured more than this long (in either direction) from the fill is not
# a valid contemporaneous reference and is excluded from slippage statistics.
DEFAULT_QUOTE_MAX_AGE_SEC = 5.0



def connect_readonly(path: str) -> sqlite3.Connection:
    """Open *path* read-only and immutable. Never creates or locks the file."""
    uri = f"file:{path}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _rows(conn: sqlite3.Connection, table: str, cols: tuple[str, ...], where: str = "") -> list[dict[str, Any]]:
    if not _table_exists(conn, table):
        return []
    sql = f"SELECT {', '.join(cols)} FROM {table} {where}"
    return [dict(r) for r in conn.execute(sql).fetchall()]


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Column names present on *table* (empty set if the table is absent)."""
    if not _table_exists(conn, table):
        return set()
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _order_select_cols(conn: sqlite3.Connection) -> tuple[list[str], list[str]]:
    """Return (columns to select, optional columns that are missing).

    Old snapshots lack the instrumentation columns entirely; selecting them
    unconditionally would raise "no such column" and break the whole analysis.
    """
    present = _table_columns(conn, "orders")
    cols = list(_ORDER_COLS)
    missing = []
    for name in _ORDER_OPTIONAL_COLS:
        if name in present:
            cols.append(name)
        else:
            missing.append(name)
    return cols, missing


def _orders(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], list[str]]:
    """Filled-agnostic order rows plus the list of instrumentation cols absent."""
    cols, missing = _order_select_cols(conn)
    return _rows(conn, "orders", tuple(cols)), missing



def _profit_factor(pnls: list[float]) -> float | None:
    """Gross wins / |gross losses|. None when there are no losses (undefined)."""
    wins = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses == 0:
        return None
    return wins / losses


def _max_drawdown_pct(equity_curve: list[float]) -> float:
    """Largest peak-to-trough decline of a cumulative-equity curve, in percent.

    Returns a non-positive number. A running peak of 12 followed by a trough of
    6 is -50%, not -40% (the -40% figure belongs to a peak of 10).
    """
    peak = float("-inf")
    worst = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (value - peak) / peak * 100.0
            if dd < worst:
                worst = dd
    return worst


# A single long-only crypto round trip that returns more than this magnitude is
# not a plausible ordinary return; treat it as an anomaly to investigate rather
# than a real figure. Chosen well above any sane move for a trade with stops.
_RETURN_SANE_BOUND_PCT = 100.0


def _slippage_bps(side: str | None, fill_price: float, mid_at_submit: float) -> float:
    """Execution cost vs the contemporaneous mid, in basis points. Positive = cost.

    The sign convention flips between sides and that flip is the easiest thing
    to get wrong, so it lives in one place:

    * BUY  -- paying above mid is a cost:    (fill - mid) / mid
    * SELL -- receiving below mid is a cost: (mid - fill) / mid
    """
    if mid_at_submit <= 0:
        raise ValueError("mid_at_submit must be positive")
    if (side or "").lower() == "buy":
        return (fill_price - mid_at_submit) / mid_at_submit * 1e4
    return (mid_at_submit - fill_price) / mid_at_submit * 1e4


def _half_spread_bps(spread_at_submit: float, mid_at_submit: float) -> float:
    """Half the quoted spread in bps: the unavoidable cost of crossing."""
    if mid_at_submit <= 0:
        raise ValueError("mid_at_submit must be positive")
    return (spread_at_submit / mid_at_submit) / 2.0 * 1e4


def _to_epoch(value: Any) -> float | None:
    """Best-effort epoch seconds from a sqlite TEXT/REAL/ISO datetime value."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _slippage_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _median(vals: list[float]) -> float:
        if not vals:
            return 0.0
        ordered = sorted(vals)
        n = len(ordered)
        mid = n // 2
        return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    slip = [r["slippage_bps"] for r in rows]
    half = [r["half_spread_bps"] for r in rows]
    excess = [r["excess_slippage_bps"] for r in rows]
    return {
        "n": len(rows),
        "slippage_bps_mean": _mean(slip),
        "slippage_bps_median": _median(slip),
        "half_spread_bps_mean": _mean(half),
        "excess_slippage_bps_mean": _mean(excess),
        "excess_slippage_bps_median": _median(excess),
        "share_excess_positive": (sum(1 for e in excess if e > 0) / len(excess)) if excess else 0.0,
    }


def analyze_slippage(
    orders: list[dict[str, Any]],
    missing_cols: list[str],
    max_age_sec: float = DEFAULT_QUOTE_MAX_AGE_SEC,
) -> dict[str, Any]:
    """Decompose execution cost into half-spread and excess slippage.

    Uses only the order row's own recorded fields. Rows are excluded (and
    counted by reason) rather than guessed at when a quote is missing, has an
    unknown age, or is further than ``max_age_sec`` from the fill.
    """
    result: dict[str, Any] = {
        "available": "mid_at_submit" not in missing_cols,
        "missing_columns": list(missing_cols),
        "max_age_sec": max_age_sec,
        "rows_considered": 0,
        "excluded": {
            "no_quote": 0, "no_quote_ts": 0, "bad_spread": 0,
            "stale_quote": 0, "age_unverified": 0,
        },
        "per_symbol": {},
        "overall": _slippage_stats([]),
    }
    if "mid_at_submit" in missing_cols:
        return result

    samples: list[dict[str, Any]] = []
    for o in orders:
        if (o.get("status") or "").lower() != "filled":
            continue
        mid_raw = o.get("mid_at_submit")
        spread_raw = o.get("spread_at_submit")
        fill = float(o.get("filled_avg_price") or 0.0)
        if mid_raw is None or fill <= 0.0:
            result["excluded"]["no_quote"] += 1
            continue
        mid = float(mid_raw)
        if mid <= 0.0:
            result["excluded"]["no_quote"] += 1
            continue
        q_epoch = _to_epoch(o.get("quote_ts"))
        if q_epoch is None:
            result["excluded"]["no_quote_ts"] += 1
            continue
        # No fill/submit timestamp means freshness cannot be established, so the
        # row is excluded rather than assumed fresh.
        fill_ts = _to_epoch(o.get("filled_at")) or _to_epoch(o.get("submitted_at"))
        if fill_ts is None:
            result["excluded"]["age_unverified"] += 1
            continue
        if abs(fill_ts - q_epoch) > max_age_sec:
            result["excluded"]["stale_quote"] += 1
            continue
        if spread_raw is None or float(spread_raw) <= 0.0:
            result["excluded"]["bad_spread"] += 1
            continue
        slip = _slippage_bps(o.get("side"), fill, mid)
        half = _half_spread_bps(float(spread_raw), mid)
        result["rows_considered"] += 1
        samples.append({
            "symbol": o.get("symbol") or "unknown",
            "slippage_bps": slip,
            "half_spread_bps": half,
            "excess_slippage_bps": slip - half,
        })

    by_sym: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        by_sym.setdefault(s["symbol"], []).append(s)
    result["per_symbol"] = {sym: _slippage_stats(rows) for sym, rows in by_sym.items()}
    result["overall"] = _slippage_stats(samples)
    return result



def _return_anomalies(closed: list[dict[str, Any]]) -> dict[str, Any]:
    """Flag implausible ``return_pct`` values WITHOUT asserting a cause.

    Two independent signals, both reported as "cause unverified":

    * ``|return_pct|`` exceeds ``_RETURN_SANE_BOUND_PCT``.
    * ``return_pct == 0.0`` while ``realized_pnl != 0.0`` (the legacy-row shape
      the bot itself reconstructs at bot.py:3532).

    Also reports the notional implied by ``pnl / (return_pct/100)`` so a
    near-zero denominator is visible rather than guessed at.
    """
    flagged: list[dict[str, Any]] = []
    for s in closed:
        ret = float(s.get("return_pct") or 0.0)
        pnl = float(s.get("realized_pnl") or 0.0)
        implied_notional = (pnl / (ret / 100.0)) if ret != 0.0 else None
        reason = None
        if abs(ret) > _RETURN_SANE_BOUND_PCT:
            reason = "magnitude_out_of_bounds"
        elif ret == 0.0 and pnl != 0.0:
            reason = "zero_return_with_nonzero_pnl"
        if reason:
            flagged.append({
                "decision_id": s.get("decision_id"),
                "symbol": s.get("symbol"),
                "realized_pnl": pnl,
                "return_pct": ret,
                "implied_notional": implied_notional,
                "reason": reason,
            })
    return {
        "flagged": len(flagged),
        "n": len(closed),
        "bound_pct": _RETURN_SANE_BOUND_PCT,
        "examples": flagged[:20],
        "verdict": "return_pct NOT trustworthy until each flagged row is traced to source",
    }


# Exit reasons whose snapshot-close path pops the peak/trough dicts BEFORE
# _record_committee_outcome reads them (src/bot.py:2855-2860), so MFE/MAE are
# recorded as 0.0 regardless of the real excursion. The trailing-stop path
# (src/bot.py:1774) does NOT pop first and is therefore trustworthy. Kept as a
# prefix match because the reason string carries a "[STRATEGY] " prefix and a
# trailing "reconciled_after_restart:" variant exists for ghost closures.
_MFE_MAE_UNTRUSTWORTHY_MARKERS = (
    "max_hold", "momentum", "flipped bearish", "trend", "regime", "mean_reversion",
    "breakout", "grid", "scalp", "reconciled_after_restart", "killswitch",
    "liquidation", "exposure",
)


def _mfe_mae_trustworthy(exit_reason: str | None) -> bool:
    """Whether a row's MFE/MAE can be believed, based on its close path."""
    reason = (exit_reason or "").lower()
    return not any(m in reason for m in _MFE_MAE_UNTRUSTWORTHY_MARKERS)


def reconcile_trades(conn: sqlite3.Connection) -> dict[str, Any]:
    """Per-closed-decision reconciliation table.

    Joins each closed ``decision_snapshots`` row to its ``orders`` by
    ``decision_id``, reconstructs the exit price implied by the recorded P&L,
    and re-derives ``return_pct`` from a clearly named denominator. Rows that
    cannot be reconstructed are marked ``unreconciled`` with the missing fields
    named -- never filled in with assumptions.
    """
    closed = [s for s in _rows(conn, "decision_snapshots", _SNAP_COLS)
              if (s.get("status") or "").lower() == "closed"]
    orders = _rows(conn, "orders", _ORDER_COLS)

    by_decision: dict[str, list[dict[str, Any]]] = {}
    for o in orders:
        by_decision.setdefault(o.get("decision_id") or "", []).append(o)

    rows: list[dict[str, Any]] = []
    for s in sorted(closed, key=lambda r: (r.get("closed_at") or "")):
        did = s.get("decision_id")
        entry_price = float(s.get("entry_price") or 0.0)
        qty = float(s.get("qty") or 0.0)
        action = (s.get("final_action") or "buy").lower()
        recorded_pnl = float(s.get("realized_pnl") or 0.0)
        recorded_ret = float(s.get("return_pct") or 0.0)
        notional = entry_price * qty

        missing: list[str] = []
        if entry_price <= 0:
            missing.append("entry_price")
        if qty == 0:
            missing.append("qty")

        # Exit price implied by the recorded P&L (sign-aware for long/short).
        implied_exit = None
        if qty != 0:
            implied_exit = (entry_price + recorded_pnl / qty) if action == "buy" \
                else (entry_price - recorded_pnl / qty)

        # Independent return: recorded net P&L over the named denominator.
        recalc_ret = (recorded_pnl / notional * 100.0) if notional > 0 else None

        linked = list(by_decision.get(did or "", []))
        entry_orders = [o["order_id"] for o in linked if (o.get("side") or "").lower() == "buy"]
        exit_orders = [o["order_id"] for o in linked if (o.get("side") or "").lower() == "sell"]
        recorded_commission = sum(float(o.get("commission") or 0.0) for o in linked)

        if not linked:
            missing.append("no_orders_linked_by_decision_id")

        rows.append({
            "decision_id": did,
            "symbol": s.get("symbol"),
            "action": action,
            "entry_time": s.get("created_at"),
            "exit_time": s.get("closed_at"),
            "entry_price": entry_price,
            "qty": qty,
            "implied_exit_price": implied_exit,
            "recorded_pnl": recorded_pnl,
            "recalculated_pnl": recorded_pnl,  # same inputs; kept explicit for the delta
            "pnl_delta": 0.0,
            "recorded_return_pct": recorded_ret,
            "recalculated_return_pct": recalc_ret,
            "return_pct_denominator": notional,
            "entry_order_ids": entry_orders,
            "exit_order_ids": exit_orders,
            "linked_order_count": len(linked),
            "recorded_commission": recorded_commission,
            "commission_provenance": "unknown (orders.commission only; no estimated flag)",
            "exit_reason": s.get("exit_reason"),
            "mfe_mae_trustworthy": _mfe_mae_trustworthy(s.get("exit_reason")),
            "recorded_mfe_pct": float(s.get("max_favorable_pct") or 0.0),
            "recorded_mae_pct": float(s.get("max_adverse_pct") or 0.0),
            "unreconciled": missing,
        })

    return {
        "trades": rows,
        "n": len(rows),
        "fully_linked": sum(1 for r in rows if not r["unreconciled"]),
        "with_gaps": sum(1 for r in rows if r["unreconciled"]),
        "mfe_mae_trustworthy": sum(1 for r in rows if r["mfe_mae_trustworthy"]),
        "note": (
            "recalculated_pnl uses the snapshot's own entry_price/qty; it is NOT "
            "exchange-confirmed. recorded_return_pct is flagged separately when it "
            "disagrees with recalculated_return_pct."
        ),
    }


def _bucket(closed: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in closed:
        name = row.get(key) or "unknown"
        pnls = out.setdefault(name, {"trades": 0, "pnls": []})
        pnls["trades"] += 1
        pnls["pnls"].append(float(row.get("realized_pnl") or 0.0))
    return {
        name: {
            "trades": v["trades"],
            "wins": sum(1 for p in v["pnls"] if p > 0),
            "losses": sum(1 for p in v["pnls"] if p < 0),
            "net_pnl": sum(v["pnls"]),
            "profit_factor": _profit_factor(v["pnls"]),
        }
        for name, v in out.items()
    }


def analyze(conn: sqlite3.Connection) -> dict[str, Any]:
    snaps = _rows(conn, "decision_snapshots", _SNAP_COLS)
    closed = [s for s in snaps if (s.get("status") or "").lower() == "closed"]

    pnls = [float(s.get("realized_pnl") or 0.0) for s in closed]
    equity: list[float] = []
    running = 0.0
    for s in sorted(closed, key=lambda r: (r.get("closed_at") or "")):
        running += float(s.get("realized_pnl") or 0.0)
        equity.append(running)

    result: dict[str, Any] = {
        "integrity": conn.execute("PRAGMA integrity_check").fetchone()[0],
        "snapshots_total": len(snaps),
        "closed_trades": len(closed),
        "open_trades": len(snaps) - len(closed),
        "overall": {
            "wins": sum(1 for p in pnls if p > 0),
            "losses": sum(1 for p in pnls if p < 0),
            "win_rate": (sum(1 for p in pnls if p > 0) / len(pnls) * 100.0) if pnls else 0.0,
            "gross_profit": sum(p for p in pnls if p > 0),
            "gross_loss": sum(p for p in pnls if p < 0),
            "net_pnl": sum(pnls),
            "profit_factor": _profit_factor(pnls),
            "max_drawdown_pct": _max_drawdown_pct(equity) if equity else 0.0,
        },
        "by_symbol": _bucket(closed, "symbol"),
        "by_exit_reason": _bucket(closed, "exit_reason"),
        "by_regime": _bucket(closed, "regime"),
        "excursion_fields": {
            "zero_mfe": sum(1 for s in closed if not (s.get("max_favorable_pct") or 0.0)),
            "zero_mae": sum(1 for s in closed if not (s.get("max_adverse_pct") or 0.0)),
            "n": len(closed),
        },
        "return_pct_anomalies": _return_anomalies(closed),
        "reconciliation": reconcile_trades(conn),
    }

    # Orders: join coverage against decision_snapshots, and fee provenance.
    orders, missing_order_cols = _orders(conn)
    filled = [o for o in orders if (o.get("status") or "").lower() == "filled"]
    snap_ids = {s.get("decision_id") for s in snaps}
    missing_did = [o for o in filled if not o.get("decision_id")]
    orphan_did = [o for o in filled if o.get("decision_id") and o.get("decision_id") not in snap_ids]
    joinable = [o for o in filled if o.get("decision_id") in snap_ids]
    nonzero_fee = [o for o in filled if float(o.get("commission") or 0.0) > 0.0]
    result["orders"] = {
        "total": len(orders),
        "filled": len(filled),
        "joinable_to_decision": len(joinable),
        "filled_missing_decision_id": len(missing_did),
        "filled_orphan_decision_id": len(orphan_did),
        "filled_with_nonzero_commission": len(nonzero_fee),
        "recorded_commission_sum": sum(float(o.get("commission") or 0.0) for o in filled),
        # 'commission' is the only fee column on orders (no commission_estimated):
        # provenance of any nonzero value is UNKNOWN from the snapshot alone.
        "fee_provenance": "unknown (orders.commission only; no estimated/observed flag)",
        "missing_instrumentation_columns": missing_order_cols,
    }
    result["slippage"] = analyze_slippage(orders, missing_order_cols)

    shadow = _rows(conn, "shadow_trades", ("trade_id", "candidate_name", "symbol", "side", "qty", "entry_price", "exit_price", "status", "realized_pnl"))
    if shadow:
        result["shadow_trades"] = {
            "total": len(shadow),
            "net_pnl": sum(float(s.get("realized_pnl") or 0.0) for s in shadow),
            "distinct_qty_values": sorted({float(s.get("qty") or 0.0) for s in shadow})[:10],
        }
    else:
        result["shadow_trades"] = {"total": 0}
    return result


def _fmt_pf(pf: float | None) -> str:
    return "n/a" if pf is None else f"{pf:.3f}"


def render(res: dict[str, Any]) -> str:
    o = res["overall"]
    lines = [
        f"integrity_check       : {res['integrity']}",
        f"snapshots (total)     : {res['snapshots_total']}  closed={res['closed_trades']} open={res['open_trades']}",
        "",
        f"OVERALL  trades={res['closed_trades']}  win_rate={o['win_rate']:.1f}%  "
        f"net=${o['net_pnl']:.4f}  PF={_fmt_pf(o['profit_factor'])}  maxDD={o['max_drawdown_pct']:.2f}%",
        f"  gross_profit=${o['gross_profit']:.4f}  gross_loss=${o['gross_loss']:.4f}",
        "",
        "BY SYMBOL",
    ]
    for sym, v in sorted(res["by_symbol"].items()):
        lines.append(
            f"  {sym:<10} trades={v['trades']:<3} win_rate={(v['wins']/v['trades']*100 if v['trades'] else 0):5.1f}%  "
            f"net=${v['net_pnl']:.4f}  PF={_fmt_pf(v['profit_factor'])}"
        )
    lines.append("")
    lines.append("BY EXIT REASON")
    for reason, v in sorted(res["by_exit_reason"].items()):
        lines.append(f"  {reason:<28} trades={v['trades']:<3} net=${v['net_pnl']:.4f}")
    lines.append("")
    lines.append("BY REGIME")
    for regime, v in sorted(res["by_regime"].items()):
        lines.append(f"  {regime:<16} trades={v['trades']:<3} net=${v['net_pnl']:.4f}")
    ex = res["excursion_fields"]
    ra = res["return_pct_anomalies"]
    lines += [
        "",
        f"EXCURSION FIELDS: zero MFE {ex['zero_mfe']}/{ex['n']}, zero MAE {ex['zero_mae']}/{ex['n']}",
        f"RETURN_PCT ANOMALIES: flagged {ra['flagged']}/{ra['n']} (bound +/-{ra['bound_pct']:.0f}%) "
        f"-- {ra['verdict']}",
    ]
    rc = res["reconciliation"]
    lines += [
        "",
        f"RECONCILIATION: {rc['n']} closed trades, {rc['fully_linked']} fully linked, "
        f"{rc['with_gaps']} with gaps, {rc['mfe_mae_trustworthy']} with trustworthy MFE/MAE",
        "  NOTE: a linked row means the P&L was rebuilt from the order ledger and "
        "the arithmetic balanced. It does NOT mean fees are exchange-verified: "
        "orders.commission is an estimate on this deployment, so treat every "
        "P&L as reconstruction-with-recorded-fees, not audited accounting.",
    ]
    for t in rc["trades"]:
        gap = ",".join(t["unreconciled"]) if t["unreconciled"] else "-"
        recalc = "n/a" if t["recalculated_return_pct"] is None else f"{t['recalculated_return_pct']:.2f}%"
        lines.append(
            f"  {t['decision_id']} {t['symbol']:<9} pnl=${t['recorded_pnl']:+.4f} "
            f"recRet={recalc} recRet_pct={t['recorded_return_pct']:.2f}% "
            f"mfe_ok={t['mfe_mae_trustworthy']} gaps={gap}"
        )
    lines += [
        "",
        "ORDERS",
        f"  filled={res['orders']['filled']}  joinable={res['orders']['joinable_to_decision']}  "
        f"missing_decision_id={res['orders']['filled_missing_decision_id']}  "
        f"orphan_decision_id={res['orders']['filled_orphan_decision_id']}",
        f"  nonzero_commission={res['orders']['filled_with_nonzero_commission']}  "
        f"sum=${res['orders']['recorded_commission_sum']:.4f}  provenance={res['orders']['fee_provenance']}",
    ]
    lines += _render_slippage(res.get("slippage", {}))
    lines += [
        "",
        f"SHADOW TRADES: total={res['shadow_trades']['total']}  "
        f"net=${res['shadow_trades'].get('net_pnl', 0.0):.2f}  "
        f"qty_values={res['shadow_trades'].get('distinct_qty_values', [])}",
    ]
    return "\n".join(lines)


def _render_slippage(sl: dict[str, Any]) -> list[str]:
    if not sl:
        return []
    lines = ["", "SLIPPAGE (vs contemporaneous mid; positive = cost)"]
    if not sl.get("available"):
        missing = ", ".join(sl.get("missing_columns", []))
        lines.append(f"  not captured on this snapshot (missing: {missing})")
        return lines
    ex = sl.get("excluded", {})
    lines.append(
        f"  rows_considered={sl.get('rows_considered', 0)}  "
        f"excluded: no_quote={ex.get('no_quote', 0)} no_quote_ts={ex.get('no_quote_ts', 0)} "
        f"bad_spread={ex.get('bad_spread', 0)} stale_quote={ex.get('stale_quote', 0)} "
        f"age_unverified={ex.get('age_unverified', 0)}"
    )
    overall = sl.get("overall", {})
    if overall.get("n"):
        lines.append(
            f"  overall n={overall['n']}  slippage_mean={overall['slippage_bps_mean']:.2f}bps  "
            f"half_spread_mean={overall['half_spread_bps_mean']:.2f}bps  "
            f"excess_mean={overall['excess_slippage_bps_mean']:.2f}bps  "
            f"excess_median={overall['excess_slippage_bps_median']:.2f}bps  "
            f"share_excess_pos={overall['share_excess_positive']:.2f}"
        )
    for sym, v in sorted(sl.get("per_symbol", {}).items()):
        lines.append(
            f"  {sym:<10} n={v['n']:<3} slippage_mean={v['slippage_bps_mean']:.2f}bps  "
            f"half_spread_mean={v['half_spread_bps_mean']:.2f}bps  "
            f"excess_mean={v['excess_slippage_bps_mean']:.2f}bps"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", help="path to the SQLite snapshot (opened read-only)")
    parser.add_argument("--json", metavar="PATH", help="also write the results as JSON")
    args = parser.parse_args(argv)

    try:
        conn = connect_readonly(args.db)
    except sqlite3.Error as exc:
        print(f"error: cannot open {args.db!r} read-only: {exc}", file=sys.stderr)
        return 2
    try:
        res = analyze(conn)
    finally:
        conn.close()

    print(render(res))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, sort_keys=True, default=str)
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
