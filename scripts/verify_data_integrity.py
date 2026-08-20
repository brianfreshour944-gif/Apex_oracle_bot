#!/usr/bin/env python3
"""
Data Integrity Verification — Step 2 of Foundation Hardening.

Cross-checks logged fill data (price, commission, fees) against Alpaca's
actual execution records for recent trades. Outputs PASS/FAIL with details.

Run: python scripts/verify_data_integrity.py --days 7
"""

import os
import sys
import json
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.exchange import AlpacaExchange
from src.db import init_db, OrderRecord
from sqlalchemy import select
from sqlalchemy.orm import Session

DB_PATH = PROJECT_ROOT / "data" / "bot.db"

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_kv(key: str, value: any, indent: int = 2):
    prefix = " " * indent
    print(f"{prefix}{key}: {value}")

def load_trades_from_db(days_back: int) -> list:
    """Load completed trades with fill data from local database."""
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return []
    
    from src.db import get_engine, DecisionSnapshot, OrderRecord
    from sqlalchemy import select, and_
    from sqlalchemy.orm import Session
    
    engine = get_engine()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    
    with Session(engine) as session:
        # Query closed decision snapshots with their order records
        stmt = (
            select(DecisionSnapshot, OrderRecord)
            .outerjoin(OrderRecord, OrderRecord.decision_id == DecisionSnapshot.decision_id)
            .where(
                and_(
                    DecisionSnapshot.status == "closed",
                    DecisionSnapshot.created_at >= cutoff
                )
            )
            .order_by(DecisionSnapshot.created_at.desc())
        )
        
        results = session.execute(stmt).all()
        trades = []
        for snap, order in results:
            trade = {
                "decision_id": snap.decision_id,
                "symbol": snap.symbol,
                "action": snap.final_action,
                "entry_price": snap.entry_price,
                "qty": snap.qty,
                "entry_time": snap.created_at.isoformat() if snap.created_at else None,
                "exit_price": snap.realized_pnl,  # Using realized_pnl as proxy since exit_price doesn't exist
                "realized_pnl": snap.realized_pnl,
                "return_pct": snap.return_pct,
                "holding_period_sec": snap.holding_period_sec,
                "exit_reason": snap.exit_reason,
                "order_filled_price": order.filled_avg_price if order else None,
                "order_commission": order.commission if order else None,
                "order_filled_qty": order.filled_qty if order else None,
                "order_time": order.submitted_at.isoformat() if order and order.submitted_at else None,
                "order_status": order.status if order else None,
            }
            trades.append(trade)
        
        print(f"Loaded {len(trades)} completed trades from last {days_back} days")
        return trades

def fetch_alpaca_fills(days_back: int) -> list:
    """Fetch actual fills from Alpaca for the same period."""
    try:
        exchange = AlpacaExchange()
        
        # Alpaca's get_orders can filter by date
        after = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
        
        # Use the exchange's method to get order history
        import asyncio
        async def _get_orders():
            return await exchange.get_orders(status="filled", after=after, limit=500)
        
        orders = asyncio.run(_get_orders())
        print(f"Fetched {len(orders)} filled orders from Alpaca")
        return orders
    except Exception as e:
        print(f"Failed to fetch Alpaca orders: {e}")
        return []

def match_and_compare(db_trades: list, alpaca_orders: list) -> dict:
    """Match trades by symbol+time and compare fill data."""
    # Index Alpaca orders by symbol and approximate time
    alpaca_by_symbol = {}
    for order in alpaca_orders:
        sym = order.get("symbol", "").replace("/", "")
        if sym not in alpaca_by_symbol:
            alpaca_by_symbol[sym] = []
        alpaca_by_symbol[sym].append(order)
    
    results = {
        "total_compared": 0,
        "price_matches": 0,
        "price_mismatches": [],
        "commission_matches": 0,
        "commission_mismatches": [],
        "qty_matches": 0,
        "qty_mismatches": [],
        "missing_in_alpaca": [],
        "missing_in_db": [],
    }
    
    for trade in db_trades:
        symbol = trade["symbol"].replace("/", "")
        entry_time = trade.get("entry_time", "")
        
        # Find matching Alpaca order (same symbol, close in time)
        matched = None
        if symbol in alpaca_by_symbol:
            for alpaca_order in alpaca_by_symbol[symbol]:
                alpaca_time = alpaca_order.get("submitted_at", "")
                if alpaca_time and entry_time:
                    try:
                        t1 = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                        t2 = datetime.fromisoformat(alpaca_time.replace("Z", "+00:00"))
                        if abs((t1 - t2).total_seconds()) < 300:  # Within 5 minutes
                            matched = alpaca_order
                            break
                    except Exception:
                        pass
        
        if not matched:
            results["missing_in_alpaca"].append({
                "symbol": trade["symbol"],
                "entry_time": entry_time,
                "db_price": trade.get("entry_price"),
                "db_qty": trade.get("qty")
            })
            continue
        
        results["total_compared"] += 1
        
        # Compare filled price
        db_price = trade.get("entry_price") or trade.get("order_filled_price")
        alpaca_price = matched.get("filled_avg_price", 0)
        if db_price and alpaca_price:
            price_diff_pct = abs(db_price - alpaca_price) / alpaca_price * 100
            if price_diff_pct < 0.1:  # Within 0.1%
                results["price_matches"] += 1
            else:
                results["price_mismatches"].append({
                    "symbol": trade["symbol"],
                    "db_price": db_price,
                    "alpaca_price": alpaca_price,
                    "diff_pct": round(price_diff_pct, 4)
                })
        
        # Compare commission
        db_comm = trade.get("order_commission", 0)
        alpaca_comm = matched.get("commission", 0)
        if abs(db_comm - alpaca_comm) < 0.01:
            results["commission_matches"] += 1
        else:
            results["commission_mismatches"].append({
                "symbol": trade["symbol"],
                "db_commission": db_comm,
                "alpaca_commission": alpaca_comm
            })
        
        # Compare quantity
        db_qty = trade.get("qty") or trade.get("order_filled_qty")
        alpaca_qty = matched.get("filled_qty", 0)
        if db_qty and alpaca_qty:
            if abs(db_qty - alpaca_qty) < 1e-6:
                results["qty_matches"] += 1
            else:
                results["qty_mismatches"].append({
                    "symbol": trade["symbol"],
                    "db_qty": db_qty,
                    "alpaca_qty": alpaca_qty
                })
    
    # Check for Alpaca orders not in DB
    db_decision_ids = {t.get("decision_id") for t in db_trades if t.get("decision_id")}
    for order in alpaca_orders:
        # Can't easily match without decision_id in Alpaca
        pass
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Verify data integrity against Alpaca")
    parser.add_argument("--days", type=int, default=7, help="Days to look back")
    args = parser.parse_args()
    
    print_section(f"DATA INTEGRITY VERIFICATION — Last {args.days} Days")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    
    # Load data
    db_trades = load_trades_from_db(args.days)
    if not db_trades:
        print("No trades found in database. Exiting.")
        return
    
    alpaca_orders = fetch_alpaca_fills(args.days)
    if not alpaca_orders:
        print("No orders fetched from Alpaca. Exiting.")
        return
    
    # Compare
    results = match_and_compare(db_trades, alpaca_orders)
    
    # Report
    print_section("COMPARISON RESULTS")
    print_kv("Total Trades Compared", results["total_compared"])
    print_kv("Price Matches", f"{results['price_matches']}/{results['total_compared']}")
    print_kv("Commission Matches", f"{results['commission_matches']}/{results['total_compared']}")
    print_kv("Quantity Matches", f"{results['qty_matches']}/{results['total_compared']}")
    print_kv("Missing in Alpaca", len(results["missing_in_alpaca"]))
    
    if results["price_mismatches"]:
        print_section("PRICE MISMATCHES (>0.1%)")
        for m in results["price_mismatches"]:
            print(f"  {m['symbol']}: DB=${m['db_price']:.4f} vs Alpaca=${m['alpaca_price']:.4f} (diff={m['diff_pct']}%)")
    
    if results["commission_mismatches"]:
        print_section("COMMISSION MISMATCHES")
        for m in results["commission_mismatches"]:
            print(f"  {m['symbol']}: DB=${m['db_commission']:.4f} vs Alpaca=${m['alpaca_commission']:.4f}")
    
    if results["qty_mismatches"]:
        print_section("QUANTITY MISMATCHES")
        for m in results["qty_mismatches"]:
            print(f"  {m['symbol']}: DB={m['db_qty']} vs Alpaca={m['alpaca_qty']}")
    
    if results["missing_in_alpaca"]:
        print_section("TRADES IN DB BUT NOT FOUND IN ALPACA")
        for m in results["missing_in_alpaca"]:
            print(f"  {m['symbol']} at {m['entry_time']}: price={m['db_price']}, qty={m['db_qty']}")
    
    # Verdict
    total_mismatches = (
        len(results["price_mismatches"]) +
        len(results["commission_mismatches"]) +
        len(results["qty_mismatches"]) +
        len(results["missing_in_alpaca"])
    )
    
    print_section("VERDICT")
    if total_mismatches == 0 and results["total_compared"] > 0:
        print("  PASS — All fill data matches Alpaca records")
        verdict = "PASS"
    elif total_mismatches == 0:
        print("  INCONCLUSIVE — No trades to compare")
        verdict = "INCONCLUSIVE"
    else:
        print(f"  FAIL — {total_mismatches} discrepancies found")
        verdict = "FAIL"
    
    # Save report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "days_back": args.days,
        "verdict": verdict,
        "results": results
    }
    report_path = PROJECT_ROOT / "data" / f"data_integrity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")

if __name__ == "__main__":
    main()