#!/usr/bin/env python3
"""
scripts/evolution_cull.py — The Monthly Cull (Level 4)
Evaluates all shadow models against the production model based on live paper-trading PnL.
"""

import os
import sys
import logging
import shutil
from sqlalchemy import select, func
from datetime import datetime, timezone, timedelta

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db import get_engine, get_db_session, ShadowTrade, DecisionSnapshot, Base
from src.telegram_alerts import send_telegram_alert
import asyncio

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CANDIDATES_DIR = os.path.join(DATA_DIR, "candidates")
PROD_MODEL_OUT = os.path.join(DATA_DIR, "grok_gqa_v9_best.pth")
PROD_SCALER_OUT = os.path.join(DATA_DIR, "feature_scaler.pkl")

def evaluate_and_cull() -> int:
    Base.metadata.create_all(get_engine())
    
    # Calculate cutoff for "this month"
    one_month_ago = datetime.now(timezone.utc) - timedelta(days=30)
    
    scores = {}
    
    with get_db_session() as session:
        # 1. Get Production PnL
        stmt_prod = select(func.sum(DecisionSnapshot.realized_pnl)).where(
            DecisionSnapshot.status == "closed",
            DecisionSnapshot.closed_at >= one_month_ago
        )
        prod_pnl = session.execute(stmt_prod).scalar() or 0.0
        scores["Production"] = prod_pnl
        
        # 2. Get Shadow Candidates PnL
        stmt_shadow = select(ShadowTrade.candidate_name, func.sum(ShadowTrade.realized_pnl)).where(
            ShadowTrade.status == "closed",
            ShadowTrade.closed_at >= one_month_ago
        ).group_by(ShadowTrade.candidate_name)
        
        shadow_results = session.execute(stmt_shadow).all()
        for candidate, pnl in shadow_results:
            scores[candidate] = pnl
            
    log.info("\n=== 🩸 EVOLUTION TOURNAMENT: THE CULL 🩸 ===")
    msg_lines = ["🩸 <b>Evolution Tournament Results</b>"]
    
    best_candidate = "Production"
    best_pnl = scores["Production"]
    
    for name, pnl in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        log.info(f"{name}: ${pnl:.2f}")
        msg_lines.append(f"- {name}: ${pnl:.2f}")
        if pnl > best_pnl:
            best_pnl = pnl
            best_candidate = name
            
    if best_candidate != "Production":
        log.info(f"🏆 {best_candidate} defeated Production! Promoting weights.")
        msg_lines.append(f"\n🏆 {best_candidate} wins! Promoting to Production.")
        
        cand_pth = os.path.join(CANDIDATES_DIR, f"{best_candidate}.pth")
        cand_scaler = os.path.join(CANDIDATES_DIR, "feature_scaler.pkl")
        
        if os.path.exists(cand_pth):
            shutil.copy(cand_pth, PROD_MODEL_OUT)
        if os.path.exists(cand_scaler):
            shutil.copy(cand_scaler, PROD_SCALER_OUT)
    else:
        log.info("🛡️ Production model defended its title. No promotion.")
        msg_lines.append(f"\n🛡️ Production defended its title. No promotion.")
        
    try:
        asyncio.run(send_telegram_alert("\n".join(msg_lines)))
    except Exception as e:
        log.error(f"Telegram alert failed: {e}")
        
    # Optional: Wipe shadow trades to start fresh for next month?
    # We'll leave them in the DB for historical record, the `closed_at >= one_month_ago` filter handles the window.
    return 0

if __name__ == "__main__":
    sys.exit(evaluate_and_cull())
