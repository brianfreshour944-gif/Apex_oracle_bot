#!/usr/bin/env python3
"""
scripts/evolution_cull.py — The Monthly Cull (Level 4)
Evaluates all shadow models against the production model based on live paper-trading PnL.
"""

import json
import logging
import os
import sys
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio

from src import model_store
from src.db import Base, DecisionSnapshot, ShadowTrade, get_db_session, get_engine
from src.telegram_alerts import send_telegram_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CANDIDATES_DIR = os.path.join(DATA_DIR, "candidates")

def evaluate_and_cull() -> int:
    Base.metadata.create_all(get_engine())
    
    # Calculate cutoff for "this month"
    one_month_ago = datetime.now(UTC) - timedelta(days=30)
    
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
        cand_config = os.path.join(CANDIDATES_DIR, f"{best_candidate}_config.json")

        # Promote weights, scaler AND architecture as one bundle into the
        # persistent model store. This used to copy only weights + scaler
        # into data/, which the live brain never reads -- and even there a
        # different-sized candidate would not have loaded against the old
        # transformer_config.json.
        if all(os.path.exists(p) for p in (cand_pth, cand_scaler, cand_config)):
            with open(cand_config) as f:
                cand_arch = json.load(f)
            with model_store.staging("transformer") as staged:
                arch_path = os.path.join(staged, "transformer_config.json")
                with open(arch_path, "w") as f:
                    # automl_pipeline trains every candidate with 8 query / 2 KV heads
                    json.dump({
                        "num_layers": cand_arch.get("layers", 4),
                        "embed_dim": cand_arch.get("embed", 128),
                        "num_q_heads": 8,
                        "num_kv_heads": 2,
                    }, f)
                model_store.promote("transformer", {
                    "grok_gqa_v9_best.pth": cand_pth,
                    "feature_scaler.pkl": cand_scaler,
                    "transformer_config.json": arch_path,
                })
        else:
            log.error(f"{best_candidate} is missing its weights, scaler or config in {CANDIDATES_DIR} -- not promoting.")
            msg_lines.append("\n⚠️ Promotion skipped: candidate files incomplete.")
    else:
        log.info("🛡️ Production model defended its title. No promotion.")
        msg_lines.append("\n🛡️ Production defended its title. No promotion.")
        
    try:
        asyncio.run(send_telegram_alert("\n".join(msg_lines)))
    except Exception as e:
        log.error(f"Telegram alert failed: {e}")
        
    # Optional: Wipe shadow trades to start fresh for next month?
    # We'll leave them in the DB for historical record, the `closed_at >= one_month_ago` filter handles the window.
    return 0

if __name__ == "__main__":
    sys.exit(evaluate_and_cull())
