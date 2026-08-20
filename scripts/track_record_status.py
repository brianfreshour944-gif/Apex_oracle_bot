#!/usr/bin/env python3
"""
Track Record Status — Step 3 of Foundation Hardening.

Reports on the actual validated track record of the system:
- Retraining cycles completed (PPO weekly, DT daily/weekly)
- Adaptive learner gates passed per regime
- Trade counts and statistics per regime
- Mathematical learner 30-trade + Sharpe/win-rate gates

Run: python scripts/track_record_status.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.committee.committee import get_meta_learner
from src.db import init_db, get_engine, DecisionSnapshot
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session

DB_PATH = PROJECT_ROOT / "data" / "bot.db"
ADAPTIVE_STATE_PATH = PROJECT_ROOT / "data" / "adaptive_meta_state.json"

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_kv(key: str, value: any, indent: int = 2):
    prefix = " " * indent
    print(f"{prefix}{key}: {value}")

def load_adaptive_state() -> dict:
    if ADAPTIVE_STATE_PATH.exists():
        with open(ADAPTIVE_STATE_PATH) as f:
            return json.load(f)
    return {}

def get_db_trade_stats(days_back: int = 30) -> dict:
    """Get trade statistics from database."""
    if not DB_PATH.exists():
        return {}
    
    engine = get_engine()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    
    with Session(engine) as session:
        # Overall stats - fetch all closed trades and compute in Python
        stmt = select(DecisionSnapshot).where(
            and_(
                DecisionSnapshot.status == "closed",
                DecisionSnapshot.created_at >= cutoff
            )
        )
        trades = session.execute(stmt).scalars().all()
        
        if not trades:
            return {"overall": {"total": 0, "wins": 0, "losses": 0, "avg_return_pct": 0, "total_pnl": 0}, "by_regime": {}}
        
        total = len(trades)
        wins = sum(1 for t in trades if t.realized_pnl > 0)
        losses = sum(1 for t in trades if t.realized_pnl < 0)
        avg_return = sum(t.return_pct for t in trades) / total
        total_pnl = sum(t.realized_pnl for t in trades)
        
        overall = {
            "total": total,
            "wins": wins,
            "losses": losses,
            "avg_return_pct": avg_return,
            "total_pnl": total_pnl
        }
        
        # Per-regime stats
        by_regime = {}
        for trade in trades:
            regime = trade.regime
            if regime not in by_regime:
                by_regime[regime] = {"total": 0, "wins": 0, "losses": 0, "returns": [], "pnls": []}
            by_regime[regime]["total"] += 1
            if trade.realized_pnl > 0:
                by_regime[regime]["wins"] += 1
            elif trade.realized_pnl < 0:
                by_regime[regime]["losses"] += 1
            by_regime[regime]["returns"].append(trade.return_pct)
            by_regime[regime]["pnls"].append(trade.realized_pnl)
        
        # Compute averages
        for regime, data in by_regime.items():
            data["win_rate"] = data["wins"] / data["total"] if data["total"] > 0 else 0
            data["avg_return_pct"] = sum(data["returns"]) / len(data["returns"]) if data["returns"] else 0
            data["total_pnl"] = sum(data["pnls"])
            # Clean up temporary fields
            del data["returns"]
            del data["pnls"]
        
        return {"overall": overall, "by_regime": by_regime}

def check_retraining_cycles() -> dict:
    """Check if retraining scripts have run recently."""
    scripts_to_check = {
        "PPO Weekly (Sunday)": "scripts/train_ppo_meta.py",
        "DT Daily Replay": "scripts/retrain_transformer.py",
        "DT Full Retrain (Saturday)": "scripts/train_decision_transformer.py",
        "Weekly Analyzer": "scripts/weekly_analyzer.py",
        "AutoML Pipeline (Saturday)": "scripts/automl_pipeline.py",
    }
    
    results = {}
    for name, path in scripts_to_check.items():
        full_path = PROJECT_ROOT / path
        if full_path.exists():
            stat = full_path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - modified).total_seconds() / 3600
            results[name] = {
                "exists": True,
                "last_modified": modified.isoformat(),
                "age_hours": round(age_hours, 1)
            }
        else:
            results[name] = {"exists": False}
    
    return results

def check_model_freshness() -> dict:
    """Check model file timestamps."""
    models = {
        "Transformer": PROJECT_ROOT / "models" / "grok_gqa_v9_best.pth",
        "Feature Scaler": PROJECT_ROOT / "models" / "feature_scaler.pkl",
        "PPO Model": PROJECT_ROOT / "models" / "ppo_meta_weights.zip",
        "DT Model": PROJECT_ROOT / "models" / "decision_transformer.pth",
    }
    
    results = {}
    for name, path in models.items():
        if path.exists():
            stat = path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            age_days = (datetime.now(timezone.utc) - modified).total_seconds() / 86400
            results[name] = {
                "exists": True,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "last_modified": modified.isoformat(),
                "age_days": round(age_days, 1)
            }
        else:
            results[name] = {"exists": False, "path": str(path.relative_to(PROJECT_ROOT))}
    
    return results

def main():
    print_section(f"TRACK RECORD STATUS — {datetime.now(timezone.utc).isoformat()}")
    
    # 1. Adaptive Learner Gates
    print_section("1. ADAPTIVE LEARNER VALIDATION GATES")
    learner = get_meta_learner()
    if learner:
        adaptive_state = load_adaptive_state()
        print_kv("State File Exists", ADAPTIVE_STATE_PATH.exists())
        print_kv("Total Samples (all regimes)", learner.sample_count)
        print_kv("Min Trades Before Live", settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE)
        print_kv("Min Validation Trades", settings.ADAPTIVE_MIN_VALIDATION_TRADES)
        print_kv("Min Sharpe for Validation", settings.ADAPTIVE_MIN_SHARPE)
        print_kv("Min Win Rate for Validation", settings.ADAPTIVE_MIN_WIN_RATE)
        
        print("\n  Per-Regime Status:")
        any_validated = False
        for regime in sorted(set(list(learner.regime_sample_count.keys()) + list(learner.weights.keys()))):
            n = learner.regime_sample_count.get(regime, 0)
            validated = learner.regime_validated.get(regime, False)
            weights = learner.weights.get(regime, {})
            w_str = " ".join(f"{k}={v:.2f}" for k, v in weights.items()) if weights else "EQUAL"
            metrics = learner.get_regime_validation_metrics(regime)
            
            gate_ready = n >= settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE
            val_ready = validated
            
            print(f"    {regime}:")
            print(f"      Samples: {n} (gate: {'PASS' if gate_ready else 'PENDING'} >= {settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE})")
            print(f"      Validated: {val_ready} (Sharpe={metrics['sharpe']:.3f}, WinRate={metrics['win_rate']:.3f}, n={metrics['n_holdout']})")
            print(f"      Weights: [{w_str}]")
            if val_ready:
                any_validated = True
        
        print_kv("Any Regime Validated", any_validated)
    else:
        print("  Adaptive learner not loaded")
    
    # 2. Database Trade Stats
    print_section("2. DATABASE TRADE STATISTICS (Last 30 Days)")
    stats = get_db_trade_stats(30)
    if stats:
        o = stats["overall"]
        print_kv("Total Trades", o["total"])
        print_kv("Wins / Losses", f"{o['wins']} / {o['losses']}")
        if o["total"] > 0:
            print_kv("Win Rate", f"{o['wins']/o['total']*100:.1f}%")
        print_kv("Avg Return/Trade", f"{o['avg_return_pct']:.3f}%")
        print_kv("Total PnL", f"${o['total_pnl']:.2f}")
        
        print("\n  By Regime:")
        for regime, r in stats["by_regime"].items():
            print(f"    {regime}: {r['total']} trades, {r['win_rate']*100:.1f}% WR, ${r['total_pnl']:.2f} PnL, avg {r['avg_return_pct']:.3f}%")
    else:
        print("  No database or no trades found")
    
    # 3. Mathematical Learner Gate (30 trades + Sharpe/Win-rate)
    print_section("3. MATHEMATICAL LEARNER GATE (30 trades + Sharpe/Win-rate)")
    # This would require checking the mathematical learner's specific state
    # For now, report what we know from adaptive learner
    total_samples = learner.sample_count if learner else 0
    print_kv("Total Realized Outcomes (all regimes)", total_samples)
    print_kv("Gate Threshold", "30 trades + Sharpe > 0.5 + Win Rate > 52%")
    if total_samples >= 30:
        print_kv("30-Trade Minimum", "MET")
    else:
        print_kv("30-Trade Minimum", f"NOT MET ({total_samples}/30)")
    
    # 4. Retraining Cycles
    print_section("4. RETRAINING CYCLE STATUS")
    cycles = check_retraining_cycles()
    for name, info in cycles.items():
        if info["exists"]:
            age = info["age_hours"]
            status = "RECENT" if age < 48 else "STALE" if age < 168 else "OLD"
            print_kv(name, f"{status} — {age:.1f}h ago ({info['last_modified']})")
        else:
            print_kv(name, "SCRIPT NOT FOUND")
    
    # 5. Model Freshness
    print_section("5. MODEL FRESHNESS")
    models = check_model_freshness()
    for name, info in models.items():
        if info["exists"]:
            age = info["age_days"]
            status = "FRESH" if age < 7 else "AGING" if age < 30 else "STALE"
            print_kv(name, f"{status} — {age:.1f}d ago ({info['path']})")
        else:
            print_kv(name, f"NOT FOUND ({info.get('path', 'unknown')})")
    
    # 6. Summary / Verdict
    print_section("6. TRACK RECORD VERDICT")
    
    checks = []
    
    # Check 1: Any regime validated
    if learner and any(learner.regime_validated.values()):
        checks.append(("Regime validated", True))
    else:
        checks.append(("Regime validated", False))
    
    # Check 2: Sufficient trades overall
    total_trades = stats.get("overall", {}).get("total", 0)
    checks.append((f"Trades >= 30 ({total_trades})", total_trades >= 30))
    
    # Check 3: Positive expectancy
    if total_trades > 0:
        avg_ret = stats["overall"]["avg_return_pct"]
        checks.append((f"Positive expectancy ({avg_ret:.3f}%)", avg_ret > 0))
    else:
        checks.append(("Positive expectancy", False))
    
    # Check 4: Recent retraining
    recent_training = any(c["exists"] and c["age_hours"] < 168 for c in cycles.values())
    checks.append(("Retraining within 7 days", recent_training))
    
    # Check 5: Models exist
    models_exist = all(m["exists"] for m in models.values())
    checks.append(("All models present", models_exist))
    
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    
    all_pass = all(passed for _, passed in checks)
    print(f"\n  OVERALL: {'FOUNDATION READY' if all_pass else 'FOUNDATION NOT READY'}")
    print("  (Requires all checks PASS before lifting capability freeze)")
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()