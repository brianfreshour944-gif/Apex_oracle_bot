"""Post-Mortem AI.

Analyzes losing trades from the database to identify common failure conditions
(regime misclassifications, bad features) and generates an analytical report.
"""

import sys
import os
import json
from datetime import datetime
from collections import Counter
from typing import Dict, Any, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.db import get_engine, DecisionSnapshot
from src.logging_config import get_logger
from sqlalchemy import select
from sqlalchemy.orm import Session

logger = get_logger("post_mortem_ai")

def analyze_losses() -> int:
    logger.info("Starting Post-Mortem Analysis")
    engine = get_engine()
    
    with Session(engine) as session:
        # Fetch closed trades with negative PnL
        stmt = select(DecisionSnapshot).where(
            DecisionSnapshot.status == "closed",
            DecisionSnapshot.realized_pnl < 0
        )
        losing_trades = session.execute(stmt).scalars().all()
        
    if not losing_trades:
        logger.info("No losing trades found in the database. Perfect record!")
        return
        
    logger.info(f"Analyzing {len(losing_trades)} losing trades...")
    
    # Analyze regimes
    regime_counter = Counter([t.regime for t in losing_trades])
    total_losses = len(losing_trades)
    
    # Analyze features
    # We will track the average RSI, ATR, and common feature states during losses
    total_rsi = 0.0
    total_atr = 0.0
    feature_counts = 0
    strategy_counter = Counter()
    causal_aggregate = Counter()
    causal_counts = Counter()
    
    for t in losing_trades:
        try:
            features = json.loads(t.feature_snapshot_json)
            total_rsi += features.get("rsi", 50.0)
            total_atr += features.get("atr", 0.0)
            strategy = features.get("selected_strategy", "unknown")
            strategy_counter[strategy] += 1
            feature_counts += 1
        except Exception:
            pass
            
        try:
            causal = json.loads(t.causal_reasoning_json)
            if "transformer" in causal and causal["transformer"]:
                for feat, val in causal["transformer"].items():
                    causal_aggregate[feat] += val
                    causal_counts[feat] += 1
        except Exception:
            pass
            
    avg_rsi = total_rsi / feature_counts if feature_counts else 50.0
    avg_atr = total_atr / feature_counts if feature_counts else 0.0
    
    # Generate the report
    report_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'post_mortem.md')
    
    with open(report_path, "w") as f:
        f.write("# 💀 Post-Mortem AI Report\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n")
        f.write(f"The system analyzed **{total_losses}** losing trades from the database to identify common points of failure.\n\n")
        
        f.write("## 1. Regime Vulnerabilities\n")
        f.write("Which market regimes cause the most losses?\n")
        for regime, count in regime_counter.most_common():
            pct = (count / total_losses) * 100
            f.write(f"- **{regime.upper()}**: {pct:.1f}% of all losses ({count} trades)\n")
            
        f.write("\n## 2. Strategy Failures\n")
        f.write("Which execution strategies are misfiring?\n")
        for strat, count in strategy_counter.most_common():
            pct = (count / total_losses) * 100
            f.write(f"- **{strat}**: {pct:.1f}% of all losses ({count} trades)\n")
            
        f.write("\n## 3. Feature Breakdown\n")
        f.write("Average technical conditions during a loss:\n")
        f.write(f"- **Average RSI:** {avg_rsi:.2f}\n")
        f.write(f"- **Average ATR (Volatility):** {avg_atr:.4f}\n\n")
        
        if causal_counts:
            f.write("## 4. Causal Reasoning (Transformer SHAP)\n")
            f.write("Which features did the Transformer incorrectly weight the highest during these losses?\n")
            for feat, count in causal_counts.most_common():
                avg_val = causal_aggregate[feat] / count
                if abs(avg_val) > 0.01:
                    direction = "Positive" if avg_val > 0 else "Negative"
                    f.write(f"- **{feat}**: Average {direction} contribution of {avg_val:+.4f}\n")
            f.write("\n")
        
        f.write("## AI Recommendations\n")
        
        # Simple heuristic recommendations
        worst_regime = regime_counter.most_common(1)[0][0] if regime_counter else "None"
        worst_strat = strategy_counter.most_common(1)[0][0] if strategy_counter else "None"
        
        f.write(f"> [!WARNING]\n")
        f.write(f"> **Critical Vulnerability Detected:** The bot is struggling in the **{worst_regime}** regime, specifically when using the **{worst_strat}** strategy.\n")
        f.write("> **Action Item:** The AutoML Pipeline should penalize the weights for this strategy in this regime, or the user should tweak the strategy parameters in `execution_strategies.py`.\n")

    logger.info(f"Post-Mortem report generated at {report_path}")
    return 0

if __name__ == "__main__":
    sys.exit(analyze_losses())
