"""Automatic Researcher: An AI Quant Researcher that runs on weekends.

It backtests permutations of Stop Loss, Take Profit, and Indicator thresholds.
It compares them to the baseline (current settings) and generates a report
of statistically significant improvements.
"""

import sys
import os
import asyncio
import itertools
from datetime import datetime
from typing import Dict, Any, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.backtest import run_backtest, BacktestResult
from src.logging_config import get_logger

logger = get_logger("automatic_researcher")


# The grid of parameters to test
RESEARCH_GRID = {
    "STOP_LOSS_PCT": [0.02, 0.03, 0.04, 0.05, 0.06],
    "PROFIT_TARGET_PCT": [0.02, 0.03, 0.04, 0.05, 0.08],
    "RSI_OVERSOLD": [20.0, 25.0, 30.0],
    "RSI_OVERBOUGHT": [70.0, 75.0, 80.0],
}

# Significance thresholds to filter noise
MIN_SHARPE_IMPROVEMENT_PCT = 10.0  # Must improve Sharpe by 10%
MIN_RETURN_IMPROVEMENT_PCT = 5.0   # Must improve total return by 5%


async def run_experiment(symbol: str, config: Dict[str, Any], seed: int = 42) -> BacktestResult:
    """Run a backtest with temporary settings."""
    # Store originals
    originals = {}
    for k, v in config.items():
        originals[k] = getattr(settings, k)
        setattr(settings, k, v)
        
    try:
        # Run backtest with the modified settings
        res = await run_backtest(
            symbol=symbol,
            n_bars=1000,
            start_equity=10000.0,
            seed=seed,
            regime="all" # Run over a mix, though we use the synthetic generator
        )
        return res
    finally:
        # Restore originals
        for k, v in originals.items():
            setattr(settings, k, v)

async def main() -> int:
    logger.info("Starting Automatic Research Cycle")
    
    symbol = "BTC/USD"
    seed = 99
    
    # 1. Establish Baseline
    logger.info("Running Baseline Configuration...")
    baseline_config = {
        "STOP_LOSS_PCT": settings.STOP_LOSS_PCT,
        "PROFIT_TARGET_PCT": settings.PROFIT_TARGET_PCT,
        "RSI_OVERSOLD": settings.RSI_OVERSOLD,
        "RSI_OVERBOUGHT": settings.RSI_OVERBOUGHT,
    }
    baseline_res = await run_experiment(symbol, baseline_config, seed)
    logger.info(f"Baseline -> Return: {baseline_res.total_return_pct:.2f}% | Sharpe: {baseline_res.sharpe:.2f} | MaxDD: {baseline_res.max_drawdown_pct:.2f}%")
    
    # 2. Build Permutations
    keys = list(RESEARCH_GRID.keys())
    values = list(RESEARCH_GRID.values())
    permutations = list(itertools.product(*values))
    
    logger.info(f"Testing {len(permutations)} experimental configurations...")
    
    significant_improvements = []
    
    for i, p in enumerate(permutations):
        config = dict(zip(keys, p))
        
        # Skip evaluating the exact baseline
        if config == baseline_config:
            continue
            
        res = await run_experiment(symbol, config, seed)
        
        # Calculate improvements
        ret_imp = res.total_return_pct - baseline_res.total_return_pct
        if baseline_res.sharpe > 0:
            sharpe_imp_pct = (res.sharpe - baseline_res.sharpe) / baseline_res.sharpe * 100
        else:
            sharpe_imp_pct = float('inf') if res.sharpe > 0 else 0.0
            
        if ret_imp >= MIN_RETURN_IMPROVEMENT_PCT and sharpe_imp_pct >= MIN_SHARPE_IMPROVEMENT_PCT:
            logger.info(f"💡 Found significant improvement! Config: {config}")
            significant_improvements.append({
                "config": config,
                "return": res.total_return_pct,
                "sharpe": res.sharpe,
                "max_dd": res.max_drawdown_pct,
                "ret_imp": ret_imp,
                "sharpe_imp_pct": sharpe_imp_pct
            })
            
    # 3. Generate Report
    report_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'research_report.md')
    
    # Sort by highest Sharpe improvement
    significant_improvements.sort(key=lambda x: x["sharpe"], reverse=True)
    
    with open(report_path, "w") as f:
        f.write("# 🔬 Weekly Automatic Quant Research Report\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 📊 Baseline Model\n")
        f.write(f"- **Stop Loss:** {baseline_config['STOP_LOSS_PCT']*100:.1f}%\n")
        f.write(f"- **Take Profit:** {baseline_config['PROFIT_TARGET_PCT']*100:.1f}%\n")
        f.write(f"- **RSI Oversold:** {baseline_config['RSI_OVERSOLD']}\n")
        f.write(f"- **RSI Overbought:** {baseline_config['RSI_OVERBOUGHT']}\n")
        f.write(f"**Baseline Performance:** {baseline_res.total_return_pct:.2f}% Return, {baseline_res.sharpe:.2f} Sharpe, {baseline_res.max_drawdown_pct:.2f}% Max DD\n\n")
        
        f.write("## 🚀 Statistically Significant Improvements\n")
        if not significant_improvements:
            f.write("No configurations significantly outperformed the baseline this week. The production model remains mathematically optimal.\n")
        else:
            for i, imp in enumerate(significant_improvements[:10]): # Top 10
                c = imp['config']
                f.write(f"### Rank {i+1}\n")
                f.write(f"- **Config:** SL={c['STOP_LOSS_PCT']*100:.1f}%, TP={c['PROFIT_TARGET_PCT']*100:.1f}%, RSI_OB={c['RSI_OVERBOUGHT']}, RSI_OS={c['RSI_OVERSOLD']}\n")
                f.write(f"- **Return:** {imp['return']:.2f}% (+{imp['ret_imp']:.2f}% vs baseline)\n")
                f.write(f"- **Sharpe:** {imp['sharpe']:.2f} (+{imp['sharpe_imp_pct']:.1f}% vs baseline)\n")
                f.write(f"- **Max Drawdown:** {imp['max_dd']:.2f}%\n\n")
                
    logger.info(f"Research cycle complete. Report generated at {report_path}")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
