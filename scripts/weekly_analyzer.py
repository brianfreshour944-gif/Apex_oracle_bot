import os
import json
import sys
import pandas as pd
from sqlalchemy import create_engine
from typing import Dict, Any

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings

def main():
    db_path = settings.DATABASE_URL
    print(f"Connecting to database: {db_path}")
    
    try:
        engine = create_engine(db_path)
        
        # Load closed trades
        query = "SELECT * FROM decision_snapshots WHERE status = 'closed'"
        try:
            df = pd.read_sql(query, engine)
        except Exception as e:
            if 'no such table' in str(e):
                print("No database or tables found yet. Start the bot to initialize it.")
                return
            raise e
        
        if df.empty:
            print("No closed trades found yet.")
            return

        print(f"\n--- Weekly Analysis Report ({len(df)} total trades) ---")
        
        # Calculate metrics by symbol
        symbols_metrics = {}
        for symbol, group in df.groupby('symbol'):
            total_trades = len(group)
            winning_trades = len(group[group['realized_pnl'] > 0])
            losing_trades = len(group[group['realized_pnl'] <= 0])
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
            
            gross_profit = group[group['realized_pnl'] > 0]['realized_pnl'].sum()
            gross_loss = abs(group[group['realized_pnl'] <= 0]['realized_pnl'].sum())
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            if gross_profit == 0 and gross_loss == 0:
                profit_factor = 0.0
                
            symbols_metrics[symbol] = {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "net_pnl": gross_profit - gross_loss
            }
            
            print(f"\n{symbol}:")
            print(f"  Trades: {total_trades}")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Profit Factor: {profit_factor:.2f}")
            print(f"  Net PnL: ${gross_profit - gross_loss:.2f}")
            
        # Ban underperforming symbols
        MIN_TRADES_TO_BAN = 5
        MIN_PROFIT_FACTOR = 0.8
        
        banned_symbols = []
        for symbol, metrics in symbols_metrics.items():
            if metrics["total_trades"] >= MIN_TRADES_TO_BAN and metrics["profit_factor"] < MIN_PROFIT_FACTOR:
                banned_symbols.append({
                    "symbol": symbol,
                    "reason": f"Profit Factor {metrics['profit_factor']:.2f} < {MIN_PROFIT_FACTOR} over {metrics['total_trades']} trades."
                })
                
        # Save to banned_symbols.json
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        ban_file = os.path.join(data_dir, 'banned_symbols.json')
        
        # Load existing so we don't accidentally unban someone if we change the query window later
        existing_bans = []
        if os.path.exists(ban_file):
            try:
                with open(ban_file, 'r') as f:
                    existing_bans = json.load(f)
            except:
                pass
                
        # Merge lists
        merged_bans = {b["symbol"]: b for b in existing_bans}
        for b in banned_symbols:
            merged_bans[b["symbol"]] = b
            
        with open(ban_file, 'w') as f:
            json.dump(list(merged_bans.values()), f, indent=4)
            
        print("\n--- Banned Symbols ---")
        if merged_bans:
            for s, info in merged_bans.items():
                print(f"❌ {s}: {info['reason']}")
        else:
            print("✅ All symbols performing acceptably.")

        # --- Threshold Optimization (Level 5) ---
        print("\n--- Adaptive Threshold Optimization ---")
        optimal_thresholds = {}
        for symbol, group in df.groupby('symbol'):
            best_threshold = 0.60
            best_pnl = float('-inf')
            # Only optimize if we have at least 10 trades to avoid curve fitting small samples
            if len(group) >= 10:
                import numpy as np
                for t in np.arange(0.50, 0.86, 0.01):
                    # Simulate taking only trades with confidence >= t
                    sim_trades = group[group['confidence'] >= t]
                    if len(sim_trades) >= 5:
                        sim_pnl = sim_trades['realized_pnl'].sum()
                        if sim_pnl > best_pnl:
                            best_pnl = sim_pnl
                            best_threshold = t
                
                if best_pnl > 0:
                    optimal_thresholds[symbol] = round(float(best_threshold), 2)
                    print(f"🧠 {symbol}: Learned optimal confidence threshold = {best_threshold:.2f} (Simulated PnL: ${best_pnl:.2f})")
                else:
                    print(f"🧠 {symbol}: No profitable threshold found. Defaulting to 0.60.")
            else:
                print(f"🧠 {symbol}: Insufficient data ({len(group)} trades). Defaulting to 0.60.")

        thresh_file = os.path.join(data_dir, 'adaptive_thresholds.json')
        # Load existing so we don't overwrite symbols with no recent data
        existing_thresh = {}
        if os.path.exists(thresh_file):
            try:
                with open(thresh_file, 'r') as f:
                    existing_thresh = json.load(f)
            except:
                pass
        
        existing_thresh.update(optimal_thresholds)
        with open(thresh_file, 'w') as f:
            json.dump(existing_thresh, f, indent=4)

    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
