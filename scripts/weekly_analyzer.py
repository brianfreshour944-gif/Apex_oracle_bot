import os
import json
import sys
from datetime import date

import pandas as pd
from sqlalchemy import create_engine
from typing import Dict, Any

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.logging_config import get_logger

logger = get_logger("weekly_analyzer")

# --- Ban policy ---------------------------------------------------------------
# A symbol is banned when its profit factor is below MIN_PROFIT_FACTOR over at
# least MIN_TRADES_TO_BAN trades.
MIN_TRADES_TO_BAN = 5
MIN_PROFIT_FACTOR = 0.8
# ...but a ban must not be permanent: every run used to merge the fresh bans
# over the persisted file with no removal path at all, so one bad week
# permanently disabled a symbol with no way back short of hand-deleting
# data/banned_symbols.json. A ban is now lifted when either
#   - the symbol's fresh metrics show recovery (>= MIN_TRADES_TO_BAN trades AND
#     profit factor >= MIN_PROFIT_FACTOR), or
#   - the ban is older than BAN_TTL_DAYS, i.e. its evidence is stale and the
#     symbol deserves a fresh probationary window.
BAN_TTL_DAYS = 30

# --- Adaptive threshold policy ------------------------------------------------
# The committee falls back to settings.DEFAULT_SCORE_THRESHOLD (0.15) when a
# symbol has no learned threshold. Learned thresholds are only persisted when
# the simulated PnL is positive, and are CLAMPED to MAX_LEARNED_SCORE_THRESHOLD:
# the sweep below searches up to 0.85, but a normal committee consensus scores
# around 0.2-0.6 (a single directional vote among abstentions scores ~0.2), so a
# high persisted threshold silently gates out every entry -- the same class of
# failure as the adversarial-veto bug it would compound.
MAX_LEARNED_SCORE_THRESHOLD = 0.45


def _parse_ban_date(value, fallback):
    """Parse a stored ban date, falling back for legacy/malformed entries.

    Bans written before `banned_at` existed have no date. Treating those as
    "banned today" (rather than assuming they are ancient) means the TTL cannot
    retroactively un-ban the whole file on the first run after this change.
    """
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return fallback


def merge_bans(existing_bans, newly_banned, symbols_metrics, today=None):
    """Merge fresh bans into the persisted list, lifting recovered/expired ones.

    Pure function (no I/O) so the policy can be unit-tested without a database.

    Returns (bans_to_persist, lifted) where `lifted` is a list of
    (symbol, reason) pairs describing every ban removed this run.
    """
    today = today or date.today()
    kept = {}
    lifted = []

    for ban in existing_bans or []:
        symbol = ban.get("symbol")
        if not symbol:
            continue
        banned_at = _parse_ban_date(ban.get("banned_at"), today)
        metrics = (symbols_metrics or {}).get(symbol)

        if (
            metrics
            and metrics.get("total_trades", 0) >= MIN_TRADES_TO_BAN
            and metrics.get("profit_factor", 0.0) >= MIN_PROFIT_FACTOR
        ):
            lifted.append((
                symbol,
                f"recovered: profit factor {metrics['profit_factor']:.2f} >= {MIN_PROFIT_FACTOR} "
                f"over {metrics['total_trades']} trades",
            ))
            continue

        age_days = (today - banned_at).days
        if age_days > BAN_TTL_DAYS:
            lifted.append((symbol, f"ban expired after {age_days} days (TTL {BAN_TTL_DAYS})"))
            continue

        kept[symbol] = {**ban, "banned_at": banned_at.isoformat()}

    # Fresh bans win over a lift in the same run (they were computed from the
    # same metrics, so this only matters for conflicting/stale input files).
    for ban in newly_banned or []:
        kept[ban["symbol"]] = {**ban, "banned_at": today.isoformat()}

    return list(kept.values()), lifted


def main() -> int:
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
        
        # Load the persisted bans so a symbol is not re-banned/cleared by a
        # change of query window. merge_bans() decides which of them survive
        # (recovery or TTL) and timestamps new ones.
        existing_bans = []
        if os.path.exists(ban_file):
            try:
                with open(ban_file, 'r') as f:
                    existing_bans = json.load(f)
            except Exception as ban_read_err:
                logger.warning(f"Could not read existing bans ({ban_read_err}) - starting a fresh list.")

        merged_bans_list, lifted_bans = merge_bans(existing_bans, banned_symbols, symbols_metrics)
        merged_bans = {b["symbol"]: b for b in merged_bans_list}

        with open(ban_file, 'w') as f:
            json.dump(merged_bans_list, f, indent=4)
            
        print("\n--- Banned Symbols ---")
        if merged_bans:
            for s, info in merged_bans.items():
                print(f"❌ {s}: {info['reason']} (since {info.get('banned_at', '?')})")
        else:
            print("✅ All symbols performing acceptably.")
        for s, why in lifted_bans:
            logger.info(f"♻️   {s}: ban lifted - {why}")

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
                    raw_threshold = round(float(best_threshold), 2)
                    if raw_threshold > MAX_LEARNED_SCORE_THRESHOLD:
                        logger.info(
                            f"🧠 {symbol}: best confidence threshold {raw_threshold:.2f} exceeds "
                            f"MAX_LEARNED_SCORE_THRESHOLD ({MAX_LEARNED_SCORE_THRESHOLD:.2f}) - clamping. "
                            f"A higher persisted threshold would silently gate out every entry the "
                            f"committee can realistically produce."
                        )
                    optimal_thresholds[symbol] = min(raw_threshold, MAX_LEARNED_SCORE_THRESHOLD)
                    print(f"🧠 {symbol}: Learned optimal confidence threshold = {optimal_thresholds[symbol]:.2f} (Simulated PnL: ${best_pnl:.2f})")
                else:
                    # Nothing is persisted here: the symbol keeps whatever
                    # threshold it already had (or the committee default). The
                    # previous wording claimed a 0.60 default that was never
                    # written anywhere.
                    print(
                        f"🧠 {symbol}: No profitable threshold found in the 0.50-0.85 search band. "
                        f"Keeping the existing/default threshold (committee default "
                        f"{settings.DEFAULT_SCORE_THRESHOLD:.2f})."
                    )
            else:
                print(
                    f"🧠 {symbol}: Insufficient data ({len(group)} trades < 10). Keeping the "
                    f"existing/default threshold (committee default {settings.DEFAULT_SCORE_THRESHOLD:.2f})."
                )

        thresh_file = os.path.join(data_dir, 'adaptive_thresholds.json')
        # Load existing so we don't overwrite symbols with no recent data
        existing_thresh = {}
        if os.path.exists(thresh_file):
            try:
                with open(thresh_file, 'r') as f:
                    existing_thresh = json.load(f)
            except Exception as thresh_read_err:
                logger.warning(f"Could not read existing thresholds ({thresh_read_err}) - rebuilding from this run.")
        
        existing_thresh.update(optimal_thresholds)
        with open(thresh_file, 'w') as f:
            json.dump(existing_thresh, f, indent=4)

    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
