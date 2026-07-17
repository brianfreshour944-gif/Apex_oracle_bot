# risk.py
import logging
import numpy as np
import db
from config import BASE_RISK_PERCENT, MAX_SINGLE_TRADE_USD, MAX_DRAWDOWN_STOP, DAILY_LOSS_LIMIT

logger = logging.getLogger(__name__)

def get_recent_performance(bot_name='apex_oracle_bot', limit=50) -> float:
    """
    Queries raw trades from database, matches BUY/SELL orders in memory (FIFO style),
    and calculates the Kelly Criterion fraction to scale position sizes.
    """
    try:
        trades = db.query_recent_trades(bot_name=bot_name, limit=limit)
        if not trades or len(trades) < 4:
            # Default to full risk if not enough data
            return 1.0

        # Group trades by symbol to match BUY/SELL pairs
        symbol_trades = {}
        for t in reversed(trades): # Process oldest to newest
            sym = t['symbol']
            if sym not in symbol_trades:
                symbol_trades[sym] = []
            symbol_trades[sym].append(t)

        realized_pnls = []

        for sym, t_list in symbol_trades.items():
            buys = []
            for t in t_list:
                side = t['side'].upper()
                qty = t['qty']
                price = t['price']
                
                if side == 'BUY':
                    buys.append((qty, price))
                elif side == 'SELL' and buys:
                    # Match FIFO style
                    sell_qty = qty
                    realized_val = 0.0
                    cost_val = 0.0
                    
                    while sell_qty > 0 and buys:
                        buy_qty, buy_price = buys[0]
                        take_qty = min(sell_qty, buy_qty)
                        
                        cost_val += take_qty * buy_price
                        realized_val += take_qty * price
                        
                        sell_qty -= take_qty
                        # Update buy queue
                        if buy_qty > take_qty:
                            buys[0] = (buy_qty - take_qty, buy_price)
                        else:
                            buys.pop(0)
                            
                    pnl = realized_val - cost_val
                    # Convert to percentage return for scaling
                    pnl_pct = pnl / cost_val if cost_val > 0 else 0.0
                    realized_pnls.append(pnl_pct)

        if len(realized_pnls) < 3:
            return 1.0

        wins = [p for p in realized_pnls if p > 0]
        losses = [p for p in realized_pnls if p <= 0]
        
        win_count = len(wins)
        total_count = len(realized_pnls)
        win_rate = win_count / total_count
        
        avg_win = sum(wins) / win_count if win_count > 0 else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if len(losses) > 0 else 0.02 # default 2% loss representation
        
        r_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        # Kelly Fraction: f* = W - (1 - W)/R
        kelly_fraction = win_rate - (1 - win_rate) / r_ratio if r_ratio > 0 else 0.0
        
        # Safe Half-Kelly
        safe_kelly = kelly_fraction * 0.5
        scale_factor = float(max(0.1, min(safe_kelly, 1.0)))
        
        logger.info(f"📊 Kelly Analysis (In-Memory FIFO): WinRate={win_rate*100:.1f}%, Win/Loss={r_ratio:.2f}, Kelly={kelly_fraction:.2f} -> Scaling={scale_factor:.2f}")
        return scale_factor

    except Exception as e:
        logger.error(f"Error calculating Kelly scaling: {e}")
        return 1.0


def check_account_killswitches(equity: float, starting_equity: float, bot_name='apex_oracle_bot') -> tuple:
    """
    Checks drawdown and daily loss limits from the database.
    Returns (breached: bool, reason: str)
    """
    if starting_equity <= 0:
        return False, ""
        
    # 1. Lifetime Drawdown check
    lifetime_drawdown = ((equity - starting_equity) / starting_equity) * 100.0
    if lifetime_drawdown <= MAX_DRAWDOWN_STOP:
        reason = f"Lifetime drawdown limit hit: {lifetime_drawdown:.2f}% (Limit: {MAX_DRAWDOWN_STOP}%)"
        return True, reason
        
    # 2. Daily Loss Limit check
    try:
        with db.engine.connect() as conn:
            result = conn.execute(
                db.text("SELECT starting_equity FROM bot_status WHERE bot_name = :bot_name"),
                {"bot_name": bot_name}
            ).first()
            if result:
                daily_start_equity = float(result[0])
                daily_loss = ((equity - daily_start_equity) / daily_start_equity) * 100.0
                if daily_loss <= DAILY_LOSS_LIMIT:
                    reason = f"Daily loss limit hit: {daily_loss:.2f}% (Limit: {DAILY_LOSS_LIMIT}%)"
                    return True, reason
    except Exception as e:
        logger.error(f"Error checking daily loss limit: {e}")
        
    return False, ""


def calculate_position_size(equity: float, price: float, atr: float, multiplier: float = 2.0) -> float:
    """
    Calculates volatility-adjusted position size in units.
    Size = (Equity * RiskPercent * KellyMultiplier) / (ATR * ATR_Multiplier)
    """
    if price <= 0 or atr <= 0:
        return 0.0
        
    # Base dollar risk
    risk_budget_usd = equity * BASE_RISK_PERCENT
    
    # Scale risk based on recent performance (Kelly Criterion)
    kelly_scale = get_recent_performance()
    scaled_risk_usd = risk_budget_usd * kelly_scale
    
    # Stop loss distance in USD
    stop_distance_usd = atr * multiplier
    
    # Check that stop loss isn't tiny
    if stop_distance_usd <= 0:
        return 0.0
        
    # Calculate target trade quantity
    qty = scaled_risk_usd / stop_distance_usd
    trade_value = qty * price
    
    # Cap position size according to hard constraints
    if trade_value > MAX_SINGLE_TRADE_USD:
        qty = MAX_SINGLE_TRADE_USD / price
        
    return float(qty)
