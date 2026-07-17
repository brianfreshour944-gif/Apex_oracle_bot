"""Modern trading bot with comprehensive type hints and robust error handling."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List
import polars as pl

from src.logging_config import get_logger
from src.config import settings
from src.db import (
    init_db,
    log_trade,
    update_bot_status,
    query_recent_trades,
    reset_daily_starting_equity,
    get_last_buy,
    get_entry_price,
)
from src.exchange import AlpacaExchange
from src.api import start_fastapi_server_async
import src.notifier as notifier
import src.strategies as strategies
import src.risk as risk

# Initialize structured logger
logger = get_logger("bot")

# Exchange + global trading state
ex: AlpacaExchange = AlpacaExchange()
entry_times: Dict[str, float] = {}        # symbol -> entry timestamp (epoch seconds)
cooldown_until: Dict[str, float] = {}     # symbol -> epoch seconds when trading is allowed again
trailing_peak: Dict[str, float] = {}      # symbol -> highest price seen since trailing stop activated
start_equity: Optional[float] = None
daily_reset_time: Optional[datetime.date] = None

# ---------- DB helpers ----------
def get_last_buy(symbol: str) -> Optional[Any]:
    """Get the most recent BUY trade for a symbol."""
    return get_last_buy(symbol)

def get_entry_price(symbol: str, fallback: float) -> float:
    """Get entry price for a symbol with fallback."""
    return get_entry_price(symbol, fallback)

# ---------- Startup position sync ----------
async def sync_positions_on_startup(positions: Dict[str, Any]) -> None:
    """Recover entry timestamps for already-held balances from the trade log."""
    logger.info("Syncing existing holdings on startup...")
    if not positions:
        logger.info("No open holdings found.")
        return

    for symbol in positions:
        row = get_last_buy(symbol)
        if row:
            entry_times[symbol] = row.timestamp.replace(tzinfo=timezone.utc).timestamp()
            logger.info(f"Synced {symbol} entry from DB ({row.timestamp}).")
        else:
            entry_times[symbol] = time.time()
            logger.warning(f"No DB entry for held {symbol}; resetting hold timer to now.")

# ---------- Order execution ----------
async def execute_trade(symbol: str, side: str, qty: float, price_estimate: float) -> bool:
    """Place a market order, log it, and notify.

    Args:
        symbol: Trading symbol
        side: 'BUY' or 'SELL'
        qty: Quantity to trade
        price_estimate: Estimated execution price

    Returns:
        True on success, False on failure
    """
    clean_qty = ex.amount_to_precision(symbol, qty)
    if clean_qty <= 0:
        logger.warning(f"Quantity rounded to 0 for {symbol}; skipping order.")
        return False

    try:
        if side == "BUY":
            order = await ex.market_buy(symbol, clean_qty)
        else:
            order = await ex.market_sell(symbol, clean_qty)

        order_id = order.get("id")
        filled_price = await ex.get_filled_price(order_id, symbol, price_estimate)

        realized_pnl = None
        if side == "SELL":
            row = get_last_buy(symbol)
            if row:
                realized_pnl = (filled_price - float(row.price)) * clean_qty

        logged_ok = log_trade(
            symbol=symbol,
            side=side,
            quantity=clean_qty,
            price=filled_price,
            pnl=realized_pnl,
            order_id=order_id
        )

        if not logged_ok:
            # Critical: trade filled but not logged
            logger.critical(
                f"Trade filled but NOT logged to DB: {side} {clean_qty:.6f} "
                f"{symbol} @ {filled_price} (order_id={order_id}). "
                f"Manual reconciliation required."
            )
            try:
                await notifier.send_killswitch_alert(
                    f"DB LOGGING FAILURE: filled {side} {clean_qty:.6f} {symbol} "
                    f"@ {filled_price} was NOT recorded. Position tracking is now "
                    f"out of sync -- reconcile manually."
                )
            except Exception as e:
                logger.error(f"Failed to send DB-logging-failure alert: {e}")

        await notifier.send_trade_alert(
            symbol=symbol,
            side=side,
            qty=clean_qty,
            price=filled_price,
            order_id=str(order_id) if order_id else None,
            pnl=realized_pnl,
            is_entry=(side == "BUY"),
        )

        logger.info(f"Trade complete: {side} {clean_qty:.6f} {symbol} @ {filled_price}")
        return True

    except Exception as e:
        logger.error(f"Order submission failed for {symbol} ({side}): {e}")
        return False

async def liquidate_all_positions(reason: str, positions: Dict[str, Any]) -> None:
    """Market-sells every held base asset immediately."""
    logger.critical(f"LIQUIDATING ALL POSITIONS: {reason}")
    for symbol, pos in positions.items():
        try:
            await execute_trade(symbol, "SELL", pos["qty"], pos["price"])
        except Exception as e:
            logger.error(f"Failed to liquidate {symbol}: {e}")

    try:
        await notifier.send_killswitch_alert(reason)
    except Exception as e:
        logger.error(f"Killswitch alert failed: {e}")

# ---------- Core loop ----------
async def run_trading_bot() -> None:
    """Main trading bot loop with modern error handling."""
    global start_equity, daily_reset_time

    logger.info("Initializing database...")
    init_db()
    logger.info(settings.log_config())

    await ex.load()

    # Initial account snapshot
    try:
        equity, buying_power, positions = await ex.get_account_snapshot(
            settings.SYMBOLS,
            settings.QUOTE_CURRENCY
        )
        start_equity = equity if equity > 0 else settings.ACCOUNT_BASE
        logger.info(f"Account connected. Equity: ${start_equity:,.2f} | Buying power: ${buying_power:,.2f}")
    except Exception as e:
        logger.critical(f"Alpaca connection failure on startup: {e}")
        await ex.close()
        return

    await sync_positions_on_startup(positions)
    reset_daily_starting_equity(start_equity)

    # Start FastAPI server in background
    await start_fastapi_server_async()

    daily_reset_time = datetime.now(timezone.utc).date()
    cycle_counter = 0

    while True:
        try:
            # Daily baseline reset at UTC midnight
            current_date = datetime.now(timezone.utc).date()
            if current_date > daily_reset_time:
                snap = await ex.get_account_snapshot(settings.SYMBOLS, settings.QUOTE_CURRENCY)
                start_equity = snap[0]
                daily_reset_time = current_date
                reset_daily_starting_equity(start_equity)
                logger.info(f"New day reset. Base equity: ${start_equity:,.2f}")

            equity, buying_power, positions = await ex.get_account_snapshot(
                settings.SYMBOLS,
                settings.QUOTE_CURRENCY
            )
            open_count = len(positions)
            portfolio_exposure = sum(p["market_value"] for p in positions.values())
            daily_pnl_pct = ((equity - start_equity) / start_equity) * 100.0 if start_equity > 0 else 0.0

            update_bot_status(
                starting_equity=start_equity,
                live_equity=equity,
                buying_power=buying_power,
                daily_pnl_pct=daily_pnl_pct,
                open_positions_count=open_count,
                status="running",
            )

            breached, reason = risk.check_account_killswitches(equity, start_equity)
            if breached:
                await liquidate_all_positions(reason, positions)
                update_bot_status(
                    starting_equity=start_equity,
                    live_equity=equity,
                    buying_power=buying_power,
                    daily_pnl_pct=daily_pnl_pct,
                    open_positions_count=0,
                    status="stopped",
                )
                logger.critical("Bot halted due to risk killswitch.")
                break

            if cycle_counter % 30 == 0:
                await notifier.send_heartbeat_alert(equity, daily_pnl_pct, open_count, buying_power)
            cycle_counter += 1

            # ---------- Dust cleanup ----------
            for symbol in list(positions.keys()):
                if positions[symbol]["market_value"] < settings.DUST_VALUE_USD:
                    min_qty = ex.get_min_qty(symbol)

                    if positions[symbol]["qty"] >= min_qty:
                        # Quantity meets the exchange minimum – safe to sell.
                        await execute_trade(
                            symbol,
                            "SELL",
                            positions[symbol]["qty"],
                            positions[symbol]["price"]
                        )
                    else:
                        # Quantity too small; skip the sell to avoid a precision‑error.
                        logger.info(
                            f"IGNORING dust {symbol}: qty {positions[symbol]['qty']} "
                            f"is below exchange min {min_qty}"
                        )

                    # Clean up local state regardless of whether we sold or ignored.
                    positions.pop(symbol, None)
                    entry_times.pop(symbol, None)

            open_count = len(positions)
            portfolio_exposure = sum(p["market_value"] for p in positions.values())

            # ---------- Evaluate each asset ----------
            for symbol in settings.SYMBOLS:
                now_ts = time.time()
                if now_ts < cooldown_until.get(symbol, 0):
                    continue

                try:
                    df = await ex.fetch_ohlcv_df(symbol, timeframe="5m", limit=200)
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    continue

                if df.is_empty() or len(df) < 50:
                    continue

                # Tag the frame so the per-symbol Kalman filter cache keys correctly.
                df = df.with_columns(pl.lit(symbol).alias("symbol"))

                regime_info = strategies.analyze_market_regime(df)
                signal = strategies.generate_trading_signal(df, regime_info)
                current_price = float(df["close"].item())
                atr = float(strategies.calculate_atr(df).item())

                if current_price <= 0:
                    continue

                has_position = symbol in positions

                # ----- EXIT -----
                if has_position:
                    qty_held = positions[symbol]["qty"]
                    avg_entry = get_entry_price(symbol, current_price)
                    pnl_pct = (current_price - avg_entry) / avg_entry if avg_entry > 0 else 0.0
                    held_hours = (now_ts - entry_times.get(symbol, now_ts)) / 3600.0

                    # --- Trailing stop (protects open profits) ---
                    if settings.TRAILING_STOP_ENABLED:
                        if pnl_pct >= settings.TRAILING_ACTIVATION_PCT:
                            prev_peak = trailing_peak.get(symbol)
                            if prev_peak is None or current_price > prev_peak:
                                trailing_peak[symbol] = current_price
                        peak = trailing_peak.get(symbol)
                        if peak is not None and current_price <= peak * (1.0 - settings.TRAILING_DISTANCE_PCT):
                            trailing_peak.pop(symbol, None)
                            exit_triggered, exit_reason = True, f"Trailing stop (peak ${peak:.2f} -> ${current_price:.2f})"
                        else:
                            exit_triggered, exit_reason = False, ""
                    else:
                        exit_triggered, exit_reason = False, ""

                    if not exit_triggered:
                        if pnl_pct >= settings.PROFIT_TARGET_PCT:
                            exit_triggered, exit_reason = True, f"Profit target ({pnl_pct*100:+.2f}%)"
                        elif pnl_pct <= -settings.STOP_LOSS_PCT:
                            exit_triggered, exit_reason = True, f"Stop loss ({pnl_pct*100:+.2f}%)"
                        elif held_hours >= settings.MAX_HOLD_HOURS:
                            exit_triggered, exit_reason = True, f"Max hold time ({held_hours:.1f}h)"
                        elif signal == "SELL":
                            exit_triggered, exit_reason = True, "Opposing sell signal"

                    if exit_triggered:
                        logger.info(f"Exit {symbol} | {exit_reason}")
                        if await execute_trade(symbol, "SELL", qty_held, current_price):
                            entry_times.pop(symbol, None)
                            trailing_peak.pop(symbol, None)
                            cooldown_until[symbol] = now_ts + 1800
                            open_count -= 1
                    else:
                        logger.info(
                            f"Holding {symbol} | Entry ${avg_entry:.2f} | Price ${current_price:.2f} | "
                            f"PnL {pnl_pct*100:+.2f}% | Regime {regime_info['regime']} | Held {held_hours:.1f}h"
                        )

                # ----- ENTRY -----
                else:
                    if signal == "BUY" and open_count < settings.MAX_OPEN_POSITIONS:
                        qty_to_buy = risk.calculate_position_size(
                            equity, current_price, atr, multiplier=settings.ATR_STOP_MULTIPLIER
                        )
                        trade_cost = qty_to_buy * current_price

                        if portfolio_exposure + trade_cost > settings.MAX_PORTFOLIO_VALUE:
                            logger.warning(f"BUY suppressed {symbol}: exposure cap would breach.")
                            continue
                        if buying_power < trade_cost:
                            logger.warning(
                                f"BUY suppressed {symbol}: insufficient buying power "
                                f"(${buying_power:.2f} < ${trade_cost:.2f})"
                            )
                            continue
                        if qty_to_buy > 0:
                            logger.info(
                                f"Buy signal {symbol} | Regime {regime_info['regime']} | "
                                f"Vol {regime_info['volatility_pct']}%"
                            )
                            if await execute_trade(symbol, "BUY", qty_to_buy, current_price):
                                entry_times[symbol] = now_ts
                                cooldown_until[symbol] = now_ts + 900
                                open_count += 1
                                portfolio_exposure += trade_cost
                                buying_power -= trade_cost

                await asyncio.sleep(2)

            await asyncio.sleep(settings.LOOP_INTERVAL_SEC)

        except Exception as e:
            logger.error(f"Critical loop error: {e}", exc_info=True)
            await asyncio.sleep(30)

    await ex.close()

if __name__ == "__main__":
    try:
        asyncio.run(run_trading_bot())
    except KeyboardInterrupt:
        logger.info("Shutdown requested. Exiting.")