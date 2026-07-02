# bot.py
import asyncio
import logging
import time
import json
from datetime import datetime, timezone

import config
from db import (
    init_db, log_trade, update_bot_status, query_recent_trades,
    reset_daily_starting_equity, SessionLocal, TradeLog, BotStatus,
)
from strategies import analyze_market_regime, generate_trading_signal, calculate_atr
from risk import check_account_killswitches, calculate_position_size
import notifier
from exchange import OKXExchange

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("bot")

# Exchange + global trading state
ex = OKXExchange()
entry_times = {}        # symbol -> entry timestamp (epoch seconds)
cooldown_until = {}     # symbol -> epoch seconds when trading is allowed again
start_equity = None
daily_reset_time = None


# ---------- DB helpers ----------
def get_last_buy(symbol: str):
    """Returns the most recent BUY TradeLog row for a symbol, or None."""
    session = SessionLocal()
    try:
        return (
            session.query(TradeLog)
            .filter(TradeLog.symbol == symbol, TradeLog.side == "BUY")
            .order_by(TradeLog.timestamp.desc())
            .first()
        )
    except Exception as e:
        logger.error(f"get_last_buy failed for {symbol}: {e}")
        return None
    finally:
        session.close()


def get_entry_price(symbol: str, fallback: float) -> float:
    row = get_last_buy(symbol)
    return float(row.price) if row else float(fallback)


# ---------- Startup position sync ----------
async def sync_positions_on_startup(positions: dict):
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
    """Places a market order, logs it, and notifies. side is 'BUY' or 'SELL'."""
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

        log_trade(symbol=symbol, side=side, qty=clean_qty, price=filled_price,
                  pnl=realized_pnl, order_id=order_id)
        await notifier.send_trade_alert(
            symbol=symbol, side=side, qty=clean_qty, price=filled_price,
            order_id=str(order_id) if order_id else None,
            pnl=realized_pnl, is_entry=(side == "BUY"),
        )
        logger.info(f"Trade complete: {side} {clean_qty:.6f} {symbol} @ {filled_price}")
        return True
    except Exception as e:
        logger.error(f"Order submission failed for {symbol} ({side}): {e}")
        return False


async def liquidate_all_positions(reason: str, positions: dict):
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


# ---------- Observability HTTP status server ----------
async def handle_status_request(reader, writer):
    try:
        session = SessionLocal()
        status_rec = session.get(BotStatus, config.BOT_NAME)
        recent = query_recent_trades(bot_name=config.BOT_NAME, limit=5)
        session.close()

        stats = {
            "status": status_rec.status if status_rec else "unknown",
            "time": datetime.now(timezone.utc).isoformat(),
            "equity": status_rec.live_equity if status_rec else 0.0,
            "starting_equity": status_rec.starting_equity if status_rec else 0.0,
            "buying_power": status_rec.buying_power if status_rec else 0.0,
            "daily_pnl_pct": status_rec.daily_pnl_pct if status_rec else 0.0,
            "open_positions": status_rec.open_positions_count if status_rec else 0,
            "recent_trades": [
                {"symbol": t["symbol"], "side": t["side"], "qty": t["qty"],
                 "price": t["price"], "timestamp": t["timestamp"].isoformat()}
                for t in recent
            ],
        }
        body = json.dumps(stats, indent=2)
        response = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
            f"{body}"
        )
    except Exception as e:
        body = json.dumps({"status": "error", "detail": str(e)})
        response = (
            "HTTP/1.1 500 Internal Server Error\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
            f"{body}"
        )
        logger.error(f"Status server handler error: {e}")
    writer.write(response.encode("utf-8"))
    await writer.drain()
    writer.close()


async def start_observability_server():
    try:
        server = await asyncio.start_server(handle_status_request, "0.0.0.0", config.STATUS_PORT)
        logger.info(f"Observability API active on http://0.0.0.0:{config.STATUS_PORT}/status")
        async with server:
            await server.serve_forever()
    except Exception as e:
        logger.error(f"Failed to start observability server: {e}")


# ---------- Core loop ----------
async def run_trading_bot():
    global start_equity, daily_reset_time

    logger.info("Initializing database...")
    init_db()
    config.log_config()

    await ex.load()

    # Initial account snapshot
    try:
        equity, buying_power, positions = await ex.get_account_snapshot(
            config.SYMBOLS, config.QUOTE_CURRENCY
        )
        start_equity = equity if equity > 0 else config.ACCOUNT_BASE
        logger.info(f"Account connected. Equity: ${start_equity:,.2f} | Buying power: ${buying_power:,.2f}")
    except Exception as e:
        logger.critical(f"OKX connection failure on startup: {e}")
        await ex.close()
        return

    await sync_positions_on_startup(positions)
    reset_daily_starting_equity(start_equity)

    asyncio.create_task(start_observability_server())

    daily_reset_time = datetime.now(timezone.utc).date()
    cycle_counter = 0

    while True:
        try:
            # Daily baseline reset at UTC midnight
            current_date = datetime.now(timezone.utc).date()
            if current_date > daily_reset_time:
                snap = await ex.get_account_snapshot(config.SYMBOLS, config.QUOTE_CURRENCY)
                start_equity = snap[0]
                daily_reset_time = current_date
                reset_daily_starting_equity(start_equity)
                logger.info(f"New day reset. Base equity: ${start_equity:,.2f}")

            equity, buying_power, positions = await ex.get_account_snapshot(
                config.SYMBOLS, config.QUOTE_CURRENCY
            )
            open_count = len(positions)
            portfolio_exposure = sum(p["market_value"] for p in positions.values())
            daily_pnl_pct = ((equity - start_equity) / start_equity) * 100.0 if start_equity > 0 else 0.0

            update_bot_status(
                starting_equity=start_equity, live_equity=equity, buying_power=buying_power,
                daily_pnl_pct=daily_pnl_pct, open_positions_count=open_count, status="running",
            )

            breached, reason = check_account_killswitches(equity, start_equity)
            if breached:
                await liquidate_all_positions(reason, positions)
                update_bot_status(
                    starting_equity=start_equity, live_equity=equity, buying_power=buying_power,
                    daily_pnl_pct=daily_pnl_pct, open_positions_count=0, status="stopped",
                )
                logger.critical("Bot halted due to risk killswitch.")
                break

            if cycle_counter % 30 == 0:
                await notifier.send_heartbeat_alert(equity, daily_pnl_pct, open_count, buying_power)
            cycle_counter += 1

            # ---------- Dust cleanup (updated) ----------
            for symbol in list(positions.keys()):
                if positions[symbol]["market_value"] < config.DUST_VALUE_USD:
                    # Retrieve the exchange‑specific minimum order quantity.
                    # You need to implement `ex.get_min_qty(symbol)` in exchange.py.
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
            for symbol in config.SYMBOLS:
                now_ts = time.time()
                if now_ts < cooldown_until.get(symbol, 0):
                    continue

                try:
                    df = await ex.fetch_ohlcv_df(symbol, timeframe="5m", limit=200)
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    continue
                if df.empty or len(df) < 50:
                    continue

                regime_info = analyze_market_regime(df)
                signal = generate_trading_signal(df, regime_info)
                current_price = float(df["close"].iloc[-1])
                atr = float(calculate_atr(df).iloc[-1])
                if current_price <= 0:
                    continue

                has_position = symbol in positions

                # ----- EXIT -----
                if has_position:
                    qty_held = positions[symbol]["qty"]
                    avg_entry = get_entry_price(symbol, current_price)
                    pnl_pct = (current_price - avg_entry) / avg_entry if avg_entry > 0 else 0.0
                    held_hours = (now_ts - entry_times.get(symbol, now_ts)) / 3600.0

                    exit_triggered = False
                    exit_reason = ""
                    if pnl_pct >= config.PROFIT_TARGET_PCT:
                        exit_triggered, exit_reason = True, f"Profit target ({pnl_pct*100:+.2f}%)"
                    elif pnl_pct <= -config.STOP_LOSS_PCT:
                        exit_triggered, exit_reason = True, f"Stop loss ({pnl_pct*100:+.2f}%)"
                    elif held_hours >= config.MAX_HOLD_HOURS:
                        exit_triggered, exit_reason = True, f"Max hold time ({held_hours:.1f}h)"
                    elif signal == "SELL":
                        exit_triggered, exit_reason = True, "Opposing sell signal"

                    if exit_triggered:
                        logger.info(f"Exit {symbol} | {exit_reason}")
                        if await execute_trade(symbol, "SELL", qty_held, current_price):
                            entry_times.pop(symbol, None)
                            cooldown_until[symbol] = now_ts + 1800
                            open_count -= 1
                    else:
                        logger.info(
                            f"Holding {symbol} | Entry ${avg_entry:.2f} | Price ${current_price:.2f} | "
                            f"PnL {pnl_pct*100:+.2f}% | Regime {regime_info['regime']} | Held {held_hours:.1f}h"
                        )

                # ----- ENTRY -----
                else:
                    if signal == "BUY" and open_count < config.MAX_OPEN_POSITIONS:
                        qty_to_buy = calculate_position_size(
                            equity, current_price, atr, multiplier=config.ATR_STOP_MULTIPLIER
                        )
                        trade_cost = qty_to_buy * current_price

                        if portfolio_exposure + trade_cost > config.MAX_PORTFOLIO_VALUE:
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

            await asyncio.sleep(config.LOOP_INTERVAL_SEC)

        except Exception as e:
            logger.error(f"Critical loop error: {e}", exc_info=True)
            await asyncio.sleep(30)

    await ex.close()


if __name__ == "__main__":
    try:
        asyncio.run(run_trading_bot())
    except KeyboardInterrupt:
        logger.info("Shutdown requested. Exiting.")

