# notifier.py
import logging
import aiohttp
from datetime import datetime, timezone
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DASHBOARD_API_KEY, BASE44_API_URL

logger = logging.getLogger(__name__)

# ========================= TELEGRAM CLIENT =========================

async def send_telegram_message(text: str, parse_mode: str = "HTML") -> bool:
    """Sends a Telegram message. Bypassed if credentials are missing."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug(f"[Telegram Bypass] {text}")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=8) as response:
                if response.status == 200:
                    return True
                else:
                    err_resp = await response.text()
                    logger.error(f"Telegram API error (Status {response.status}): {err_resp}")
                    return False
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")
        return False


async def send_trade_alert(symbol: str, side: str, qty: float, price: float, order_id: str = None, pnl: float = None, is_entry: bool = True):
    """Sends a trade alert to Telegram."""
    emoji = "🟢 [BUY]" if side.upper() == "BUY" else "🔴 [SELL]"
    action_type = "Entry" if is_entry else "Exit"
    
    msg = (
        f"🤖 <b>Apex Oracle Bot - Trade Alert</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Action:</b> {emoji} ({action_type})\n"
        f"<b>Asset:</b> <code>{symbol}</code>\n"
        f"<b>Quantity:</b> <code>{qty:.6f}</code>\n"
        f"<b>Price:</b> <code>${price:,.4f}</code>\n"
        f"<b>Value:</b> <code>${qty * price:,.2f}</code>\n"
    )
    
    if pnl is not None:
        pnl_emoji = "💵" if pnl >= 0 else "📉"
        msg += f"<b>PnL realized:</b> {pnl_emoji} <code>${pnl:+.2f}</code>\n"
        
    if order_id:
        msg += f"<b>Order ID:</b> <code>{order_id[-12:]}...</code>\n"
        
    msg += f"━━━━━━━━━━━━━━━━━━━━"
    await send_telegram_message(msg)


async def send_heartbeat_alert(equity: float, daily_pnl_pct: float, open_positions: int, buying_power: float):
    """Sends a summary heartbeat to Telegram."""
    pnl_sign = "+" if daily_pnl_pct >= 0 else ""
    status_emoji = "🟢 Running" if daily_pnl_pct >= -1.0 else "🟡 Soft Drawdown"
    
    msg = (
        f"📈 <b>Apex Oracle Bot - Heartbeat Status</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>System Status:</b> {status_emoji}\n"
        f"<b>Account Equity:</b> <code>${equity:,.2f}</code>\n"
        f"<b>Buying Power:</b> <code>${buying_power:,.2f}</code>\n"
        f"<b>Daily PnL:</b> <code>{pnl_sign}{daily_pnl_pct:.2f}%</code>\n"
        f"<b>Open Positions:</b> <code>{open_positions}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    await send_telegram_message(msg)


async def send_killswitch_alert(reason: str):
    """Sends an urgent killswitch warning to Telegram."""
    msg = (
        f"🚨 <b>URGENT: Apex Oracle Bot Killswitch Activated</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Breach Reason:</b> <code>{reason}</code>\n"
        f"<b>Action Taken:</b> Liquidated all active positions and halted trading loop.\n"
        f"⚠️ Please investigate the system status immediately.\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    await send_telegram_message(msg)


# ========================= BASE44 DASHBOARD CLIENT =========================

async def log_trade_base44(symbol: str, side: str, qty: float, price: float, realized_pnl: float = None, bot_name: str = 'apex_oracle_bot'):
    """Logs trade data to the Base44 dashboard API asynchronously."""
    if not DASHBOARD_API_KEY:
        logger.debug(f"[Base44 Bypass] Trade Log: {symbol} {side} {qty} @ {price}")
        return

    headers = {"api-key": DASHBOARD_API_KEY, "Content-Type": "application/json"}
    payload = {
        "bot_name": bot_name,
        "symbol": symbol,
        "side": side.upper(),
        "qty": float(qty),
        "entry_price": float(price),
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "status": "closed" if side.upper() == "SELL" else "open",
    }
    if realized_pnl is not None:
        payload["pnl"] = float(realized_pnl)

    url = f"{BASE44_API_URL}/Trade/"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=5) as response:
                if not response.ok:
                    logger.error(f"Base44 trade log failed: Status {response.status}")
    except Exception as e:
        logger.error(f"Base44 trade logging exception: {e}")


async def push_heartbeat_base44(equity: float, buying_power: float, daily_pnl_pct: float, open_positions_count: int, trades_today: int = 0, status: str = "running", bot_name: str = 'apex_oracle_bot'):
    """Updates/inserts BotStatus metrics to the Base44 dashboard API asynchronously."""
    if not DASHBOARD_API_KEY:
        logger.debug(f"[Base44 Bypass] Heartbeat: Equity=${equity:.2f}, positions={open_positions_count}")
        return

    headers = {"api-key": DASHBOARD_API_KEY, "Content-Type": "application/json"}
    payload = {
        "bot_name": bot_name,
        "status": status,
        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        "account_equity": float(equity),
        "buying_power": float(buying_power),
        "daily_pnl": float(daily_pnl_pct),
        "total_pnl": 0.0,
        "open_positions_count": int(open_positions_count),
        "trades_today": int(trades_today),
    }

    try:
        async with aiohttp.ClientSession() as session:
            # 1. Fetch to see if record exists
            url_get = f"{BASE44_API_URL}/BotStatus/"
            async with session.get(url_get, headers=headers, params={"bot_name": bot_name}, timeout=5) as get_resp:
                records = []
                if get_resp.status == 200:
                    records = await get_resp.json()
                
                if records and len(records) > 0:
                    # Update existing record
                    record_id = records[0]['id']
                    url_put = f"{BASE44_API_URL}/BotStatus/{record_id}/"
                    async with session.put(url_put, json=payload, headers=headers, timeout=5) as put_resp:
                        if not put_resp.ok:
                            logger.error(f"Base44 heartbeat update failed: Status {put_resp.status}")
                else:
                    # Create new record
                    url_post = f"{BASE44_API_URL}/BotStatus/"
                    async with session.post(url_post, json=payload, headers=headers, timeout=5) as post_resp:
                        if not post_resp.ok:
                            logger.error(f"Base44 heartbeat creation failed: Status {post_resp.status}")
    except Exception as e:
        logger.error(f"Base44 heartbeat exception: {e}")
