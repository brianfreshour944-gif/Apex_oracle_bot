  
"""Critical alerting via Telegram with email fallback."""  
  
from __future__ import annotations  
  
import asyncio  
import smtplib  
import time  
from email.message import EmailMessage  
  
import httpx  
  
from src.config import settings  
from src.logging_config import get_logger  
  
logger = get_logger("alerts")  
  
_last_sent = {}  
  
def _in_cooldown(key):  
    now = time.monotonic()  
    last = _last_sent.get(key, 0.0)  
    if now - last < settings.ALERT_COOLDOWN_SEC:  
        return True  
    _last_sent[key] = now  
    return False  
  
async def send_alert(message, key="default"):  
    if _in_cooldown(key):  
        logger.debug(f"Alert suppressed (cooldown): {message}")  
        return  
    logger.critical(f"ALERT: {message}")  
    await _send_telegram(message)  
    if not settings.TELEGRAM_BOT_TOKEN:  
        await _send_email(message)  
  
async def _send_telegram(message):  
    if not (settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID):  
        return  
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"  
    try:  
        async with httpx.AsyncClient(timeout=10.0) as client:  
            await client.post(url, json={"chat_id": settings.TELEGRAM_CHAT_ID, "text": message})  
    except Exception as e:  
        logger.error(f"Telegram alert failed: {e}")  
  
async def _send_email(message):  
    if not settings.ALERT_EMAIL:  
        return  
    try:  
        msg = EmailMessage()  
        msg["Subject"] = f"[{settings.BOT_NAME}] CRITICAL ALERT"  
        msg["To"] = settings.ALERT_EMAIL  
        msg.set_content(message)  
        with smtplib.SMTP("localhost", 25, timeout=10) as smtp:  
            smtp.send_message(msg)  
    except Exception as e:  
        logger.error(f"Email alert failed: {e}")  
