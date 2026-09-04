"""Telegram alert notifications."""

import httpx

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


async def send_telegram_alert(message: str) -> bool:
    """Sends an HTML message to the configured Telegram chat."""
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        logger.debug("Telegram alert skipped (credentials not configured)")
        return False

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=10.0)
            if response.status_code != 200:
                logger.error(f"Telegram alert failed: {response.text}")
                return False
            return True
    except Exception as e:
        logger.error(f"Telegram alert exception: {e}")
        return False
