"""Redis cross-process state manager for multi-bot synchronization."""

import json
from typing import Dict, Any, Optional
from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

try:
    import redis.asyncio as redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class RedisStateManager:
    """Manages shared regime state and cross-bot synchronization over Redis."""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
        self.client = None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to Redis server."""
        if not _REDIS_AVAILABLE:
            logger.debug("redis-py package not installed. Skipping Redis connection.")
            return False

        try:
            self.client = redis.from_url(self.redis_url, socket_timeout=3.0)
            await self.client.ping()
            self._connected = True
            logger.info(f"Connected to Redis state manager at {self.redis_url}")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed ({self.redis_url}): {e}. Operating in local fallback mode.")
            self._connected = False
            return False

    async def set_regime_flag(self, flag_data: Dict[str, Any]) -> bool:
        """Set shared regime flag in Redis."""
        if not self._connected or not self.client:
            return False
        try:
            await self.client.set("apex:regime_flag", json.dumps(flag_data))
            return True
        except Exception as e:
            logger.error(f"Failed to set regime flag in Redis: {e}")
            return False

    async def get_regime_flag(self) -> Dict[str, Any]:
        """Get shared regime flag from Redis with local default fallback."""
        default_flag = {
            "pause_grok": False,
            "pause_oracle": False,
            "grok_multiplier": 1.0,
            "oracle_multiplier": 1.0,
            "regime": "normal",
        }
        if not self._connected or not self.client:
            return default_flag

        try:
            val = await self.client.get("apex:regime_flag")
            if val:
                return json.loads(val)
            return default_flag
        except Exception as e:
            logger.error(f"Failed to read regime flag from Redis: {e}")
            return default_flag

    async def close(self) -> None:
        """Close Redis connection."""

        if self.client:
            try:
                await self.client.close()
            except Exception:
                pass
            self._connected = False
