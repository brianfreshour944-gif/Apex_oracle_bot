"""Main trading bot implementation."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from src.config import settings
from src.logging_config import get_logger
from src.db import init_db
from src.exchange import AlpacaExchange
from src.api import start_fastapi_server_async
from src.strategies import TradingStrategy
from src.risk import RiskManager

logger = get_logger("bot")

# Global state
ex: Optional[AlpacaExchange] = None
strategy: Optional[TradingStrategy] = None
risk_manager: Optional[RiskManager] = None

async def process_signal_for_symbol(symbol: str, current_price: float, risk_manager: RiskManager, strategy: TradingStrategy, ex: AlpacaExchange) -> None:
    """Processes signal for a single symbol asynchronously."""
    try:
        # Get current positions
        positions = await ex.get_positions()
        position_dict = {p["symbol"].replace("/", ""): p for p in positions}
        current_position = position_dict.get(symbol.replace("/", ""))

        # Generate trading signal
        signal = await strategy.generate_trading_signal(
            symbol,
            current_price,
            current_position
        )

        logger.info(f"Signal for {symbol} @ ${current_price:.2f}: {signal['action']} (regime: {signal['regime']}, RSI: {signal['rsi']:.2f})")

        # Execute signal if not "hold" or "stand_aside"
        if signal["action"] in ["buy", "sell"]:
            # Check position limit before entering
            risk_status = await risk_manager.update_account_status()
            if risk_status["status"] == "position_limit_exceeded":
                logger.warning(f"Skipping entry for {symbol}: position limit reached")
                return
            if risk_status["status"] == "exposure_limit_exceeded":
                logger.warning(f"Skipping entry for {symbol}: exposure cap reached")
                return

            # Calculate position size
            position_size, sizing_status = risk_manager.calculate_position_size(
                symbol,
                current_price,
                signal["regime"]
            )

            if sizing_status != "ok":
                logger.warning(f"Position sizing failed for {symbol}: {sizing_status}")
                return

            # Place order
            order_result = await ex.create_order(
                symbol=symbol,
                qty=position_size,
                side=signal["action"],
                type="market"
            )

            logger.info(f"Order executed: {signal['action']} {position_size} {symbol} @ ${current_price:.2f}")
            logger.debug(f"Order result: {order_result}")

        elif signal["action"] == "close" and current_position:
            # Close existing position
            qty = float(current_position["qty"])
            side = "sell" if qty > 0 else "buy"
            qty_abs = abs(qty)

            order_result = await ex.create_order(
                symbol=symbol,
                qty=qty_abs,
                side=side,
                type="market"
            )

            logger.info(f"Position closed: {symbol} (was {qty}) - reason: {signal.get('reason', 'unknown')}")
            logger.debug(f"Close order result: {order_result}")

    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")


async def monitor_killswitch(risk_manager: RiskManager) -> None:
    """Background task to monitor killswitch continuously."""
    import os
    while True:
        try:
            if await risk_manager.check_killswitch_conditions():
                logger.critical("KILLSWITCH ACTIVATED - Liquidating all positions")
                await risk_manager.liquidate_all_positions()
                os._exit(1)
            await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"Killswitch monitor error: {e}")
            await asyncio.sleep(10)


async def run_trading_bot() -> None:
    """Main trading bot loop."""
    global ex

    try:
        logger.info("Initializing Apex Oracle Bot v2.0.0")
        logger.info("=================================")
        logger.info("Configuration:")
        logger.info(f"Bot Name: {settings.BOT_NAME}")
        logger.info(f"Database: {settings.DATABASE_URL}")
        logger.info(f"Exchange: Alpaca Crypto (Paper={settings.ALPACA_BASE_URL.endswith('paper-api.alpaca.markets')})")
        logger.info(f"Symbols: {settings.SYMBOLS}")
        logger.info("=================================")

        # Initialize database (handle connection failures gracefully)
        try:
            init_db()
            logger.info("Database connected successfully")
        except Exception as e:
            logger.warning(f"Database connection failed (will retry later): {e}")
            logger.info("Running in offline mode - some features may be limited")

        logger.info(settings.log_config())

        # Initialize exchange
        ex = AlpacaExchange()
        try:
            await ex.load()
            logger.info("Alpaca exchange connected")
        except RuntimeError:
            # Re-raise configuration and auth errors which are fatal
            raise
        except Exception as e:
            logger.warning(f"Alpaca exchange connection failed on startup: {e}. Bot will start in offline/retry mode.")

        # Initialize trading strategy and risk manager
        strategy = TradingStrategy(ex)
        risk_manager = RiskManager(ex)
        logger.info("Trading strategy and risk manager initialized")

        # Start API server
        await start_fastapi_server_async()
        logger.info("FastAPI server started")

        # Start Killswitch monitor
        asyncio.create_task(monitor_killswitch(risk_manager))
        logger.info("Killswitch monitor started")

        # Main trading loop
        logger.info("Bot initialization complete. Starting event-driven trading loop.")

        last_eval_time = {s: 0.0 for s in settings.SYMBOLS}

        try:
            async for bar_msg in ex.listen_crypto_bars(settings.SYMBOLS):
                symbol = bar_msg.get("S")
                current_price = float(bar_msg.get("c"))
                
                # Throttle evaluation to LOOP_INTERVAL_SEC to avoid REST rate limits on get_bars/get_positions
                now = time.time()
                if now - last_eval_time.get(symbol, 0) < settings.LOOP_INTERVAL_SEC:
                    continue
                last_eval_time[symbol] = now

                # Dispatch signal processing to a background task so it doesn't block the WebSocket stream
                asyncio.create_task(process_signal_for_symbol(symbol, current_price, risk_manager, strategy, ex))

        except Exception as e:
            logger.error(f"WebSocket stream error: {e}")
            await asyncio.sleep(5)  # Wait before potential restart logic (handled externally)

    except KeyboardInterrupt:
        logger.info("Shutdown requested. Exiting gracefully.")
        if ex:
            await ex.close()
    except Exception as e:
        logger.error(f"Fatal error in bot: {e}", exc_info=True)
        if ex:
            await ex.close()
        raise

def run_bot() -> None:
    """Synchronous wrapper for async bot."""
    asyncio.run(run_trading_bot())