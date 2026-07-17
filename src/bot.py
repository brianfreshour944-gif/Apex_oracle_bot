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
        await ex.load()
        logger.info("Alpaca exchange connected")

        # Initialize trading strategy and risk manager
        strategy = TradingStrategy(ex)
        risk_manager = RiskManager(ex)
        logger.info("Trading strategy and risk manager initialized")

        # Start API server
        await start_fastapi_server_async()
        logger.info("FastAPI server started")

        # Main trading loop
        logger.info("Bot initialization complete. Starting trading loop.")

        while True:
            try:
                # Check killswitch conditions first
                if await risk_manager.check_killswitch_conditions():
                    logger.critical("KILLSWITCH ACTIVATED - Liquidating all positions")
                    await risk_manager.liquidate_all_positions()
                    break

                # Get current positions
                positions = await ex.get_positions()
                position_dict = {p["symbol"]: p for p in positions}

                # Analyze each symbol and generate signals
                for symbol in settings.SYMBOLS:
                    try:
                        # Get current price
                        bars = await ex.get_bars(symbol, "1Min", 1)
                        if len(bars) == 0:
                            logger.warning(f"No price data for {symbol}")
                            continue

                        current_price = float(bars["close"][0])

                        # Get current position for this symbol
                        current_position = position_dict.get(symbol)

                        # Generate trading signal
                        signal = await strategy.generate_trading_signal(
                            symbol,
                            current_price,
                            current_position
                        )

                        logger.info(f"Signal for {symbol}: {signal['action']} (regime: {signal['regime']}, RSI: {signal['rsi']:.2f})")

                        # Execute signal if not "hold" or "stand_aside"
                        if signal["action"] in ["buy", "sell"]:
                            # Check position limit before entering
                            risk_status = await risk_manager.update_account_status()
                            if risk_status["status"] == "position_limit_exceeded":
                                logger.warning(f"Skipping entry for {symbol}: position limit reached")
                                continue
                            if risk_status["status"] == "exposure_limit_exceeded":
                                logger.warning(f"Skipping entry for {symbol}: exposure cap reached")
                                continue

                            # Calculate position size
                            position_size, sizing_status = risk_manager.calculate_position_size(
                                symbol,
                                current_price,
                                signal["regime"]
                            )

                            if sizing_status != "ok":
                                logger.warning(f"Position sizing failed for {symbol}: {sizing_status}")
                                continue

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

                # Check risk status
                risk_status = await risk_manager.update_account_status()
                if risk_status["status"] != "risk_ok":
                    logger.warning(f"Risk status: {risk_status['status']}")

                # Sleep until next cycle
                await asyncio.sleep(settings.LOOP_INTERVAL_SEC)

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

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