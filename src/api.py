"""Modern FastAPI API server replacing the raw asyncio status server."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

from src.config import settings
from src.db import (
    query_recent_trades,
    SessionLocal,
    BotStatus,
)
from sqlalchemy import select

logger = logging.getLogger("api")

# FastAPI app setup
app = FastAPI(
    title="Apex Oracle Bot API",
    description="Modern FastAPI API for trading bot status and monitoring",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TradeResponse(BaseModel):
    """Pydantic model for trade response."""
    symbol: str
    side: str
    price: float
    quantity: float
    value: float
    timestamp: datetime

class StatusResponse(BaseModel):
    """Pydantic model for status response."""
    status: str
    time: datetime
    equity: float
    starting_equity: float
    daily_starting_equity: float
    buying_power: float
    daily_pnl_pct: float
    open_positions: int
    recent_trades: List[TradeResponse]

@app.get(
    "/status",
    response_model=StatusResponse,
    summary="Get bot status",
    description="Returns current bot status, equity, and recent trades",
    response_description="Bot status information",
)
async def get_status() -> StatusResponse:
    """FastAPI endpoint for bot status."""
    try:
        session = SessionLocal()
        try:
            # Get bot status
            stmt = select(BotStatus).where(BotStatus.bot_name == settings.BOT_NAME)
            status_rec = session.execute(stmt).scalars().first()

            # Get recent trades
            recent_trades = query_recent_trades(bot_name=settings.BOT_NAME, limit=5)

            # Build response
            response_data = {
                "status": status_rec.status if status_rec else "unknown",
                "time": datetime.now(timezone.utc),
                "equity": status_rec.live_equity if status_rec else 0.0,
                "starting_equity": status_rec.starting_equity if status_rec else 0.0,
                "daily_starting_equity": status_rec.daily_starting_equity if status_rec else 0.0,
                "buying_power": status_rec.buying_power if status_rec else 0.0,
                "daily_pnl_pct": status_rec.daily_pnl_pct if status_rec else 0.0,
                "open_positions": status_rec.open_positions_count if status_rec else 0,
                "recent_trades": [
                    {
                        "symbol": t["symbol"],
                        "side": t["side"],
                        "price": t["price"],
                        "quantity": t["quantity"],
                        "value": t["value"],
                        "timestamp": t["timestamp"],
                    }
                    for t in recent_trades
                ],
            }

            return response_data

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )

@app.get(
    "/health",
    summary="Health check",
    description="Simple health check endpoint",
    response_description="Health status",
)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bot_name": settings.BOT_NAME,
    }

def start_fastapi_server() -> None:
    """Start the FastAPI server using uvicorn."""
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=settings.STATUS_PORT,
            log_level="info",
            access_log=True,
            workers=1,
        )
    except Exception as e:
        logger.error(f"FastAPI server failed to start: {e}")
        raise

async def start_fastapi_server_async() -> None:
    """Start FastAPI server asynchronously."""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=settings.STATUS_PORT,
        log_level="info",
        access_log=True,
        workers=1,
    )
    server = uvicorn.Server(config)

    # Run in background task
    asyncio.create_task(server.serve())

    logger.info(f"FastAPI server started on http://0.0.0.0:{settings.STATUS_PORT}")