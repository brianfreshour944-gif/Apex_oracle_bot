"""Modern FastAPI server implementation."""

import asyncio
import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management."""
    logger.info("Starting FastAPI server")
    yield
    logger.info("Shutting down FastAPI server")

# Create FastAPI app
app = FastAPI(
    title="Apex Oracle Bot API",
    description="Modern trading bot API with FastAPI",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API router
api_router = APIRouter(prefix="/api/v1")

@api_router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "bot_name": settings.BOT_NAME,
        "version": "2.0.0",
        "database": str(settings.DATABASE_URL),
    }

@api_router.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get bot configuration."""
    return {
        "bot_name": settings.BOT_NAME,
        "symbols": settings.SYMBOLS,
        "risk_settings": {
            "base_risk_percent": settings.BASE_RISK_PERCENT,
            "max_single_trade_usd": settings.MAX_SINGLE_TRADE_USD,
        },
    }

# Include API router
app.include_router(api_router)

async def start_fastapi_server_async() -> None:
    """Start FastAPI server asynchronously."""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=settings.STATUS_PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)

    # Run server in background
    asyncio.create_task(server.serve())

    logger.info(f"FastAPI server started on port {settings.STATUS_PORT}")
    logger.info("API documentation available at /docs")
    logger.info("Health check available at /api/v1/health")