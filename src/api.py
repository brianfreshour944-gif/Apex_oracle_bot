"""Modern FastAPI server implementation."""

import asyncio
import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management."""
    logger.info("Starting FastAPI server")
    yield
    logger.info("Shutting down FastAPI server")


app = FastAPI(
    title="Apex Oracle Bot API",
    description="Modern trading bot API with FastAPI",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api/v1")


@api_router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "database": str(settings.DATABASE_URL),
        "symbols": settings.SYMBOLS,
    }


@api_router.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get bot configuration."""
    return {
        "symbols": settings.SYMBOLS,
        "risk_settings": {
            "base_risk_percent": settings.BASE_RISK_PERCENT,
            "max_single_trade_usd": settings.MAX_SINGLE_TRADE_USD,
        },
    }


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
    asyncio.create_task(server.serve())

    logger.info(f"FastAPI server started on port {settings.STATUS_PORT}")
    logger.info("API documentation available at /docs")
    logger.info("Health check available at /api/v1/health")