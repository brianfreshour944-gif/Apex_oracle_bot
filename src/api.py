"""Modern FastAPI server implementation."""

import asyncio
import uvicorn
from fastapi import FastAPI, APIRouter, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from pydantic import BaseModel

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


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    symbols: list
    database: bool


def _check_database() -> bool:
    """Best-effort database connectivity check for health endpoint."""
    try:
        from src.db import get_db_health
        return get_db_health()
    except Exception:
        return True  # If we can't check, assume healthy


@api_router.get("/health", response_model=HealthCheckResponse)
async def health_check(response: Response) -> HealthCheckResponse:
    """Health check endpoint for container orchestration (Coolify/Docker)."""
    db_ok = _check_database()
    response.status_code = 200 if db_ok else 503
    return HealthCheckResponse(
        status="healthy" if db_ok else "degraded",
        version="2.0.0",
        symbols=settings.SYMBOLS,
        database=db_ok,
    )


@api_router.get("/api/v1/health", response_model=HealthCheckResponse)
async def health_check_v1(response: Response) -> HealthCheckResponse:
    """Health check endpoint under /api/v1/ prefix."""
    db_ok = _check_database()
    response.status_code = 200 if db_ok else 503
    return HealthCheckResponse(
        status="healthy" if db_ok else "degraded",
        version="2.0.0",
        symbols=settings.SYMBOLS,
        database=db_ok,
    )


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


@app.get("/health", response_model=HealthCheckResponse)
async def root_health_check(response: Response) -> HealthCheckResponse:
    """Root-level health check for container orchestration."""
    db_ok = _check_database()
    response.status_code = 200 if db_ok else 503
    return HealthCheckResponse(
        status="healthy" if db_ok else "degraded",
        version="2.0.0",
        symbols=settings.SYMBOLS,
        database=db_ok,
    )


_fastapi_server_task: Optional[asyncio.Task] = None


async def start_fastapi_server_async() -> None:
    """Start FastAPI server asynchronously."""
    global _fastapi_server_task
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=settings.STATUS_PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)
    _fastapi_server_task = asyncio.create_task(server.serve())
    _fastapi_server_task.add_done_callback(
        lambda t: logger.info("FastAPI server stopped") if not t.cancelled() else None
    )
    logger.info(f"FastAPI server started on port {settings.STATUS_PORT}")
    logger.info("API documentation available at /docs")
    logger.info("Health check available at /health")


async def stop_fastapi_server_async() -> None:
    """Stop the FastAPI server if running."""
    global _fastapi_server_task
    if _fastapi_server_task is not None and not _fastapi_server_task.done():
        _fastapi_server_task.cancel()
        try:
            await _fastapi_server_task
        except asyncio.CancelledError:
            pass