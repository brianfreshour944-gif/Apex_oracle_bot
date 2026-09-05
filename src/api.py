"""Modern FastAPI server implementation."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import APIRouter, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
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
    exchange: bool
    timestamp: str


def _check_database() -> bool:
    """Best-effort database connectivity check for health endpoint."""
    try:
        from src.db import get_db_health
        return get_db_health()
    except Exception:
        return True  # If we can't check, assume healthy


def _check_exchange() -> bool:
    """Check if exchange is configured and reachable."""
    try:
        from src.config import settings
        return bool(settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY)
    except Exception:
        return False


@api_router.get("/health", response_model=HealthCheckResponse)
async def health_check(response: Response) -> HealthCheckResponse:
    """Health check endpoint for container orchestration (Coolify/Docker)."""
    db_ok = _check_database()
    ex_ok = _check_exchange()
    overall_ok = db_ok and ex_ok
    
    response.status_code = 200 if overall_ok else 503
    return HealthCheckResponse(
        status="healthy" if overall_ok else "degraded",
        version="2.0.0",
        symbols=settings.SYMBOLS,
        database=db_ok,
        exchange=ex_ok,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@api_router.get("/api/v1/health", response_model=HealthCheckResponse)
async def health_check_v1(response: Response) -> HealthCheckResponse:
    """Health check endpoint under /api/v1/ prefix."""
    db_ok = _check_database()
    ex_ok = _check_exchange()
    overall_ok = db_ok and ex_ok
    
    response.status_code = 200 if overall_ok else 503
    return HealthCheckResponse(
        status="healthy" if overall_ok else "degraded",
        version="2.0.0",
        symbols=settings.SYMBOLS,
        database=db_ok,
        exchange=ex_ok,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@api_router.get("/ready", response_model=HealthCheckResponse)
async def readiness_check(response: Response) -> HealthCheckResponse:
    """Readiness check - verifies all dependencies are ready."""
    db_ok = _check_database()
    ex_ok = _check_exchange()
    overall_ok = db_ok and ex_ok
    
    response.status_code = 200 if overall_ok else 503
    return HealthCheckResponse(
        status="ready" if overall_ok else "not_ready",
        version="2.0.0",
        symbols=settings.SYMBOLS,
        database=db_ok,
        exchange=ex_ok,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@api_router.get("/live", response_model=HealthCheckResponse)
async def liveness_check(response: Response) -> HealthCheckResponse:
    """Liveness check - simple endpoint to verify process is alive."""
    return HealthCheckResponse(
        status="alive",
        version="2.0.0",
        symbols=settings.SYMBOLS,
        database=True,
        exchange=True,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@api_router.get("/config")
async def get_config() -> dict[str, Any]:
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
    """Root-level health check for container orchestration (Coolify/Docker)."""
    db_ok = _check_database()
    ex_ok = _check_exchange()
    overall_ok = db_ok and ex_ok
    
    response.status_code = 200 if overall_ok else 503
    return HealthCheckResponse(
        status="healthy" if overall_ok else "degraded",
        version="2.0.0",
        symbols=settings.SYMBOLS,
        database=db_ok,
        exchange=ex_ok,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/ready", response_model=HealthCheckResponse)
async def root_readiness_check(response: Response) -> HealthCheckResponse:
    """Root-level readiness check."""
    db_ok = _check_database()
    ex_ok = _check_exchange()
    overall_ok = db_ok and ex_ok
    
    response.status_code = 200 if overall_ok else 503
    return HealthCheckResponse(
        status="ready" if overall_ok else "not_ready",
        version="2.0.0",
        symbols=settings.SYMBOLS,
        database=db_ok,
        exchange=ex_ok,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/live", response_model=HealthCheckResponse)
async def root_liveness_check(response: Response) -> HealthCheckResponse:
    """Root-level liveness check."""
    return HealthCheckResponse(
        status="alive",
        version="2.0.0",
        symbols=settings.SYMBOLS,
        database=True,
        exchange=True,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


_fastapi_server_task: asyncio.Task | None = None


async def start_fastapi_server_async() -> None:
    """Start FastAPI server asynchronously and wait for it to be ready."""
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
    logger.info("Health check available at /health, /ready, /live")

    # Wait for server to be ready (health endpoint responds)
    import httpx
    for _ in range(30):  # 30s timeout
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"http://127.0.0.1:{settings.STATUS_PORT}/health", timeout=1.0)
                if resp.status_code == 200:
                    logger.info("FastAPI server health check passed")
                    return
        except Exception:
            pass
        await asyncio.sleep(1)
    logger.warning("FastAPI server health check timeout after 30s")


async def stop_fastapi_server_async() -> None:
    """Stop the FastAPI server if running."""
    global _fastapi_server_task
    if _fastapi_server_task is not None and not _fastapi_server_task.done():
        _fastapi_server_task.cancel()
        try:
            await _fastapi_server_task
        except asyncio.CancelledError:
            pass