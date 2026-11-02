"""
Main FastAPI Application Entrypoint.
Blockchain-Based Byzantine-Resilient Federated Learning for Smart Grid Energy Theft Detection.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import time
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import router
from backend.config import get_settings
from backend.database.adapter import get_database_adapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for application startup and shutdown lifecycle."""
    settings = get_settings()
    logger.info("================================================================================")
    logger.info(f"Starting {settings.project_name}")
    logger.info(f"Environment : {settings.environment.upper()}")
    logger.info(f"CORS Origins: {settings.cors_origins}")
    logger.info("================================================================================")

    # 1. Connect & bootstrap database
    try:
        db = get_database_adapter()
        provider = db.get_provider_name()
        logger.info(f"Database provider '{provider}' operational and bootstrapped.")
    except Exception as exc:
        logger.error(f"Failed to initialize database during startup: {exc}")

    # 2. Check blockchain adapter
    try:
        from blockchain.service import get_blockchain_adapter
        bc = get_blockchain_adapter()
        connected = bc.is_connected()
        logger.info(f"Blockchain adapter operational (connected={connected}, provider={settings.blockchain_provider_type}).")
    except Exception as exc:
        logger.warning(f"Blockchain adapter check notice: {exc}")

    logger.info("Application startup complete. Ready to receive requests.")
    yield

    # Clean shutdown
    logger.info("Initiating application shutdown...")
    try:
        from backend.database.adapter import reset_database_adapter
        reset_database_adapter()
        logger.info("Database resources released successfully.")
    except Exception as exc:
        logger.warning(f"Error during database shutdown: {exc}")
    logger.info("Application shutdown finished.")


settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    version="1.0.0",
    description=(
        "Production-Grade FastAPI Backend for Byzantine-Resilient Federated Learning in Smart Grids.\n\n"
        "### Key Architecture & Operational Invariants:\n"
        "- **Privacy-Preserving Telemetry**: Raw individual smart meter time-series readings and consumer PII are "
        "STRICTLY REDACTED in all global responses. Only statistical aggregates and fraud ratios are served.\n"
        "- **Dual Database Architecture**: Automatically connects to MongoDB when available; seamlessly falls back to a "
        "fully documented, thread-safe in-memory adapter for rapid local development.\n"
        "- **Byzantine Resilience**: Coordinates multi-criteria trust evaluation (cosine similarity, server validation delta, "
        "reputation EMA) to isolate poisoning attacks without penalizing non-IID regional variations.\n"
        "- **Verifiable Blockchain Auditing**: On-chain audit trail records from the Ethereum ReputationRegistry smart contract "
        "with canonical SHA-256 integrity verification (`[TAMPER-EVIDENT OK]`).\n\n"
        "⚠️ **PRODUCTION SECURITY WARNING**:\n"
        "All simulation and federated learning control endpoints (`POST /api/fl/start`, `POST /api/simulation/run`) "
        "require mutual TLS, OAuth2/JWT token authentication, and strict Role-Based Access Control (RBAC) in production. "
        "In this local development environment, controls are unauthenticated for prototyping purposes."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ------------------------------------------------------------------------------
# 1. Configured Local-Origin CORS Middleware
# ------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# 2. Security Headers & Production Warning Middleware
# ------------------------------------------------------------------------------
@app.middleware("http")
async def security_and_warning_headers_middleware(request: Request, call_next):
    """
    Inject mandatory security headers and production warning headers on all HTTP responses.
    """
    response = await call_next(request)
    response.headers["X-Production-Warning"] = (
        "Authentication required for production deployments. Currently running in unauthenticated development mode."
    )
    response.headers["X-Privacy-Enforcement"] = "Strict - Raw Meter Readings Redacted"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


# ------------------------------------------------------------------------------
# 3. Global Exception Handlers
# ------------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Clean JSON response for invalid experiment parameters."""
    logger.warning(f"Request validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "One or more request parameters failed safety constraints.",
            "details": exc.errors(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all unhandled exception handler."""
    logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ------------------------------------------------------------------------------
# 4. Root Landing Endpoint
# ------------------------------------------------------------------------------
@app.get("/", tags=["Root"])
def root_index():
    """System welcome and API documentation index."""
    return {
        "project": settings.project_name,
        "environment": settings.environment,
        "status": "ONLINE",
        "documentation": "/docs",
        "health_check": "/health",
        "privacy_policy": "Raw meter readings are strictly redacted from all public and global responses.",
        "production_warning": "Control endpoints require authentication for production deployments.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Mount the API routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=False,
    )
