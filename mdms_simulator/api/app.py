"""
FastAPI Application for MDMS Simulator.
Serves high-performance REST APIs for smart meter telemetry ingestion,
pseudonymous validation, local client data partitioning, and data-quality reporting.

DISCLAIMER:
This software is a research prototype simulator for Byzantine-resilient Federated
Learning. It is NOT a certified utility Meter Data Management System (MDMS).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mdms_simulator.api.routes import router
from mdms_simulator.config import LEGAL_DISCLAIMER, MDMSConfig
from mdms_simulator.service import MDMSService

# Module-level singleton instance for runtime and API dependencies
mdms_service_instance = MDMSService()


def create_app(config: MDMSConfig = None, service: MDMSService = None) -> FastAPI:
    """Factory creating and configuring the MDMS FastAPI application."""
    global mdms_service_instance
    if service is not None:
        mdms_service_instance = service
    elif config is not None:
        mdms_service_instance = MDMSService(config=config)

    app = FastAPI(
        title="MDMS Simulator API",
        version="1.0.0",
        description=(
            "### Smart Grid Meter Data Management System (MDMS) Simulator\n\n"
            "> **LEGAL & RESEARCH NOTICE:**\n"
            f"> {LEGAL_DISCLAIMER}\n\n"
            "This service simulates an AMI / MDMS utility ingestion pipeline. It receives telemetry "
            "from simulated smart meters, performs rigorous validation, deduplication, and quality tracking, "
            "and strictly partitions raw consumption data into isolated client substation vaults (`client_A` - `client_E`). "
            "Under Federated Learning privacy constraints, the central FL server never receives raw meter readings."
        ),
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS configuration for local React / dashboard frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
