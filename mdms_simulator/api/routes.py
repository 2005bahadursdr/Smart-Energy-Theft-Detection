"""
FastAPI Routes for MDMS Simulator API.
Exposes telemetry ingestion, meter status lookups, client partition summaries,
and data quality audits.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from mdms_simulator.config import LEGAL_DISCLAIMER
from mdms_simulator.models import (
    BatchIngestRequest,
    ClientPartitionSummary,
    DataQualityReport,
    IngestSummary,
    MeterReadingPayload,
    MeterStatusResponse,
)
from mdms_simulator.partitioner import RawDataPrivacyViolationError
from mdms_simulator.service import MDMSService

router = APIRouter(prefix="/api/v1", tags=["MDMS Simulator"])


def get_mdms_service() -> MDMSService:
    """Dependency provider for MDMSService singleton."""
    from mdms_simulator.api.app import mdms_service_instance
    return mdms_service_instance


# ==============================================================================
# Ingestion Endpoints
# ==============================================================================

@router.post(
    "/ingest/reading",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Ingest single smart meter reading"
)
async def ingest_single_reading(
    payload: MeterReadingPayload,
    service: MDMSService = Depends(get_mdms_service)
) -> Dict[str, Any]:
    """
    Ingest a single pseudonymous smart meter reading.
    Validates schema, timestamps, client assignment, and range bounds.
    """
    success, message = service.ingest_reading(payload)
    if not success:
        return {
            "status": "rejected",
            "accepted": False,
            "message": message,
            "meter_id": payload.meter_id,
            "timestamp": payload.timestamp
        }

    return {
        "status": "accepted",
        "accepted": True,
        "message": message,
        "meter_id": payload.meter_id,
        "timestamp": payload.timestamp,
        "client_id": payload.client_id
    }


@router.post(
    "/ingest/batch",
    response_model=IngestSummary,
    status_code=status.HTTP_200_OK,
    summary="Batch ingest smart meter readings"
)
async def ingest_batch_readings(
    request: BatchIngestRequest,
    service: MDMSService = Depends(get_mdms_service)
) -> IngestSummary:
    """
    Ingest an array of smart meter readings in a high-performance batch transaction.
    """
    return service.ingest_batch(request.readings)


@router.post(
    "/ingest/file",
    response_model=IngestSummary,
    status_code=status.HTTP_200_OK,
    summary="Ingest readings directly from simulated CSV file"
)
async def ingest_csv_file(
    file_path: str = Query(
        default="data/synthetic/simulated_meter_data.csv",
        description="Path to synthetic meter CSV dataset"
    ),
    max_rows: Optional[int] = Query(
        default=None,
        description="Optional limit on rows to ingest"
    ),
    service: MDMSService = Depends(get_mdms_service)
) -> IngestSummary:
    """
    Ingest smart meter readings exported from the smart-meter simulator CSV.
    Partitions readings directly into local substation client vaults.
    """
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulator CSV file not found at: {file_path}"
        )

    try:
        summary = service.ingest_csv_file(path, max_rows=max_rows)
        return summary
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest CSV: {e}"
        )


@router.post(
    "/ingest/sgcc",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Ingest and partition the benchmark SGCC dataset"
)
async def ingest_sgcc_dataset(
    num_meters: int = Query(
        default=1000,
        ge=100,
        le=42372,
        description="Number of smart meters to sample from SGCC (100 to 42,372)"
    ),
    raw_path: Optional[str] = Query(
        default=None,
        description="Optional custom path to SGCC raw CSV"
    ),
    service: MDMSService = Depends(get_mdms_service)
) -> Dict[str, Any]:
    """
    Ingests the real-world benchmark SGCC smart meter dataset, cleans missing readings,
    clusters meters into non-IID regional substations, and extracts sliding sequences (30 days).
    """
    try:
        p_path = Path(raw_path) if raw_path else None
        summary = service.ingest_sgcc_dataset(raw_csv_path=p_path, num_meters=num_meters)
        return summary
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process SGCC dataset: {e}"
        )


# ==============================================================================
# Meter Status & Telemetry Queries
# ==============================================================================

@router.get(
    "/meters/{meter_id}/status",
    response_model=MeterStatusResponse,
    summary="Query operational meter status and telemetry health"
)
async def get_meter_status(
    meter_id: str,
    service: MDMSService = Depends(get_mdms_service)
) -> MeterStatusResponse:
    """
    Get current status, latest reading, hardware tamper flags,
    and data quality score for a specific meter.
    """
    status_resp = service.get_meter_status(meter_id)
    if not status_resp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meter '{meter_id}' has not recorded any readings in MDMS."
        )
    return status_resp


@router.get(
    "/meters",
    response_model=List[Dict[str, Any]],
    summary="List registered meters"
)
async def list_meters(
    client_id: Optional[str] = Query(None, description="Filter by substation client ID"),
    meter_status: Optional[str] = Query(None, description="Filter by meter status (e.g. NORMAL, OUTAGE)"),
    limit: int = Query(100, ge=1, le=1000, description="Max meters to return"),
    service: MDMSService = Depends(get_mdms_service)
) -> List[Dict[str, Any]]:
    """List meters recorded in the MDMS with filtering."""
    return service.list_meters(client_id=client_id, status=meter_status, limit=limit)


@router.get(
    "/meters/{meter_id}/history",
    response_model=List[Dict[str, Any]],
    summary="Retrieve recent telemetry history for a meter"
)
async def get_meter_history(
    meter_id: str,
    limit: int = Query(168, ge=1, le=720, description="Max historical hourly readings"),
    service: MDMSService = Depends(get_mdms_service)
) -> List[Dict[str, Any]]:
    """Retrieve chronologically ordered recent telemetry readings for a meter."""
    return service.storage.get_meter_history(meter_id, limit=limit)


# ==============================================================================
# Client Partitions & Privacy Guarantees
# ==============================================================================

@router.get(
    "/clients",
    response_model=Dict[str, Any],
    summary="List configured client partitions and local vault summaries"
)
async def list_clients(
    service: MDMSService = Depends(get_mdms_service)
) -> Dict[str, Any]:
    """
    List all configured substation client partitions, local storage paths,
    and privacy status.
    """
    return service.get_all_client_summaries()


@router.get(
    "/clients/{client_id}/summary",
    response_model=Dict[str, Any],
    summary="Get single client partition metadata"
)
async def get_client_summary(
    client_id: str,
    service: MDMSService = Depends(get_mdms_service)
) -> Dict[str, Any]:
    """Retrieve metadata and local vault status for a specific client."""
    summary = service.get_client_summary(client_id)
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client '{client_id}' is not in configured partitions."
        )
    return summary


@router.get(
    "/clients/{client_id}/verify-privacy",
    response_model=Dict[str, Any],
    summary="Verify the FL Server raw reading privacy boundary"
)
async def verify_privacy_boundary(
    client_id: str,
    service: MDMSService = Depends(get_mdms_service)
) -> Dict[str, Any]:
    """
    Demonstrates and validates that the Central FL Server is blocked
    from accessing raw customer telemetry.
    """
    vault = service.partitioner.get_vault(client_id)
    if not vault:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client '{client_id}' not found."
        )

    # Attempt to access raw data as FL_SERVER (must be blocked)
    privacy_blocked = False
    error_msg = ""
    try:
        vault.access_raw_readings(requesting_entity="FL_SERVER")
    except RawDataPrivacyViolationError as ex:
        privacy_blocked = True
        error_msg = str(ex)

    return {
        "client_id": client_id,
        "test_requesting_entity": "FL_SERVER",
        "access_granted": not privacy_blocked,
        "privacy_firewall_status": "BLOCKED" if privacy_blocked else "VULNERABLE",
        "security_policy": "The central FL server must NEVER receive raw meter readings.",
        "diagnostic_message": error_msg
    }


# ==============================================================================
# Data Quality & System Health
# ==============================================================================

@router.get(
    "/data-quality",
    response_model=DataQualityReport,
    summary="Get comprehensive data-quality statistics"
)
async def get_data_quality_report(
    service: MDMSService = Depends(get_mdms_service)
) -> DataQualityReport:
    """
    Audit report detailing total received, accepted, duplicate,
    missing reading, and late event counts.
    """
    return service.get_data_quality_report()


@router.get(
    "/health",
    response_model=Dict[str, Any],
    summary="Service health and legal disclaimer"
)
async def health_check(
    service: MDMSService = Depends(get_mdms_service)
) -> Dict[str, Any]:
    """Health check endpoint containing system status and legal disclaimer."""
    return {
        "status": "operational",
        "service": "MDMS Simulator",
        "version": service.config.version,
        "database_path": str(service.config.db_path),
        "total_stored_readings": service.storage.get_total_readings_count(),
        "disclaimer": LEGAL_DISCLAIMER
    }
