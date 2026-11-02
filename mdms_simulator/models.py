"""
Pydantic Schemas and Data Models for MDMS Simulator.
Defines telemetry payloads, validation contracts, ingestion responses,
meter status summaries, and data-quality reporting structures.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class MeterReadingPayload(BaseModel):
    """
    Standard smart meter telemetry ingestion payload.
    Conforms to pseudonymous privacy standards without PII.
    """
    timestamp: str = Field(
        ...,
        description="ISO-8601 formatted timestamp string (e.g. '2026-01-01 00:00:00')"
    )
    meter_id: str = Field(
        ...,
        description="Pseudonymous meter identifier (e.g. 'MTR-A-0001')",
        pattern=r"^MTR-[A-Z0-9]+-[0-9A-Fa-f]+$"
    )
    client_id: str = Field(
        ...,
        description="Substation feeder identifier (e.g. 'client_A')",
        pattern=r"^client_[A-Za-z0-9]+$"
    )
    consumption_kwh: Optional[float] = Field(
        default=None,
        description="Energy reading in kWh (None/NaN represents communication packet loss)"
    )
    meter_status: str = Field(
        default="NORMAL",
        description="Firmware operational state: NORMAL, OUTAGE, DISCONNECTED, COMM_FAILURE, TAMPER_ALERT"
    )
    event_type: str = Field(
        default="NONE",
        description="Grid event: NONE, GRID_OUTAGE, DISCONNECTION, VACANCY, SOLAR_SELF_GEN, SUSTAINED_DROP, COMM_FAILURE, TAMPER_EVENT"
    )
    hardware_tamper_flag: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Binary hardware sensor alert (0 = normal, 1 = alarm)"
    )

    # Simulated Ground-Truth Fields (Research prototype validation only; not used for utility logic)
    simulated_theft_label: Optional[int] = Field(
        default=0,
        description="Synthetic ground-truth label (0 = legitimate, 1 = simulated theft)"
    )
    simulated_theft_type: Optional[str] = Field(
        default="NONE",
        description="Simulated theft modality name"
    )
    simulated_actual_kwh: Optional[float] = Field(
        default=None,
        description="Simulated un-tampered physical demand for research error evaluation"
    )
    simulated_unreported_kwh: Optional[float] = Field(
        default=0.0,
        description="Simulated unregistered energy amount (actual - reported)"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_format(cls, v: str) -> str:
        """Validate that timestamp can be parsed as a valid datetime."""
        try:
            # Supports both space and T separators
            cleaned = v.replace("T", " ")
            datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")
            return cleaned
        except ValueError:
            raise ValueError(f"Invalid timestamp format: '{v}'. Expected ISO format 'YYYY-MM-DD HH:MM:SS'.")


class BatchIngestRequest(BaseModel):
    """Payload for batch telemetry ingestion."""
    readings: List[MeterReadingPayload] = Field(
        ...,
        description="List of meter readings for batch ingestion"
    )


class IngestSummary(BaseModel):
    """Summary response for an ingestion transaction."""
    status: str = Field(..., description="Overall ingestion outcome ('success', 'partial', 'rejected')")
    received: int = Field(..., description="Total readings submitted")
    accepted: int = Field(..., description="Readings successfully validated and stored")
    duplicates: int = Field(..., description="Duplicate readings detected and deduplicated")
    missing_readings: int = Field(..., description="Readings with null/NaN consumption flagged")
    late_arrivals: int = Field(..., description="Readings arriving after historical watermark")
    rejected: int = Field(..., description="Malformed or out-of-range readings rejected")
    errors: List[str] = Field(default_factory=list, description="Diagnostic error messages")


class MeterStatusResponse(BaseModel):
    """Current operational health and data telemetry summary for a single smart meter."""
    meter_id: str
    client_id: str
    current_status: str
    latest_timestamp: Optional[str] = None
    latest_consumption_kwh: Optional[float] = None
    latest_event_type: Optional[str] = None
    hardware_tamper_active: bool = False
    total_readings_recorded: int = 0
    missing_readings_count: int = 0
    data_quality_score_pct: float = 100.0


class ClientPartitionSummary(BaseModel):
    """Metadata summary of a client's local data vault."""
    client_id: str
    meter_count: int
    readings_count: int
    local_storage_path: str
    privacy_guarantee: str = (
        "STRICT LOCAL PRIVACY: Raw meter readings are stored exclusively in the client's "
        "local directory. The central FL server has zero access to raw telemetry."
    )


class DataQualityReport(BaseModel):
    """Fleet-wide data-quality audit report."""
    disclaimer: str
    generated_at: str
    total_received: int
    total_accepted: int
    total_duplicates: int
    total_missing_readings: int
    total_late_arrivals: int
    total_rejected: int
    overall_completion_rate_pct: float
    client_breakdown: Dict[str, Dict[str, Any]]
