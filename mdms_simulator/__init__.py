"""
mdms_simulator module.
Blockchain-Based Byzantine-Resilient Federated Learning for Smart Grid Energy Theft Detection.

Provides an MDMS-like ingestion, validation, and privacy-preserving data partitioning service.
DISCLAIMER: Research prototype simulator only. Not certified for utility billing or legal metrology.
"""

from mdms_simulator.config import LEGAL_DISCLAIMER, MDMSConfig
from mdms_simulator.models import (
    BatchIngestRequest,
    ClientPartitionSummary,
    DataQualityReport,
    IngestSummary,
    MeterReadingPayload,
    MeterStatusResponse,
)
from mdms_simulator.partitioner import (
    ClientDataVault,
    LocalPartitionManager,
    RawDataPrivacyViolationError,
)
from mdms_simulator.quality import DataQualityTracker
from mdms_simulator.service import MDMSService
from mdms_simulator.storage import MDMSStorage
from mdms_simulator.validator import TelemetryValidator, ValidationResult

__all__ = [
    "LEGAL_DISCLAIMER",
    "MDMSConfig",
    "MeterReadingPayload",
    "BatchIngestRequest",
    "IngestSummary",
    "MeterStatusResponse",
    "ClientPartitionSummary",
    "DataQualityReport",
    "TelemetryValidator",
    "ValidationResult",
    "DataQualityTracker",
    "MDMSStorage",
    "ClientDataVault",
    "LocalPartitionManager",
    "RawDataPrivacyViolationError",
    "MDMSService",
]
