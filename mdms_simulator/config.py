"""
MDMS Simulator Configuration and Operational Parameters.

DISCLAIMER:
This software is a research and prototyping simulator designed for
Blockchain-Based Byzantine-Resilient Federated Learning smart grid research.
It is NOT a certified utility Meter Data Management System (MDMS) and is
not intended for legal metrology, revenue billing, or grid dispatch.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


LEGAL_DISCLAIMER = (
    "RESEARCH SIMULATOR ONLY: This software simulates Meter Data Management System (MDMS) "
    "ingestion, validation, and client-side data isolation for Federated Learning research. "
    "It is NOT a certified utility MDMS (such as ANSI C12.19, IEC 61968, or DLMS/COSEM certified) "
    "and must not be used for commercial revenue billing, utility grid settlements, or legal metrology."
)


@dataclass
class MDMSConfig:
    """Configuration settings for the MDMS Simulator."""

    # Project metadata
    service_name: str = "MDMS Simulator"
    version: str = "1.0.0"
    disclaimer: str = LEGAL_DISCLAIMER

    # Ingestion & Validation Thresholds
    min_consumption_kwh: float = 0.0          # Non-negative consumption limit
    max_consumption_kwh: float = 2500.0       # Sensible max threshold (allows industrial peak loads)
    max_future_skew_seconds: int = 3600       # Maximum future timestamp tolerance (1 hour)
    late_arrival_threshold_hours: int = 12    # Hours behind current high-watermark to qualify as late event

    # Configurable client partition list (3 to 5 clients matching grid topology)
    client_ids: List[str] = field(default_factory=lambda: [
        "client_A",  # Industrial Zone Substation
        "client_B",  # Commercial District Substation
        "client_C",  # Urban Residential Feeder 1
        "client_D",  # Urban Residential Feeder 2
        "client_E",  # Rural / Semi-Urban Feeder
    ])

    # Storage paths
    base_processed_dir: Path = Path("data/processed")
    db_path: Path = Path("data/processed/mdms_store.db")
    clients_dir: Path = Path("data/processed/clients")

    # API Server configuration
    host: str = "127.0.0.1"
    port: int = 8001
    batch_size: int = 64

    def ensure_directories(self) -> None:
        """Create necessary processed data directories and client partitions."""
        self.base_processed_dir.mkdir(parents=True, exist_ok=True)
        self.clients_dir.mkdir(parents=True, exist_ok=True)
        for client_id in self.client_ids:
            (self.clients_dir / client_id).mkdir(parents=True, exist_ok=True)
