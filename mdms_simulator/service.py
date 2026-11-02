"""
MDMS Service Core Orchestrator.
Coordinates telemetry validation, deduplication, temporal watermark evaluation,
local SQLite persistence, data quality tracking, and regional client partitioning.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from mdms_simulator.config import MDMSConfig
from mdms_simulator.models import (
    IngestSummary,
    MeterReadingPayload,
    MeterStatusResponse,
    DataQualityReport,
)
from mdms_simulator.partitioner import LocalPartitionManager
from mdms_simulator.quality import DataQualityTracker
from mdms_simulator.storage import MDMSStorage
from mdms_simulator.validator import TelemetryValidator

logger = logging.getLogger("mdms_service")


class MDMSService:
    """
    Central service interface for MDMS simulation.
    """

    def __init__(self, config: Optional[MDMSConfig] = None, db_path: Optional[Path] = None):
        self.config = config or MDMSConfig()
        self.config.ensure_directories()

        self.validator = TelemetryValidator(self.config)
        self.quality = DataQualityTracker(self.config)
        self.storage = MDMSStorage(db_path=db_path, config=self.config)
        self.partitioner = LocalPartitionManager(self.config)

    def ingest_reading(self, payload: MeterReadingPayload) -> Tuple[bool, str]:
        """
        Ingest and validate single meter telemetry reading.

        Returns
        -------
        Tuple[bool, str]
            (accepted, message)
        """
        # 1. Duplicate check (in-memory fast cache)
        if self.quality.check_duplicate(payload.meter_id, payload.timestamp):
            self.quality.record_event(
                client_id=payload.client_id,
                is_accepted=False,
                is_duplicate=True,
                is_missing=False,
                is_late=False,
                is_rejected=False
            )
            return False, f"Duplicate reading detected for meter '{payload.meter_id}' at {payload.timestamp}."

        # 2. Schema and Range Validation
        val_res = self.validator.validate(payload)
        if not val_res:
            self.quality.record_event(
                client_id=payload.client_id,
                is_accepted=False,
                is_duplicate=False,
                is_missing=False,
                is_late=False,
                is_rejected=True
            )
            return False, val_res.error_message or "Validation failed."

        # 3. Temporal watermark check (late arrival detection)
        dt = datetime.strptime(payload.timestamp, "%Y-%m-%d %H:%M:%S")
        is_late = self.quality.evaluate_arrival_timing(payload.client_id, dt)
        is_missing = val_res.is_missing

        # 4. Local SQLite Persistence
        success = self.storage.insert_reading(payload, is_late=is_late, is_missing=is_missing)
        if not success:
            # Caught by database unique index constraint
            self.quality.record_event(
                client_id=payload.client_id,
                is_accepted=False,
                is_duplicate=True,
                is_missing=False,
                is_late=False,
                is_rejected=False
            )
            return False, f"Duplicate reading rejected by database index for meter '{payload.meter_id}'."

        # 5. Route to Client Local Vault (Strict Federated Learning Local Isolation)
        self.partitioner.route_reading(payload, is_late=is_late)
        self.partitioner.vaults[payload.client_id].flush_to_disk()

        # 6. Update Data Quality Tracker
        self.quality.register_key(payload.meter_id, payload.timestamp)
        self.quality.record_event(
            client_id=payload.client_id,
            is_accepted=True,
            is_duplicate=False,
            is_missing=is_missing,
            is_late=is_late,
            is_rejected=False
        )

        return True, "Reading ingested successfully."

    def ingest_batch(self, readings: List[MeterReadingPayload]) -> IngestSummary:
        """
        Batch ingest list of meter readings with high-throughput processing.
        """
        received = len(readings)
        accepted = 0
        duplicates = 0
        missing = 0
        late = 0
        rejected = 0
        errors: List[str] = []

        valid_storage_batch: List[Tuple[MeterReadingPayload, bool, bool]] = []

        for payload in readings:
            # 1. Fast duplicate check
            if self.quality.check_duplicate(payload.meter_id, payload.timestamp):
                duplicates += 1
                self.quality.record_event(
                    client_id=payload.client_id,
                    is_accepted=False,
                    is_duplicate=True,
                    is_missing=False,
                    is_late=False,
                    is_rejected=False
                )
                continue

            # 2. Validation
            val_res = self.validator.validate(payload)
            if not val_res:
                rejected += 1
                errors.append(f"{payload.meter_id} @ {payload.timestamp}: {val_res.error_message}")
                self.quality.record_event(
                    client_id=payload.client_id,
                    is_accepted=False,
                    is_duplicate=False,
                    is_missing=False,
                    is_late=False,
                    is_rejected=True
                )
                continue

            # 3. Timing & Missing
            dt = datetime.strptime(payload.timestamp, "%Y-%m-%d %H:%M:%S")
            is_late = self.quality.evaluate_arrival_timing(payload.client_id, dt)
            is_miss = val_res.is_missing

            if is_late:
                late += 1
            if is_miss:
                missing += 1

            valid_storage_batch.append((payload, is_late, is_miss))
            self.quality.register_key(payload.meter_id, payload.timestamp)

        # 4. Batch Database Insert
        if valid_storage_batch:
            ins_count, db_dups = self.storage.insert_batch(valid_storage_batch)
            duplicates += db_dups
            accepted = ins_count

            # 5. Route to Client Vaults
            for payload, is_late, is_miss in valid_storage_batch:
                self.partitioner.route_reading(payload, is_late=is_late)
                self.quality.record_event(
                    client_id=payload.client_id,
                    is_accepted=True,
                    is_duplicate=False,
                    is_missing=is_miss,
                    is_late=is_late,
                    is_rejected=False
                )

            # Flush client vaults
            self.partitioner.flush_all()

        status_str = "success"
        if rejected > 0 or duplicates > 0:
            status_str = "partial" if accepted > 0 else "rejected"

        return IngestSummary(
            status=status_str,
            received=received,
            accepted=accepted,
            duplicates=duplicates,
            missing_readings=missing,
            late_arrivals=late,
            rejected=rejected,
            errors=errors[:20]  # Cap error list to prevent massive responses
        )

    def ingest_csv_file(
        self,
        csv_path: Path,
        chunk_size: int = 5000,
        max_rows: Optional[int] = None
    ) -> IngestSummary:
        """
        Ingest smart meter simulator CSV export file in streamed chunks.
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"Source CSV not found at: {csv_path}")

        total_received = 0
        total_accepted = 0
        total_duplicates = 0
        total_missing = 0
        total_late = 0
        total_rejected = 0
        all_errors: List[str] = []

        logger.info(f"Ingesting smart meter telemetry from CSV: {csv_path}...")

        rows_read = 0
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            if max_rows and rows_read >= max_rows:
                break

            if max_rows and (rows_read + len(chunk) > max_rows):
                chunk = chunk.iloc[:(max_rows - rows_read)]

            payloads: List[MeterReadingPayload] = []
            for _, row in chunk.iterrows():
                try:
                    val = row.get("consumption_kwh")
                    val_float = None if (pd.isna(val) or val is None) else float(val)

                    payload = MeterReadingPayload(
                        timestamp=str(row["timestamp"]),
                        meter_id=str(row["meter_id"]),
                        client_id=str(row["client_id"]),
                        consumption_kwh=val_float,
                        meter_status=str(row.get("meter_status", "NORMAL")),
                        event_type=str(row.get("event_type", "NONE")),
                        hardware_tamper_flag=int(row.get("hardware_tamper_flag", 0)),
                        simulated_theft_label=int(row.get("simulated_theft_label", 0)) if pd.notna(row.get("simulated_theft_label")) else 0,
                        simulated_theft_type=str(row.get("simulated_theft_type", "NONE")),
                        simulated_actual_kwh=float(row["simulated_actual_kwh"]) if pd.notna(row.get("simulated_actual_kwh")) else None,
                        simulated_unreported_kwh=float(row["simulated_unreported_kwh"]) if pd.notna(row.get("simulated_unreported_kwh")) else 0.0,
                    )
                    payloads.append(payload)
                except Exception as ex:
                    total_rejected += 1
                    all_errors.append(f"Row conversion error: {ex}")

            summary = self.ingest_batch(payloads)
            total_received += summary.received
            total_accepted += summary.accepted
            total_duplicates += summary.duplicates
            total_missing += summary.missing_readings
            total_late += summary.late_arrivals
            total_rejected += summary.rejected
            all_errors.extend(summary.errors)

            rows_read += len(chunk)

        status_str = "success" if (total_rejected == 0 and total_duplicates == 0) else "partial"
        return IngestSummary(
            status=status_str,
            received=total_received,
            accepted=total_accepted,
            duplicates=total_duplicates,
            missing_readings=total_missing,
            late_arrivals=total_late,
            rejected=total_rejected,
            errors=all_errors[:20]
        )

    def get_meter_status(self, meter_id: str) -> Optional[MeterStatusResponse]:
        """Fetch current operational status of a meter."""
        return self.storage.get_meter_status(meter_id)

    def list_meters(
        self,
        client_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List registered meters with current status."""
        return self.storage.list_meters(client_id=client_id, meter_status=status, limit=limit)

    def get_data_quality_report(self) -> DataQualityReport:
        """Generate comprehensive data-quality statistics."""
        return self.quality.generate_report()

    def get_client_summary(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Fetch local partition summary for a specific client."""
        vault = self.partitioner.get_vault(client_id)
        if not vault:
            return None
        summary = vault.get_summary()
        summary["quality_stats"] = self.quality.client_stats.get(client_id, {})
        return summary

    def get_all_client_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all client partition summaries."""
        summaries = self.partitioner.get_all_summaries()
        for cid, s in summaries.items():
            s["quality_stats"] = self.quality.client_stats.get(cid, {})
        return summaries

    def ingest_sgcc_dataset(
        self,
        raw_csv_path: Optional[Path] = None,
        num_meters: int = 1000
    ) -> Dict[str, Any]:
        """
        Process, clean, and partition the raw SGCC benchmark smart meter dataset
        into regional substation vaults (client_A to client_E).
        """
        from mdms_simulator.sgcc_processor import SGCCDatasetProcessor
        processor = SGCCDatasetProcessor(raw_csv_path=raw_csv_path, config=self.config)
        return processor.process_and_partition(num_meters=num_meters)
