"""
Comprehensive Unit & Integration Test Suite for MDMS Simulator.
Validates:
1. Schema, timestamp, pseudonymity, range, and PII validation.
2. Deduplication of telemetry readings.
3. Missing reading (NaN/None) handling.
4. Late-arriving event detection and temporal watermarks.
5. Local SQLite persistence, indexing, and operational status queries.
6. Client partitioning across regional substations (client_A - client_E).
7. STRICT PRIVACY INVARIANT: The Central FL Server must NEVER receive raw meter readings.
8. Fleet-wide data quality statistics calculation.
9. FastAPI REST API endpoints.
"""

from datetime import datetime
from pathlib import Path
import tempfile
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from mdms_simulator.api.app import create_app
from mdms_simulator.config import LEGAL_DISCLAIMER, MDMSConfig
from mdms_simulator.models import BatchIngestRequest, MeterReadingPayload
from mdms_simulator.partitioner import (
    ClientDataVault,
    LocalPartitionManager,
    RawDataPrivacyViolationError,
)
from mdms_simulator.quality import DataQualityTracker
from mdms_simulator.service import MDMSService
from mdms_simulator.storage import MDMSStorage
from mdms_simulator.validator import TelemetryValidator


# ==============================================================================
# 1. Validation Tests (Schema, Pseudonym, Ranges, PII)
# ==============================================================================

def test_valid_telemetry_payload():
    """Verify that a compliant smart meter telemetry reading passes validation."""
    validator = TelemetryValidator()
    payload = MeterReadingPayload(
        timestamp="2026-01-01 12:00:00",
        meter_id="MTR-C-0042",
        client_id="client_C",
        consumption_kwh=1.85,
        meter_status="NORMAL",
        event_type="NONE",
        hardware_tamper_flag=0
    )
    result = validator.validate(payload)
    assert result.is_valid is True
    assert result.is_missing is False


def test_invalid_pseudonym_format():
    """Verify non-pseudonymous meter identifiers are rejected."""
    validator = TelemetryValidator()
    # Malformed identifier without prefix
    try:
        MeterReadingPayload(
            timestamp="2026-01-01 12:00:00",
            meter_id="invalid_meter_123",
            client_id="client_C",
            consumption_kwh=1.0
        )
        pytest.fail("Should have raised Pydantic validation error for invalid pseudonym regex.")
    except Exception:
        pass


def test_pii_detected_in_identifier():
    """Verify identifiers containing personal information (like email) are rejected."""
    validator = TelemetryValidator()
    try:
        MeterReadingPayload(
            timestamp="2026-01-01 12:00:00",
            meter_id="user.name@domain.com",
            client_id="client_C",
            consumption_kwh=1.0
        )
        pytest.fail("Should have rejected PII in meter identifier.")
    except Exception:
        pass


def test_invalid_client_substation():
    """Verify readings referencing unconfigured client substations are rejected."""
    validator = TelemetryValidator()
    payload = MeterReadingPayload(
        timestamp="2026-01-01 12:00:00",
        meter_id="MTR-X-0001",
        client_id="client_Z",
        consumption_kwh=1.0
    )
    result = validator.validate(payload)
    assert result.is_valid is False
    assert "not in configured substation partitions" in result.error_message


def test_consumption_range_bounds():
    """Verify non-negative constraint and maximum capacity thresholds."""
    validator = TelemetryValidator()

    # Valid zero
    zero_p = MeterReadingPayload(
        timestamp="2026-01-01 12:00:00",
        meter_id="MTR-A-0001",
        client_id="client_A",
        consumption_kwh=0.0
    )
    assert validator.validate(zero_p).is_valid is True

    # Negative consumption
    neg_p = MeterReadingPayload(
        timestamp="2026-01-01 12:00:00",
        meter_id="MTR-A-0001",
        client_id="client_A",
        consumption_kwh=-1.5
    )
    res_neg = validator.validate(neg_p)
    assert res_neg.is_valid is False
    assert "Negative consumption reading" in res_neg.error_message

    # Exceeds max bounds
    extreme_p = MeterReadingPayload(
        timestamp="2026-01-01 12:00:00",
        meter_id="MTR-A-0001",
        client_id="client_A",
        consumption_kwh=5000.0
    )
    res_ext = validator.validate(extreme_p)
    assert res_ext.is_valid is False
    assert "exceeds maximum threshold" in res_ext.error_message


def test_missing_reading_nan_handling():
    """Verify null or NaN consumption is accepted and flagged as missing telemetry."""
    validator = TelemetryValidator()
    missing_p = MeterReadingPayload(
        timestamp="2026-01-01 12:00:00",
        meter_id="MTR-A-0001",
        client_id="client_A",
        consumption_kwh=None,
        meter_status="COMM_FAILURE",
        event_type="COMM_FAILURE"
    )
    res = validator.validate(missing_p)
    assert res.is_valid is True
    assert res.is_missing is True


# ==============================================================================
# 2. Deduplication & Late Arrival Tests
# ==============================================================================

def test_deduplication_tracking():
    """Verify duplicate (meter_id, timestamp) keys are detected and counted."""
    tracker = DataQualityTracker()
    m_id = "MTR-B-0010"
    ts = "2026-01-01 08:00:00"

    assert tracker.check_duplicate(m_id, ts) is False
    tracker.register_key(m_id, ts)
    assert tracker.check_duplicate(m_id, ts) is True


def test_late_arriving_event_detection():
    """Verify that a reading arriving after the client watermark is tagged as late."""
    config = MDMSConfig(late_arrival_threshold_hours=12)
    tracker = DataQualityTracker(config=config)
    client_id = "client_A"

    # Watermark set to Jan 10
    dt_now = datetime(2026, 1, 10, 12, 0, 0)
    assert tracker.evaluate_arrival_timing(client_id, dt_now) is False

    # Event from Jan 1 arrives now (9 days late > 12 hours threshold)
    dt_late = datetime(2026, 1, 1, 12, 0, 0)
    assert tracker.evaluate_arrival_timing(client_id, dt_late) is True

    # Event from Jan 10 08:00 (4 hours behind watermark < 12h threshold)
    dt_recent = datetime(2026, 1, 10, 8, 0, 0)
    assert tracker.evaluate_arrival_timing(client_id, dt_recent) is False


# ==============================================================================
# 3. Storage & Service Orchestration Tests
# ==============================================================================

def test_service_single_and_duplicate_ingest():
    """Verify service validates, persists, and deduplicates readings."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = MDMSConfig(
            base_processed_dir=tmp_path,
            db_path=tmp_path / "test_store.db",
            clients_dir=tmp_path / "clients"
        )
        service = MDMSService(config=config, db_path=config.db_path)

        payload = MeterReadingPayload(
            timestamp="2026-01-01 10:00:00",
            meter_id="MTR-C-0100",
            client_id="client_C",
            consumption_kwh=1.45,
            meter_status="NORMAL"
        )

        # First ingestion
        ok1, msg1 = service.ingest_reading(payload)
        assert ok1 is True
        assert "successfully" in msg1

        # Second ingestion (duplicate)
        ok2, msg2 = service.ingest_reading(payload)
        assert ok2 is False
        assert "Duplicate" in msg2

        # Verify meter status query
        status = service.get_meter_status("MTR-C-0100")
        assert status is not None
        assert status.meter_id == "MTR-C-0100"
        assert status.client_id == "client_C"
        assert status.latest_consumption_kwh == 1.45
        assert status.total_readings_recorded == 1
        assert status.data_quality_score_pct == 100.0


def test_service_batch_ingestion():
    """Verify high-throughput batch ingestion."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = MDMSConfig(
            base_processed_dir=tmp_path,
            db_path=tmp_path / "test_store.db",
            clients_dir=tmp_path / "clients"
        )
        service = MDMSService(config=config, db_path=config.db_path)

        batch = [
            MeterReadingPayload(
                timestamp=f"2026-01-01 {h:02d}:00:00",
                meter_id="MTR-A-0001",
                client_id="client_A",
                consumption_kwh=round(1.0 + h * 0.1, 2)
            )
            for h in range(24)
        ]

        summary = service.ingest_batch(batch)
        assert summary.received == 24
        assert summary.accepted == 24
        assert summary.duplicates == 0
        assert summary.rejected == 0

        # Submitting the exact same batch again must result in 24 duplicates
        dup_summary = service.ingest_batch(batch)
        assert dup_summary.received == 24
        assert dup_summary.accepted == 0
        assert dup_summary.duplicates == 24


# ==============================================================================
# 4. Client Partitioning & Federated Learning Privacy Boundary Tests
# ==============================================================================

def test_client_local_partitioning():
    """Verify readings are partitioned strictly into local client storage."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = MDMSConfig(
            base_processed_dir=tmp_path,
            db_path=tmp_path / "test_store.db",
            clients_dir=tmp_path / "clients"
        )
        service = MDMSService(config=config, db_path=config.db_path)

        payload_a = MeterReadingPayload(
            timestamp="2026-01-01 10:00:00",
            meter_id="MTR-A-0005",
            client_id="client_A",
            consumption_kwh=10.5
        )
        payload_b = MeterReadingPayload(
            timestamp="2026-01-01 10:00:00",
            meter_id="MTR-B-0008",
            client_id="client_B",
            consumption_kwh=5.2
        )

        service.ingest_reading(payload_a)
        service.ingest_reading(payload_b)

        vault_a = service.partitioner.get_vault("client_A")
        vault_b = service.partitioner.get_vault("client_B")

        assert vault_a.readings_csv_path.exists()
        assert vault_b.readings_csv_path.exists()

        df_a = pd.read_csv(vault_a.readings_csv_path)
        df_b = pd.read_csv(vault_b.readings_csv_path)

        # Isolation check: client_A only contains MTR-A-0005
        assert set(df_a["meter_id"]) == {"MTR-A-0005"}
        # Isolation check: client_B only contains MTR-B-0008
        assert set(df_b["meter_id"]) == {"MTR-B-0008"}


def test_strict_privacy_invariable_central_fl_server_blocked():
    """
    CRITICAL PRIVACY TEST:
    Under Federated Learning guarantees, the Central FL Server is strictly forbidden
    from accessing raw meter telemetry from any client vault.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        vault_dir = tmp_path / "client_A"
        vault = ClientDataVault(client_id="client_A", client_dir=vault_dir)

        payload = MeterReadingPayload(
            timestamp="2026-01-01 10:00:00",
            meter_id="MTR-A-0001",
            client_id="client_A",
            consumption_kwh=15.0
        )
        vault.store_reading(payload)
        vault.flush_to_disk()

        # 1. Local client access is permitted
        local_df = vault.access_raw_readings(requesting_entity="client_A")
        assert len(local_df) == 1

        # 2. Central FL server access MUST RAISE RawDataPrivacyViolationError!
        with pytest.raises(RawDataPrivacyViolationError) as exc_info:
            vault.access_raw_readings(requesting_entity="FL_SERVER")

        assert "Under Federated Learning privacy guarantees" in str(exc_info.value)

        # 3. GLOBAL_AGGREGATOR also forbidden
        with pytest.raises(RawDataPrivacyViolationError):
            vault.access_raw_readings(requesting_entity="GLOBAL_AGGREGATOR")


# ==============================================================================
# 5. Data Quality Report & API Integration Tests
# ==============================================================================

def test_data_quality_report_metrics():
    """Verify comprehensive data quality metrics calculation."""
    tracker = DataQualityTracker()
    # 10 accepted, 2 duplicates, 1 missing, 1 late, 1 rejected
    for _ in range(8):
        tracker.record_event("client_A", is_accepted=True, is_duplicate=False, is_missing=False, is_late=False, is_rejected=False)
    # Missing
    tracker.record_event("client_A", is_accepted=True, is_duplicate=False, is_missing=True, is_late=False, is_rejected=False)
    # Late
    tracker.record_event("client_A", is_accepted=True, is_duplicate=False, is_missing=False, is_late=True, is_rejected=False)
    # Duplicate
    tracker.record_event("client_A", is_accepted=False, is_duplicate=True, is_missing=False, is_late=False, is_rejected=False)
    # Rejected
    tracker.record_event("client_A", is_accepted=False, is_duplicate=False, is_missing=False, is_late=False, is_rejected=True)

    report = tracker.generate_report()
    assert report.total_received == 12
    assert report.total_accepted == 10
    assert report.total_missing_readings == 1
    assert report.total_late_arrivals == 1
    assert report.total_duplicates == 1
    assert report.total_rejected == 1
    assert report.overall_completion_rate_pct == 90.0


def test_fastapi_endpoints():
    """Integration test of MDMS FastAPI REST endpoints via TestClient."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = MDMSConfig(
            base_processed_dir=tmp_path,
            db_path=tmp_path / "api_test.db",
            clients_dir=tmp_path / "clients"
        )
        service = MDMSService(config=config, db_path=config.db_path)
        app = create_app(config=config, service=service)
        client = TestClient(app)

        # 1. Health endpoint
        res_health = client.get("/api/v1/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "operational"
        assert "RESEARCH SIMULATOR ONLY" in res_health.json()["disclaimer"]

        # 2. Ingest single reading
        reading_payload = {
            "timestamp": "2026-01-01 12:00:00",
            "meter_id": "MTR-C-0999",
            "client_id": "client_C",
            "consumption_kwh": 2.45,
            "meter_status": "NORMAL",
            "event_type": "NONE",
            "hardware_tamper_flag": 0
        }
        res_ingest = client.post("/api/v1/ingest/reading", json=reading_payload)
        assert res_ingest.status_code == 200
        assert res_ingest.json()["status"] == "accepted"

        # 3. Query meter status
        res_status = client.get("/api/v1/meters/MTR-C-0999/status")
        assert res_status.status_code == 200
        data = res_status.json()
        assert data["meter_id"] == "MTR-C-0999"
        assert data["latest_consumption_kwh"] == 2.45

        # 4. Verify privacy endpoint
        res_priv = client.get("/api/v1/clients/client_C/verify-privacy")
        assert res_priv.status_code == 200
        assert res_priv.json()["privacy_firewall_status"] == "BLOCKED"
        assert res_priv.json()["access_granted"] is False

        # 5. Data quality endpoint
        res_dq = client.get("/api/v1/data-quality")
        assert res_dq.status_code == 200
        assert res_dq.json()["total_accepted"] == 1
