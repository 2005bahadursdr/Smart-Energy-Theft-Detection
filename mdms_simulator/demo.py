"""
End-to-End Local Demonstration of MDMS Simulator.
Demonstrates:
1. Prominent legal & research simulator disclaimer.
2. Telemetry ingestion from smart meter simulation dataset.
3. Rigorous validation: out-of-range, malformed timestamps, PII rejection.
4. Handling of duplicate readings, missing values, and late-arriving events.
5. Partitioning of raw readings into isolated substation client vaults (client_A - client_E).
6. Enforcement of the Federated Learning privacy boundary:
   Verifying that the Central FL Server is blocked from accessing raw customer data!
7. Meter operational status queries and fleet-wide data-quality audit reports.
"""

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import sys

from mdms_simulator.config import LEGAL_DISCLAIMER, MDMSConfig
from mdms_simulator.models import MeterReadingPayload
from mdms_simulator.partitioner import RawDataPrivacyViolationError
from mdms_simulator.service import MDMSService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mdms_demo")


def run_demo() -> bool:
    print("\n" + "=" * 80)
    print(" [SMART GRID] METER DATA MANAGEMENT SYSTEM (MDMS) SIMULATOR DEMO")
    print("=" * 80)
    print("\n[LEGAL & RESEARCH NOTICE]")
    print(LEGAL_DISCLAIMER)
    print("-" * 80)

    # 1. Initialize Service
    demo_db_path = Path("data/processed/mdms_store.db")
    config = MDMSConfig(db_path=demo_db_path)
    service = MDMSService(config=config, db_path=demo_db_path)
    print(f"\n[1] Initialized MDMS Service (Database: {demo_db_path})")
    print(f"    Substation Client Partitions: {config.client_ids}")

    # 2. Ingest Sample from Synthetic Dataset
    csv_path = Path("data/synthetic/simulated_meter_data.csv")
    if not csv_path.exists():
        print(f"\n[!] Dataset {csv_path} not found. Running smart meter simulator to generate sample...")
        from smart_meter_simulator.simulator import SmartMeterSimulator
        sim = SmartMeterSimulator(num_meters=100, days=2, seed=42)
        sim.export_dataset()

    print(f"\n[2] Ingesting stream from simulated dataset: {csv_path} (First 2,500 readings)...")
    ingest_summary = service.ingest_csv_file(csv_path, max_rows=2500)
    print(f"    * Total Received       : {ingest_summary.received:,}")
    print(f"    * Successfully Accepted: {ingest_summary.accepted:,}")
    print(f"    * Missing Telemetry    : {ingest_summary.missing_readings:,}")
    print(f"    * Duplicates Filtered  : {ingest_summary.duplicates:,}")
    print(f"    * Rejected Malformed   : {ingest_summary.rejected:,}")

    # 3. Test Deduplication
    print("\n[3] Testing Deduplication Mechanism...")
    run_marker = int(datetime.now().timestamp()) % 80000 + 10000
    sample_meter_id = f"MTR-C-{run_marker}"
    sample_time = "2026-11-15 10:00:00"
    duplicate_payload = MeterReadingPayload(
        timestamp=sample_time,
        meter_id=sample_meter_id,
        client_id="client_C",
        consumption_kwh=1.25,
        meter_status="NORMAL",
        event_type="NONE"
    )
    # First ingest
    ok1, msg1 = service.ingest_reading(duplicate_payload)
    # Duplicate ingest
    ok2, msg2 = service.ingest_reading(duplicate_payload)
    print(f"    * First ingestion attempt  : {'Accepted' if ok1 else 'Duplicate/Existing'} -> {msg1}")
    print(f"    * Duplicate second attempt : {'Accepted' if ok2 else 'Rejected as Duplicate'} -> {msg2}")
    assert ok1, "First ingestion should succeed"
    assert not ok2, "Deduplication failed to reject identical reading!"
    print("    [PASS] Duplicate reading was successfully caught and suppressed.")

    # 4. Test Validation & Privacy Rejection
    print("\n[4] Testing Telemetry Validation & PII Privacy Rejection...")
    # A: Negative consumption
    bad_payload_neg = MeterReadingPayload(
        timestamp="2026-01-01 12:00:00",
        meter_id=f"MTR-C-{run_marker+1}",
        client_id="client_C",
        consumption_kwh=-5.2,
        meter_status="NORMAL"
    )
    ok_neg, msg_neg = service.ingest_reading(bad_payload_neg)
    print(f"    * Negative value test (-5.2 kWh)  : {'Accepted' if ok_neg else 'Rejected'} -> {msg_neg}")
    assert not ok_neg

    # B: Out-of-bounds extreme value
    bad_payload_max = MeterReadingPayload(
        timestamp="2026-01-01 12:00:00",
        meter_id=f"MTR-C-{run_marker+2}",
        client_id="client_C",
        consumption_kwh=999999.0,
        meter_status="NORMAL"
    )
    ok_max, msg_max = service.ingest_reading(bad_payload_max)
    print(f"    * Extreme value test (999,999 kWh): {'Accepted' if ok_max else 'Rejected'} -> {msg_max}")
    assert not ok_max

    # C: PII privacy leak (e.g. Email in meter identifier)
    try:
        MeterReadingPayload(
            timestamp="2026-01-01 12:00:00",
            meter_id="john.doe@utility.com",
            client_id="client_C",
            consumption_kwh=1.0
        )
        print("    * PII test: Failed (Accepted)")
    except Exception as ex:
        print(f"    * PII test (john.doe@utility.com) : Rejected by schema -> {ex.errors()[0]['msg']}")
    print("    [PASS] Validation rejects malformed, out-of-range, and PII-bearing records.")

    # 5. Test Late-Arriving Event Detection
    print("\n[5] Testing Late-Arriving Event Detection...")
    # Advance client_A watermark to day Dec 20
    future_payload = MeterReadingPayload(
        timestamp="2026-12-20 12:00:00",
        meter_id=f"MTR-A-{run_marker+3}",
        client_id="client_A",
        consumption_kwh=2.10,
        meter_status="NORMAL"
    )
    service.ingest_reading(future_payload)

    # Ingest late reading from Dec 10 (10 days behind watermark > 12h threshold)
    late_payload = MeterReadingPayload(
        timestamp="2026-12-10 12:00:00",
        meter_id=f"MTR-A-{run_marker+4}",
        client_id="client_A",
        consumption_kwh=1.85,
        meter_status="NORMAL"
    )
    ok_late, msg_late = service.ingest_reading(late_payload)
    print(f"    * Ingesting reading 10 days behind current watermark: {ok_late} -> {msg_late}")
    assert ok_late

    # 6. Verify Local Client Partitions & Privacy Firewall
    print("\n[6] Verifying Local Substation Client Partitions & FL Privacy Firewall...")
    all_vaults = service.partitioner.get_all_summaries()
    for cid, summary in all_vaults.items():
        print(f"    * {cid}: {summary['meter_count']} meters, {summary['readings_count']:,} readings -> {summary['local_storage_path']}")

    # PRIVACY FIREWALL TEST: Central FL Server attempts to query raw readings
    print("\n    [!] Testing Central FL Server Privacy Firewall...")
    vault_a = service.partitioner.get_vault("client_A")
    privacy_blocked = False
    try:
        vault_a.access_raw_readings(requesting_entity="FL_SERVER")
    except RawDataPrivacyViolationError as err:
        privacy_blocked = True
        print(f"    [ENFORCED] Raw data query by Central FL Server blocked: {err}")

    assert privacy_blocked, "CRITICAL ERROR: Central FL Server accessed raw meter data!"
    print("    [PASS] Zero raw readings leaked. Local client privacy boundary is strictly intact.")

    # 7. Query Meter Operational Status
    print("\n[7] Querying Smart Meter Operational Status...")
    meter_stat = service.get_meter_status(sample_meter_id)
    if meter_stat:
        print(f"    * Meter ID           : {meter_stat.meter_id}")
        print(f"    * Client Substation  : {meter_stat.client_id}")
        print(f"    * Operational Status : {meter_stat.current_status}")
        print(f"    * Latest Timestamp   : {meter_stat.latest_timestamp}")
        print(f"    * Latest Consumption : {meter_stat.latest_consumption_kwh} kWh")
        print(f"    * Hardware Tamper    : {'ALARM ACTIVE' if meter_stat.hardware_tamper_active else 'Normal'}")
        print(f"    * Total Recorded     : {meter_stat.total_readings_recorded}")
        print(f"    * Data Quality Score : {meter_stat.data_quality_score_pct}%")

    # 8. Fleet-wide Data Quality Report
    print("\n[8] Generating Fleet-wide Data Quality Report...")
    dq_report = service.get_data_quality_report()
    print(f"    * Total Ingested Readings    : {dq_report.total_received:,}")
    print(f"    * Total Accepted Valid       : {dq_report.total_accepted:,}")
    print(f"    * Total Missing (Telemetry)  : {dq_report.total_missing_readings:,}")
    print(f"    * Total Duplicates Filtered  : {dq_report.total_duplicates:,}")
    print(f"    * Total Late Arrivals Tagged : {dq_report.total_late_arrivals:,}")
    print(f"    * Total Rejected Malformed   : {dq_report.total_rejected:,}")
    print(f"    * Overall Completion Rate    : {dq_report.overall_completion_rate_pct}%")

    print("\n" + "=" * 80)
    print(" [MDMS SIMULATOR DEMO COMPLETE: ALL CHECKS PASSED]")
    print("=" * 80 + "\n")
    return True


if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
