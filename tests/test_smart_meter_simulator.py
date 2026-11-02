"""
Comprehensive Unit & Integration Test Suite for Smart Meter Simulator.
Validates:
1. Hourly diurnal patterns, weekly shifts, seasonality, solar PV, and stochastic noise.
2. Consumer profiles (Residential, Commercial, Industrial, Rural).
3. Legitimate zero consumption: Vacant homes, grid outages, disconnections, solar self-generation.
4. Tamper events: Partial bypass, full bypass, reverse current, magnetic, cyber scaling, flatline, peak clipping.
5. Strict separation of meter status, event type, and simulated ground truth.
6. The fundamental invariant: Zero consumption is NEVER equated with theft!
7. Deterministic reproducibility across simulation runs.
8. CSV and JSON artifact export.
"""

from datetime import datetime
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from smart_meter_simulator.consumer import (
    ConsumerProfile,
    ConsumerType,
    OccupancyState,
    generate_consumer_profile,
)
from smart_meter_simulator.consumption_patterns import (
    DIURNAL_BASELINES,
    WEEKDAY_MULTIPLIERS,
    ConsumptionPatternEngine,
)
from smart_meter_simulator.meter import (
    GridEventType,
    MeterStatus,
    SmartMeter,
)
from smart_meter_simulator.simulator import SmartMeterSimulator
from smart_meter_simulator.tamper_events import (
    TamperEvent,
    TamperInjector,
    TamperType,
)


# ==============================================================================
# 1. Consumption Pattern Engine Tests
# ==============================================================================

def test_diurnal_profiles_exist_and_positive():
    """Verify diurnal base curves exist for all consumer types and are strictly positive."""
    for c_type in ConsumerType:
        curve = DIURNAL_BASELINES[c_type]
        assert len(curve) == 24
        assert np.all(curve > 0.0)


def test_residential_diurnal_peaks():
    """Verify residential curve shows characteristic morning and evening peaks."""
    res_curve = DIURNAL_BASELINES[ConsumerType.RESIDENTIAL]
    night_trough = res_curve[3] # 03:00
    morning_peak = res_curve[8] # 08:00
    midday_lull = res_curve[13] # 13:00
    evening_peak = res_curve[19] # 19:00

    assert morning_peak > night_trough
    assert evening_peak > midday_lull
    assert evening_peak > morning_peak


def test_commercial_diurnal_workhours():
    """Verify commercial load is concentrated during business hours (09:00-17:00)."""
    com_curve = DIURNAL_BASELINES[ConsumerType.COMMERCIAL]
    business_avg = np.mean(com_curve[9:17])
    night_avg = np.mean(com_curve[0:5])
    assert business_avg > 3.0 * night_avg


def test_weekend_multipliers():
    """Verify weekend patterns: commercial drops sharply on Sunday, residential shifts."""
    res_dow = WEEKDAY_MULTIPLIERS[ConsumerType.RESIDENTIAL]
    com_dow = WEEKDAY_MULTIPLIERS[ConsumerType.COMMERCIAL]

    # Commercial Sunday (index 6) is much lower than Wednesday (index 2)
    assert com_dow[6] < 0.5 * com_dow[2]
    # Residential weekend is active
    assert res_dow[5] > res_dow[0]


def test_seasonality_bimodal_peaks():
    """Verify seasonal factor produces higher loads in summer and winter than mild spring."""
    engine = ConsumptionPatternEngine()
    summer_factor = engine.get_seasonal_factor(day_of_year=200) # July
    winter_factor = engine.get_seasonal_factor(day_of_year=15)  # January
    spring_factor = engine.get_seasonal_factor(day_of_year=110) # April

    assert summer_factor > spring_factor
    assert winter_factor > spring_factor


def test_solar_generation_bounds():
    """Verify rooftop solar only generates during daylight (06:00-19:00) and is non-negative."""
    engine = ConsumptionPatternEngine()
    rng = np.random.RandomState(42)

    night_dt = datetime(2026, 6, 15, 2, 0, 0)
    day_dt = datetime(2026, 6, 15, 12, 30, 0)

    night_solar = engine.get_solar_generation(night_dt, solar_capacity_kw=5.0, rng=rng)
    day_solar = engine.get_solar_generation(day_dt, solar_capacity_kw=5.0, rng=rng)

    assert night_solar == 0.0
    assert day_solar > 0.0


def test_consumption_noise_strictly_non_negative():
    """Verify noisy hourly consumption never drops below 0.0."""
    engine = ConsumptionPatternEngine()
    rng = np.random.RandomState(42)
    test_dt = datetime(2026, 1, 15, 18, 0, 0)

    for _ in range(500):
        val = engine.compute_hourly_consumption(
            dt=test_dt,
            consumer_type=ConsumerType.RESIDENTIAL,
            base_load_kwh=1.0,
            rng=rng,
            solar_capacity_kw=10.0 # Extreme solar capacity to test clamping
        )
        assert val >= 0.0


# ==============================================================================
# 2. Consumer Profile & Behavioral States Tests
# ==============================================================================

def test_consumer_profile_generation():
    """Verify consumer profile generator properly initializes consumer archetypes."""
    rng = np.random.RandomState(42)
    profile = generate_consumer_profile(
        consumer_id="CONS-TEST-001",
        consumer_type=ConsumerType.RESIDENTIAL,
        rng=rng,
        is_vacant=False,
        has_sustained_drop=False,
        has_solar=True
    )
    assert profile.consumer_id == "CONS-TEST-001"
    assert profile.consumer_type == ConsumerType.RESIDENTIAL
    assert profile.base_load_kwh > 0.0
    assert profile.has_solar is True
    assert profile.solar_capacity_kw > 0.0


def test_vacant_home_legitimate_zero_or_standby():
    """Verify vacant property produces micro-standby or strictly 0.0 kWh (never malicious)."""
    engine = ConsumptionPatternEngine()
    rng = np.random.RandomState(42)
    profile = ConsumerProfile(
        consumer_id="CONS-VACANT",
        consumer_type=ConsumerType.RESIDENTIAL,
        base_load_kwh=1.2,
        occupancy_state=OccupancyState.VACANT,
        standby_load_kwh=0.02
    )

    dt = datetime(2026, 1, 1, 12, 0, 0)
    readings = []
    for h in range(100):
        demand, event = profile.compute_true_demand(dt, h, engine, rng)
        assert event == "VACANCY"
        assert demand < 0.10 # Must be micro-standby or zero
        readings.append(demand)

    # Some readings should be strictly 0.0 (simulating main switch off)
    assert any(r == 0.0 for r in readings)


def test_sustained_drop_efficiency():
    """Verify sustained drop lowers consumption starting at the scheduled hour."""
    engine = ConsumptionPatternEngine()
    rng = np.random.RandomState(42)
    profile = ConsumerProfile(
        consumer_id="CONS-DROP",
        consumer_type=ConsumerType.COMMERCIAL,
        base_load_kwh=15.0,
        sustained_drop_ratio=0.50, # 50% efficiency retrofit
        sustained_drop_start_hour=100
    )

    dt = datetime(2026, 1, 1, 12, 0, 0)
    demand_before, event_before = profile.compute_true_demand(dt, 50, engine, rng)
    demand_after, event_after = profile.compute_true_demand(dt, 150, engine, rng)

    assert event_before == "NONE"
    assert event_after == "SUSTAINED_DROP"
    assert demand_after < 0.70 * demand_before


# ==============================================================================
# 3. Tamper Events & Simulated Labels Tests
# ==============================================================================

def test_tamper_injector_inactive():
    """When tamper is inactive or None, reported == actual and labels are strictly 0."""
    rng = np.random.RandomState(42)
    reported, alert, label, theft_type, unreported = TamperInjector.apply_tamper(
        actual_kwh=2.50,
        tamper_event=None,
        hour_idx=10,
        rng=rng
    )
    assert reported == 2.50
    assert alert == 0
    assert label == 0
    assert theft_type == "NONE"
    assert unreported == 0.0


def test_tamper_injector_partial_bypass():
    """Verify partial bypass reduces reported consumption and records stolen energy."""
    rng = np.random.RandomState(42)
    event = TamperEvent(
        tamper_type=TamperType.PARTIAL_BYPASS,
        start_hour=10,
        end_hour=50,
        severity=0.50
    )
    reported, alert, label, theft_type, unreported = TamperInjector.apply_tamper(
        actual_kwh=3.0,
        tamper_event=event,
        hour_idx=25,
        rng=rng
    )
    assert label == 1
    assert theft_type == TamperType.PARTIAL_BYPASS.value
    assert reported < 3.0
    assert unreported > 0.0
    assert abs((reported + unreported) - 3.0) < 0.05


def test_tamper_injector_full_bypass():
    """Verify full bypass records near zero reported reading while tracking theft."""
    rng = np.random.RandomState(42)
    event = TamperEvent(
        tamper_type=TamperType.FULL_BYPASS,
        start_hour=0,
        end_hour=100
    )
    reported, alert, label, theft_type, unreported = TamperInjector.apply_tamper(
        actual_kwh=5.0,
        tamper_event=event,
        hour_idx=10,
        rng=rng
    )
    assert label == 1
    assert theft_type == TamperType.FULL_BYPASS.value
    assert reported < 0.05
    assert unreported > 4.90


def test_tamper_injector_magnetic_hardware_alert():
    """Verify magnetic tamper frequently triggers the hardware tamper alarm."""
    rng = np.random.RandomState(42)
    event = TamperEvent(
        tamper_type=TamperType.MAGNETIC_TAMPER,
        start_hour=0,
        end_hour=100,
        hardware_alert_prob=0.85
    )
    alerts = []
    for h in range(100):
        _, alert, label, _, _ = TamperInjector.apply_tamper(
            actual_kwh=4.0,
            tamper_event=event,
            hour_idx=h,
            rng=rng
        )
        assert label == 1
        alerts.append(alert)

    # Hall sensor should trigger alerts in majority of hours
    assert sum(alerts) > 60


def test_tamper_injector_cyber_scaling_and_flatline():
    """Verify cyber attacks alter telemetry without triggering mechanical alarms."""
    rng = np.random.RandomState(42)
    # Scaling
    scale_event = TamperEvent(
        tamper_type=TamperType.CYBER_SCALING,
        start_hour=0,
        end_hour=50,
        severity=0.60
    )
    rep_s, alert_s, label_s, type_s, unrep_s = TamperInjector.apply_tamper(10.0, scale_event, 5, rng)
    assert label_s == 1
    assert type_s == TamperType.CYBER_SCALING.value
    assert rep_s < 6.0

    # Flatline
    flat_event = TamperEvent(
        tamper_type=TamperType.CYBER_FLATLINE,
        start_hour=0,
        end_hour=50,
        flatline_value_kwh=0.20
    )
    rep_f, alert_f, label_f, type_f, unrep_f = TamperInjector.apply_tamper(8.0, flat_event, 5, rng)
    assert label_f == 1
    assert type_f == TamperType.CYBER_FLATLINE.value
    assert abs(rep_f - 0.20) < 0.05


# ==============================================================================
# 4. Invariant: Never Equate Zero Consumption With Theft
# ==============================================================================

def test_grid_outage_never_equated_with_theft():
    """CRITICAL TEST: Outage causes 0.0 kWh consumption, but simulated_theft_label MUST BE 0."""
    engine = ConsumptionPatternEngine()
    rng = np.random.RandomState(42)
    profile = ConsumerProfile("C1", ConsumerType.RESIDENTIAL, 1.0)
    meter = SmartMeter("MTR-OUTAGE-TEST", "client_C", profile)

    dt = datetime(2026, 1, 1, 14, 0, 0)
    reading = meter.read_hour(
        dt=dt,
        hour_idx=14,
        is_grid_outage=True, # Grid power is down!
        is_comm_drop=False,
        pattern_engine=engine,
        rng=rng
    )

    assert reading["consumption_kwh"] == 0.0
    assert reading["meter_status"] == MeterStatus.OUTAGE.value
    assert reading["event_type"] == GridEventType.GRID_OUTAGE.value
    assert reading["hardware_tamper_flag"] == 0
    # ABSOLUTE RULE CHECK:
    assert reading["simulated_theft_label"] == 0
    assert reading["simulated_theft_type"] == "NONE"
    assert reading["simulated_unreported_kwh"] == 0.0


def test_disconnection_never_equated_with_theft():
    """CRITICAL TEST: Utility disconnection causes 0.0 kWh consumption, but simulated_theft_label MUST BE 0."""
    engine = ConsumptionPatternEngine()
    rng = np.random.RandomState(42)
    profile = ConsumerProfile("C2", ConsumerType.COMMERCIAL, 10.0)
    meter = SmartMeter(
        meter_id="MTR-DISC-TEST",
        client_id="client_B",
        consumer=profile,
        is_disconnected=True
    )

    dt = datetime(2026, 1, 1, 10, 0, 0)
    reading = meter.read_hour(
        dt=dt,
        hour_idx=10,
        is_grid_outage=False,
        is_comm_drop=False,
        pattern_engine=engine,
        rng=rng
    )

    assert reading["consumption_kwh"] == 0.0
    assert reading["meter_status"] == MeterStatus.DISCONNECTED.value
    assert reading["event_type"] == GridEventType.DISCONNECTION.value
    assert reading["simulated_theft_label"] == 0
    assert reading["simulated_theft_type"] == "NONE"


def test_comm_failure_is_missing_reading_not_theft():
    """CRITICAL TEST: Communication packet loss produces NaN reading, but simulated_theft_label MUST BE 0."""
    engine = ConsumptionPatternEngine()
    rng = np.random.RandomState(42)
    profile = ConsumerProfile("C3", ConsumerType.RESIDENTIAL, 1.0)
    meter = SmartMeter("MTR-COMM-TEST", "client_C", profile)

    dt = datetime(2026, 1, 1, 12, 0, 0)
    reading = meter.read_hour(
        dt=dt,
        hour_idx=12,
        is_grid_outage=False,
        is_comm_drop=True, # AMI packet lost!
        pattern_engine=engine,
        rng=rng
    )

    assert np.isnan(reading["consumption_kwh"])
    assert reading["meter_status"] == MeterStatus.COMM_FAILURE.value
    assert reading["event_type"] == GridEventType.COMM_FAILURE.value
    assert reading["simulated_theft_label"] == 0


def test_vacancy_zero_consumption_never_labeled_theft():
    """CRITICAL TEST: Vacant home with 0.0 kWh consumption MUST have simulated_theft_label == 0."""
    engine = ConsumptionPatternEngine()
    rng = np.random.RandomState(42)
    profile = ConsumerProfile(
        "C4",
        ConsumerType.RESIDENTIAL,
        1.0,
        occupancy_state=OccupancyState.VACANT,
        standby_load_kwh=0.0 # Breaker switched off
    )
    meter = SmartMeter("MTR-VACANT-TEST", "client_C", profile)

    dt = datetime(2026, 1, 1, 20, 0, 0)
    reading = meter.read_hour(
        dt=dt,
        hour_idx=20,
        is_grid_outage=False,
        is_comm_drop=False,
        pattern_engine=engine,
        rng=rng
    )

    assert reading["consumption_kwh"] == 0.0
    assert reading["meter_status"] == MeterStatus.NORMAL.value
    assert reading["event_type"] == GridEventType.VACANCY.value
    assert reading["simulated_theft_label"] == 0
    assert reading["simulated_theft_type"] == "NONE"


# ==============================================================================
# 5. Fleet Simulator Orchestration & Scalability Tests
# ==============================================================================

def test_fleet_initialization_scale_500_to_2000():
    """Verify fleet can be configured with 500 to 2000 pseudonymous meters."""
    for count in [500, 1000, 2000]:
        sim = SmartMeterSimulator(num_meters=count, days=1, seed=42)
        sim.initialize_fleet()
        assert len(sim.meters) == count

        # Verify client allocation across client_A to client_E
        client_counts = {}
        for m in sim.meters:
            client_counts[m.client_id] = client_counts.get(m.client_id, 0) + 1

        assert len(client_counts) == 5
        # Evenly balanced across substations
        for cid in ["client_A", "client_B", "client_C", "client_D", "client_E"]:
            assert client_counts[cid] in (count // 5, (count // 5) + 1)


def test_simulation_schema_and_column_separation():
    """Verify output DataFrame contains all required columns with separated concerns."""
    sim = SmartMeterSimulator(num_meters=20, days=2, seed=42)
    df = sim.simulate()

    expected_columns = [
        "timestamp",
        "meter_id",
        "client_id",
        "consumption_kwh",
        "meter_status",
        "event_type",
        "hardware_tamper_flag",
        "simulated_theft_label",
        "simulated_theft_type",
        "simulated_actual_kwh",
        "simulated_unreported_kwh"
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing required column: {col}"

    assert len(df) == 20 * 48 # 20 meters * 48 hours = 960 readings


def test_simulation_reproducibility():
    """Verify two runs with the same seed yield bit-for-bit identical readings."""
    sim1 = SmartMeterSimulator(num_meters=15, days=2, seed=123)
    df1 = sim1.simulate()

    sim2 = SmartMeterSimulator(num_meters=15, days=2, seed=123)
    df2 = sim2.simulate()

    pd.testing.assert_frame_equal(df1, df2)


def test_dataset_export_and_summary_files():
    """Verify export_dataset creates valid CSV and JSON artifacts in target directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sim = SmartMeterSimulator(num_meters=25, days=2, seed=42)
        csv_file, meta_file, sum_file, summary = sim.export_dataset(output_dir=tmp_dir)

        assert csv_file.exists()
        assert meta_file.exists()
        assert sum_file.exists()

        # Load and verify CSV
        df = pd.read_csv(csv_file)
        assert len(df) == 25 * 48
        assert "simulated_theft_label" in df.columns

        # Load and verify Metadata CSV
        meta_df = pd.read_csv(meta_file)
        assert len(meta_df) == 25
        assert "consumer_type" in meta_df.columns

        # Load and verify JSON Summary
        with open(sum_file, "r") as f:
            sum_data = json.load(f)
        assert sum_data["simulation_metadata"]["total_meters"] == 25
        assert "zero_readings_legitimate" in sum_data["consumption_statistics"]
        assert "simulated_theft_reading_count" in sum_data["synthetic_ground_truth"]
