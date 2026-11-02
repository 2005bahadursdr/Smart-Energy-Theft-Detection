"""
Consumer Profile Module.
Defines consumer archetypes, baseline energy demands, occupancy states (active vs vacant),
sustained efficiency drops, and rooftop solar configurations.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple
import numpy as np

from smart_meter_simulator.consumption_patterns import (
    ConsumerType,
    ConsumptionPatternEngine,
)


class OccupancyState(str, Enum):
    """Consumer occupancy or structural consumption state."""
    ACTIVE = "ACTIVE"                   # Normal occupied operation
    VACANT = "VACANT"                   # Unoccupied premises (zero or micro-standby load)
    SUSTAINED_DROP = "SUSTAINED_DROP"   # Permanent efficiency drop (retrofit, downsizing)


@dataclass
class ConsumerProfile:
    """
    Profile representing a smart meter customer account.
    """
    consumer_id: str
    consumer_type: ConsumerType
    base_load_kwh: float
    occupancy_state: OccupancyState = OccupancyState.ACTIVE

    # Rooftop Solar PV Attributes
    has_solar: bool = False
    solar_capacity_kw: float = 0.0

    # Sustained Efficiency Drop Attributes
    sustained_drop_ratio: float = 0.0   # Fraction reduced (e.g. 0.40 = 40% reduction)
    sustained_drop_start_hour: Optional[int] = None

    # Vacancy Timing Attributes
    vacant_start_hour: Optional[int] = None
    vacant_end_hour: Optional[int] = None
    standby_load_kwh: float = 0.02      # Smoke detectors, router, minimal standby

    def compute_true_demand(
        self,
        dt: datetime,
        hour_idx: int,
        pattern_engine: ConsumptionPatternEngine,
        rng: np.random.RandomState
    ) -> Tuple[float, str]:
        """
        Compute true physical electrical demand before any meter-level tampering.

        Returns
        -------
        Tuple[float, str]
            true_demand_kwh : float (true physical energy consumed)
            event_type : str (legitimate behavioral event, e.g. "NONE", "VACANCY", "SUSTAINED_DROP", "SOLAR_SELF_GEN")
        """
        # 1. Vacant Home State (Legitimate Zero / Standby)
        is_vacant_now = (
            self.occupancy_state == OccupancyState.VACANT
            or (
                self.vacant_start_hour is not None
                and self.vacant_end_hour is not None
                and self.vacant_start_hour <= hour_idx <= self.vacant_end_hour
            )
        )
        if is_vacant_now:
            # Standby load with slight micro-noise, occasionally zero if breaker turned off
            if rng.rand() < 0.25:
                # Main circuit switch off: strictly 0.0 kWh legitimate zero!
                return 0.0, "VACANCY"
            standby = max(0.0, self.standby_load_kwh + rng.normal(0, 0.005))
            return round(standby, 4), "VACANCY"

        # 2. Base Load Calculation
        effective_base = self.base_load_kwh

        # 3. Sustained Drop (Non-malicious efficiency improvement, HVAC upgrade, or occupant departure)
        is_sustained_drop = (
            self.occupancy_state == OccupancyState.SUSTAINED_DROP
            or (
                self.sustained_drop_start_hour is not None
                and hour_idx >= self.sustained_drop_start_hour
            )
        )
        if is_sustained_drop and self.sustained_drop_ratio > 0.0:
            effective_base *= (1.0 - self.sustained_drop_ratio)

        # 4. Hourly Demand via Pattern Engine
        solar_cap = self.solar_capacity_kw if self.has_solar else 0.0
        gross_consumption = pattern_engine.compute_hourly_consumption(
            dt=dt,
            consumer_type=self.consumer_type,
            base_load_kwh=effective_base,
            rng=rng,
            solar_capacity_kw=0.0 # Calculate gross first
        )

        event_name = "NONE"
        if is_sustained_drop:
            event_name = "SUSTAINED_DROP"

        # 5. Solar Offset Calculation
        if self.has_solar and solar_cap > 0.0:
            solar_gen = pattern_engine.get_solar_generation(dt, solar_cap, rng)
            if solar_gen > 0.0:
                net_demand = max(0.0, gross_consumption - solar_gen)
                if net_demand == 0.0:
                    event_name = "SOLAR_SELF_GEN"
                return round(net_demand, 4), event_name

        return round(gross_consumption, 4), event_name


def generate_consumer_profile(
    consumer_id: str,
    consumer_type: ConsumerType,
    rng: np.random.RandomState,
    is_vacant: bool = False,
    has_sustained_drop: bool = False,
    has_solar: bool = False,
    total_hours: int = 720
) -> ConsumerProfile:
    """
    Factory helper to generate realistic consumer profiles tailored to consumer archetypes.
    """
    # Baseline load distributions (kWh/hour average)
    if consumer_type == ConsumerType.RESIDENTIAL:
        base_load = float(rng.uniform(0.6, 1.8))
        standby = float(rng.uniform(0.01, 0.04))
    elif consumer_type == ConsumerType.COMMERCIAL:
        base_load = float(rng.uniform(8.0, 30.0))
        standby = float(rng.uniform(0.10, 0.40))
    elif consumer_type == ConsumerType.INDUSTRIAL:
        base_load = float(rng.uniform(40.0, 150.0))
        standby = float(rng.uniform(2.0, 8.0))
    else: # RURAL
        base_load = float(rng.uniform(1.2, 4.5))
        standby = float(rng.uniform(0.02, 0.06))

    vacant_start = None
    vacant_end = None
    occupancy = OccupancyState.ACTIVE
    if is_vacant:
        # Either permanently vacant (e.g. unleased or seasonal home) or vacant during a vacation interval
        if rng.rand() < 0.40:
            occupancy = OccupancyState.VACANT
        else:
            vacant_start = int(rng.randint(total_hours // 6, total_hours // 2))
            vacant_duration = int(rng.randint(72, 360)) # 3 to 15 days
            vacant_end = min(total_hours - 1, vacant_start + vacant_duration)

    drop_start = None
    drop_ratio = 0.0
    if has_sustained_drop:
        drop_start = int(rng.randint(total_hours // 4, total_hours * 3 // 4))
        drop_ratio = float(rng.uniform(0.30, 0.55)) # 30-55% drop

    solar_cap = 0.0
    if has_solar:
        if consumer_type == ConsumerType.RESIDENTIAL:
            solar_cap = float(rng.uniform(3.0, 7.0))
        elif consumer_type == ConsumerType.COMMERCIAL:
            solar_cap = float(rng.uniform(20.0, 60.0))
        elif consumer_type == ConsumerType.RURAL:
            solar_cap = float(rng.uniform(5.0, 15.0))

    return ConsumerProfile(
        consumer_id=consumer_id,
        consumer_type=consumer_type,
        base_load_kwh=round(base_load, 3),
        occupancy_state=occupancy,
        has_solar=has_solar,
        solar_capacity_kw=round(solar_cap, 2),
        sustained_drop_ratio=round(drop_ratio, 3),
        sustained_drop_start_hour=drop_start,
        vacant_start_hour=vacant_start,
        vacant_end_hour=vacant_end,
        standby_load_kwh=round(standby, 4)
    )
