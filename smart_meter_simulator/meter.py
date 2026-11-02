"""
Smart Meter Device Module.
Represents an individual physical/edge smart meter with pseudonymous identity,
telemetry status, hardware sensor alarms, and ground-truth synthetic tracking.
Strictly isolates meter telemetry, operational events, and simulated labels.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import numpy as np

from smart_meter_simulator.consumer import ConsumerProfile
from smart_meter_simulator.consumption_patterns import ConsumptionPatternEngine
from smart_meter_simulator.tamper_events import TamperEvent, TamperInjector, TamperType


class MeterStatus(str, Enum):
    """
    Physical operational status emitted by smart meter firmware.
    """
    NORMAL = "NORMAL"                   # Nominal grid connection and functioning
    OUTAGE = "OUTAGE"                   # De-energized grid line / blackout
    DISCONNECTED = "DISCONNECTED"       # Remote disconnect switch opened by utility
    COMM_FAILURE = "COMM_FAILURE"       # AMI cellular / RF mesh communication drop
    TAMPER_ALERT = "TAMPER_ALERT"       # Hardware sensor triggered alarm


class GridEventType(str, Enum):
    """
    Categorical grid and environmental events.
    Kept separate from synthetic theft labels to avoid conflating legitimate anomalies with theft.
    """
    NONE = "NONE"                       # Nominal standard operation
    GRID_OUTAGE = "GRID_OUTAGE"         # Feeder / substation blackout
    DISCONNECTION = "DISCONNECTION"     # Utility service disconnect
    VACANCY = "VACANCY"                 # Legitimate zero or micro-standby from unoccupied premises
    SOLAR_SELF_GEN = "SOLAR_SELF_GEN"   # Zero net import due to rooftop solar generation
    SUSTAINED_DROP = "SUSTAINED_DROP"   # Non-malicious efficiency retrofit / occupancy reduction
    COMM_FAILURE = "COMM_FAILURE"       # Missing reading / packet drop
    TAMPER_EVENT = "TAMPER_EVENT"       # Active synthetic tampering incident


@dataclass
class SmartMeter:
    """
    Smart Meter Device model with pseudonymous identifier.
    """
    meter_id: str
    client_id: str                      # Substation / feeder assignment (e.g. client_A)
    consumer: ConsumerProfile
    tamper_event: Optional[TamperEvent] = None

    # Utility Disconnection Schedule
    is_disconnected: bool = False
    disconnect_start_hour: Optional[int] = None
    disconnect_end_hour: Optional[int] = None

    def read_hour(
        self,
        dt: datetime,
        hour_idx: int,
        is_grid_outage: bool,
        is_comm_drop: bool,
        pattern_engine: ConsumptionPatternEngine,
        rng: np.random.RandomState
    ) -> Dict[str, Any]:
        """
        Record and synthesize single hourly telemetry reading.

        Enforces fundamental rule: Zero consumption is NOT equated with theft.
        Outages, disconnections, vacancies, and solar generation remain legitimate (simulated_theft_label = 0).

        Returns
        -------
        Dict[str, Any]
            Structured reading record with separated telemetry, status, event, and simulated truth.
        """
        iso_timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")

        # 1. SCENARIO: Grid Outage (Blackout on feeder)
        if is_grid_outage:
            return {
                "timestamp": iso_timestamp,
                "meter_id": self.meter_id,
                "client_id": self.client_id,
                "consumption_kwh": 0.0,
                "meter_status": MeterStatus.OUTAGE.value,
                "event_type": GridEventType.GRID_OUTAGE.value,
                "hardware_tamper_flag": 0,
                # Ground-truth simulation labels (strictly separate)
                "simulated_theft_label": 0,
                "simulated_theft_type": TamperType.NONE.value,
                "simulated_actual_kwh": 0.0,
                "simulated_unreported_kwh": 0.0
            }

        # 2. SCENARIO: Utility Disconnection (Service shutoff)
        is_disc_now = self.is_disconnected or (
            self.disconnect_start_hour is not None
            and self.disconnect_end_hour is not None
            and self.disconnect_start_hour <= hour_idx <= self.disconnect_end_hour
        )
        if is_disc_now:
            return {
                "timestamp": iso_timestamp,
                "meter_id": self.meter_id,
                "client_id": self.client_id,
                "consumption_kwh": 0.0,
                "meter_status": MeterStatus.DISCONNECTED.value,
                "event_type": GridEventType.DISCONNECTION.value,
                "hardware_tamper_flag": 0,
                # Ground-truth simulation labels (strictly separate)
                "simulated_theft_label": 0,
                "simulated_theft_type": TamperType.NONE.value,
                "simulated_actual_kwh": 0.0,
                "simulated_unreported_kwh": 0.0
            }

        # 3. SCENARIO: Communication Failure (Telemetry packet lost in transit)
        if is_comm_drop:
            # Meter hardware may be normal, but telemetry record is missing (represented as NaN)
            return {
                "timestamp": iso_timestamp,
                "meter_id": self.meter_id,
                "client_id": self.client_id,
                "consumption_kwh": np.nan,
                "meter_status": MeterStatus.COMM_FAILURE.value,
                "event_type": GridEventType.COMM_FAILURE.value,
                "hardware_tamper_flag": 0,
                # Ground-truth simulation labels (strictly separate: missing reading is not theft)
                "simulated_theft_label": 0,
                "simulated_theft_type": TamperType.NONE.value,
                "simulated_actual_kwh": 0.0,
                "simulated_unreported_kwh": 0.0
            }

        # 4. SCENARIO: Physical Energy Consumption Behind Meter
        true_demand, consumer_event = self.consumer.compute_true_demand(
            dt=dt,
            hour_idx=hour_idx,
            pattern_engine=pattern_engine,
            rng=rng
        )

        # 5. SCENARIO: Check for Active Synthetic Tamper
        is_tamper_active = (
            self.tamper_event is not None
            and self.tamper_event.is_active(hour_idx)
        )

        if is_tamper_active:
            # Apply simulated physical / cyber tamper
            (
                reported_kwh,
                alert_flag,
                theft_label,
                theft_type,
                unreported_kwh
            ) = TamperInjector.apply_tamper(
                actual_kwh=true_demand,
                tamper_event=self.tamper_event,
                hour_idx=hour_idx,
                rng=rng
            )

            meter_status = MeterStatus.TAMPER_ALERT.value if alert_flag else MeterStatus.NORMAL.value
            event_type = GridEventType.TAMPER_EVENT.value

            return {
                "timestamp": iso_timestamp,
                "meter_id": self.meter_id,
                "client_id": self.client_id,
                "consumption_kwh": reported_kwh,
                "meter_status": meter_status,
                "event_type": event_type,
                "hardware_tamper_flag": alert_flag,
                # Ground-truth simulation labels
                "simulated_theft_label": theft_label,
                "simulated_theft_type": theft_type,
                "simulated_actual_kwh": true_demand,
                "simulated_unreported_kwh": unreported_kwh
            }

        # 6. SCENARIO: Legitimate Normal / Vacant / Solar / Sustained Drop
        event_type_str = GridEventType.NONE.value
        if consumer_event == "VACANCY":
            event_type_str = GridEventType.VACANCY.value
        elif consumer_event == "SOLAR_SELF_GEN":
            event_type_str = GridEventType.SOLAR_SELF_GEN.value
        elif consumer_event == "SUSTAINED_DROP":
            event_type_str = GridEventType.SUSTAINED_DROP.value

        return {
            "timestamp": iso_timestamp,
            "meter_id": self.meter_id,
            "client_id": self.client_id,
            "consumption_kwh": true_demand,
            "meter_status": MeterStatus.NORMAL.value,
            "event_type": event_type_str,
            "hardware_tamper_flag": 0,
            # Ground-truth simulation labels (strictly legitimate)
            "simulated_theft_label": 0,
            "simulated_theft_type": TamperType.NONE.value,
            "simulated_actual_kwh": true_demand,
            "simulated_unreported_kwh": 0.0
        }
