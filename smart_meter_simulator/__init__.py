"""
smart_meter_simulator module.
Blockchain-Based Byzantine-Resilient Federated Learning for Smart Grid Energy Theft Detection.

Exposes high-fidelity synthetic smart meter simulation components with separated
operational status, event types, and simulated ground-truth metrics.
"""

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
from smart_meter_simulator.simulator import (
    SmartMeterSimulator,
    build_parser,
    main,
)
from smart_meter_simulator.tamper_events import (
    TamperEvent,
    TamperInjector,
    TamperType,
)

__all__ = [
    "ConsumerType",
    "OccupancyState",
    "ConsumerProfile",
    "generate_consumer_profile",
    "ConsumptionPatternEngine",
    "DIURNAL_BASELINES",
    "WEEKDAY_MULTIPLIERS",
    "MeterStatus",
    "GridEventType",
    "SmartMeter",
    "TamperType",
    "TamperEvent",
    "TamperInjector",
    "SmartMeterSimulator",
    "build_parser",
    "main",
]
