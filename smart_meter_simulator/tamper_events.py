"""
Tamper Events Module.
Simulates physical, electromagnetic, and cyber smart meter tampering modalities.
Maintains strict separation between hardware tamper alerts, reported consumption,
and ground-truth synthetic labels.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np


class TamperType(str, Enum):
    """
    Physical and cyber/firmware tampering modalities.
    All instances generated are explicitly labeled as synthetic/simulated.
    """
    NONE = "NONE"
    PARTIAL_BYPASS = "PARTIAL_BYPASS"       # Physical jumper bypassing 30-70% current
    FULL_BYPASS = "FULL_BYPASS"             # Physical jumper bypassing >98% current
    REVERSE_CURRENT = "REVERSE_CURRENT"     # Wiring reversal or CT reversed
    MAGNETIC_TAMPER = "MAGNETIC_TAMPER"     # Neodymium magnet saturation
    CYBER_SCALING = "CYBER_SCALING"         # Firmware/calibration factor manipulation
    CYBER_FLATLINE = "CYBER_FLATLINE"       # Reporting frozen low standby constant
    PEAK_CLIPPING = "PEAK_CLIPPING"         # Capping peak-period consumption
    RANDOM_UNDER_REPORTING = "RANDOM_UNDER_REPORTING" # Stochastic reduction


@dataclass
class TamperEvent:
    """
    Specification of a synthetic tampering period applied to a pseudonymous meter.
    """
    tamper_type: TamperType
    start_hour: int
    end_hour: int
    severity: float = 0.50             # Attack attenuation factor or parameter
    hardware_alert_prob: float = 0.20  # Probability meter hardware sensor detects tamper
    flatline_value_kwh: float = 0.15   # For CYBER_FLATLINE attacks
    peak_cap_kwh: float = 1.00         # For PEAK_CLIPPING attacks

    def is_active(self, current_hour_idx: int) -> bool:
        """Check if tamper event is active at the given simulation hour index."""
        return self.start_hour <= current_hour_idx <= self.end_hour


class TamperInjector:
    """
    Transforms true energy demand into reported meter consumption under tampering,
    while calculating hardware alerts and simulated ground-truth metrics.
    """

    @staticmethod
    def apply_tamper(
        actual_kwh: float,
        tamper_event: Optional[TamperEvent],
        hour_idx: int,
        rng: np.random.RandomState
    ) -> Tuple[float, int, int, str, float]:
        """
        Apply synthetic tampering to consumption reading.

        Parameters
        ----------
        actual_kwh : float
            True physical energy consumed behind the meter.
        tamper_event : Optional[TamperEvent]
            Scheduled tamper event, if any.
        hour_idx : int
            Current simulation hour index.
        rng : np.random.RandomState
            Reproducible pseudorandom generator.

        Returns
        -------
        Tuple[float, int, int, str, float]
            reported_kwh : float (reading registered by the meter)
            hardware_tamper_flag : int (0 or 1, hardware sensor alert)
            simulated_theft_label : int (0 or 1, synthetic ground-truth label)
            simulated_theft_type : str (tamper modality name)
            simulated_unreported_kwh : float (stolen/unregistered energy: actual - reported)
        """
        # Case 1: No tamper scheduled or event inactive at this hour
        if tamper_event is None or not tamper_event.is_active(hour_idx):
            return (
                round(actual_kwh, 4),
                0,                          # hardware_tamper_flag = 0
                0,                          # simulated_theft_label = 0 (ground truth)
                TamperType.NONE.value,      # simulated_theft_type = "NONE"
                0.0                         # simulated_unreported_kwh = 0.0
            )

        t_type = tamper_event.tamper_type
        reported_kwh = actual_kwh
        alert_flag = 0

        # Physical & Cyber Tamper Transformations
        if t_type == TamperType.PARTIAL_BYPASS:
            # Bypass fraction: meter records (1 - severity) of real current
            bypass_ratio = min(0.95, max(0.10, tamper_event.severity + rng.normal(0, 0.02)))
            reported_kwh = actual_kwh * (1.0 - bypass_ratio)
            # Physical inspection / cover switch alert chance
            alert_flag = 1 if rng.rand() < tamper_event.hardware_alert_prob else 0

        elif t_type == TamperType.FULL_BYPASS:
            # Virtually all current bypasses the meter; only small residual leakage recorded
            leakage = rng.uniform(0.000, 0.015)
            reported_kwh = min(actual_kwh, leakage)
            alert_flag = 1 if rng.rand() < tamper_event.hardware_alert_prob else 0

        elif t_type == TamperType.REVERSE_CURRENT:
            # Current reversed: electronic meter clamps to 0 or records minimal reverse noise
            reported_kwh = 0.0
            # Modern smart meters usually flag reverse energy register
            alert_flag = 1 if rng.rand() < max(0.70, tamper_event.hardware_alert_prob) else 0

        elif t_type == TamperType.MAGNETIC_TAMPER:
            # CT core saturated by magnet: drops measurement by 60% to 90%
            attenuation = rng.uniform(0.60, 0.90)
            reported_kwh = actual_kwh * (1.0 - attenuation)
            # Modern smart meters contain Hall effect sensors; high detection rate
            alert_flag = 1 if rng.rand() < max(0.85, tamper_event.hardware_alert_prob) else 0

        elif t_type == TamperType.CYBER_SCALING:
            # Hacked calibration multiplier: steady under-reporting
            scale = min(0.90, max(0.15, 1.0 - tamper_event.severity))
            reported_kwh = actual_kwh * scale
            # Cyber attacks do not trigger mechanical/magnetic tamper sensors
            alert_flag = 1 if rng.rand() < 0.05 else 0

        elif t_type == TamperType.CYBER_FLATLINE:
            # Static constant reading emitted to disguise variable load
            reported_kwh = min(actual_kwh, tamper_event.flatline_value_kwh + rng.normal(0, 0.005))
            alert_flag = 1 if rng.rand() < 0.03 else 0

        elif t_type == TamperType.PEAK_CLIPPING:
            # Caps load at a threshold (e.g. 1.0 kWh) to avoid peak pricing tiers
            if actual_kwh > tamper_event.peak_cap_kwh:
                reported_kwh = tamper_event.peak_cap_kwh + rng.normal(0, 0.02)
            else:
                reported_kwh = actual_kwh
            alert_flag = 1 if rng.rand() < 0.02 else 0

        elif t_type == TamperType.RANDOM_UNDER_REPORTING:
            # Random reduction fraction between 25% and 65%
            reduction = rng.uniform(0.25, 0.65)
            reported_kwh = actual_kwh * (1.0 - reduction)
            alert_flag = 1 if rng.rand() < 0.05 else 0

        # Safety clamps
        reported_kwh = max(0.0, round(float(reported_kwh), 4))
        unreported_kwh = max(0.0, round(float(actual_kwh - reported_kwh), 4))

        return (
            reported_kwh,
            int(alert_flag),
            1,                    # simulated_theft_label = 1 (active synthetic tampering)
            t_type.value,         # simulated_theft_type
            unreported_kwh        # simulated_unreported_kwh
        )
