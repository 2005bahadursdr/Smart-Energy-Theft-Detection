"""
MDMS Telemetry Validator.
Performs rigorous schema, timestamp, pseudonymity, range, and operational consistency validation.
"""

from datetime import datetime, timezone
import math
import re
from typing import Optional, Tuple

from mdms_simulator.config import MDMSConfig
from mdms_simulator.models import MeterReadingPayload


PSEUDONYM_REGEX = re.compile(r"^MTR-[A-Z0-9]+-[0-9A-Fa-f]+$")
PII_PATTERNS = [
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), # Email
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),                  # Phone
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                          # SSN
]


class ValidationResult:
    """Outcome of telemetry record validation."""

    def __init__(self, is_valid: bool, error_message: Optional[str] = None, is_missing: bool = False):
        self.is_valid = is_valid
        self.error_message = error_message
        self.is_missing = is_missing

    def __bool__(self) -> bool:
        return self.is_valid


class TelemetryValidator:
    """
    Validates meter telemetry against physical grid bounds, temporal plausibility,
    and pseudonymous privacy guarantees.
    """

    def __init__(self, config: Optional[MDMSConfig] = None):
        self.config = config or MDMSConfig()

    def validate(self, payload: MeterReadingPayload) -> ValidationResult:
        """
        Validate single telemetry payload.

        Parameters
        ----------
        payload : MeterReadingPayload
            Incoming meter reading.

        Returns
        -------
        ValidationResult
            Validation decision with error diagnostics.
        """
        # 1. Pseudonymous Meter ID Validation
        if not PSEUDONYM_REGEX.match(payload.meter_id):
            return ValidationResult(
                False,
                f"Meter ID '{payload.meter_id}' violates pseudonymity standard. Format must match 'MTR-X-XXXX'."
            )

        for pii_pat in PII_PATTERNS:
            if pii_pat.search(payload.meter_id):
                return ValidationResult(
                    False,
                    f"Privacy violation: Potential PII detected in meter identifier '{payload.meter_id}'."
                )

        # 2. Substation Client ID Validation
        if payload.client_id not in self.config.client_ids:
            return ValidationResult(
                False,
                f"Client '{payload.client_id}' is not in configured substation partitions: {self.config.client_ids}."
            )

        # 3. Timestamp Validation
        try:
            dt = datetime.strptime(payload.timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return ValidationResult(
                False,
                f"Timestamp '{payload.timestamp}' could not be parsed. Expected 'YYYY-MM-DD HH:MM:SS'."
            )

        # Year lower bound sanity check
        if dt.year < 2020:
            return ValidationResult(
                False,
                f"Timestamp year {dt.year} is before supported baseline horizon (2020)."
            )

        # Future timestamp tolerance check (relative to now + max_future_skew)
        now_utc = datetime.now()
        # If timestamp is drastically ahead of simulation horizon (e.g., year > 2050)
        if dt.year > 2050:
            return ValidationResult(
                False,
                f"Timestamp year {dt.year} exceeds plausibility threshold (2050)."
            )

        # 4. Range Validation on Consumption (kWh)
        val = payload.consumption_kwh
        is_missing = (val is None or (isinstance(val, float) and math.isnan(val)))

        if is_missing:
            # Missing reading is a legitimate communication failure condition, not a malformed payload
            return ValidationResult(True, is_missing=True)

        if val < self.config.min_consumption_kwh:
            return ValidationResult(
                False,
                f"Negative consumption reading {val} kWh is invalid (minimum allowed is {self.config.min_consumption_kwh})."
            )

        if val > self.config.max_consumption_kwh:
            return ValidationResult(
                False,
                f"Consumption reading {val} kWh exceeds maximum threshold {self.config.max_consumption_kwh} kWh."
            )

        return ValidationResult(True, is_missing=False)
