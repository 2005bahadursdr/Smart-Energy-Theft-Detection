"""
MDMS Data Quality Tracker.
Monitors duplicate telemetry, missing intervals, late-arriving events,
and per-client data completion statistics.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Set, Tuple

from mdms_simulator.config import MDMSConfig
from mdms_simulator.models import DataQualityReport


class DataQualityTracker:
    """
    Maintains telemetry statistics, high watermarks, duplicate caches,
    and completion rates.
    """

    def __init__(self, config: Optional[MDMSConfig] = None):
        self.config = config or MDMSConfig()
        self.total_received: int = 0
        self.total_accepted: int = 0
        self.total_duplicates: int = 0
        self.total_missing_readings: int = 0
        self.total_late_arrivals: int = 0
        self.total_rejected: int = 0

        # Memory cache of observed (meter_id, timestamp) keys for fast deduplication
        self._seen_keys: Set[Tuple[str, str]] = set()

        # Watermarks (latest timestamp observed per client)
        self._client_watermarks: Dict[str, datetime] = {}

        # Per-client counters: client_id -> {"accepted": ..., "missing": ..., "duplicates": ..., "late": ...}
        self.client_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "received": 0,
            "accepted": 0,
            "duplicates": 0,
            "missing": 0,
            "late": 0,
            "rejected": 0,
        })

    def check_duplicate(self, meter_id: str, timestamp: str) -> bool:
        """Check if (meter_id, timestamp) pair has already been ingested."""
        key = (meter_id, timestamp)
        if key in self._seen_keys:
            return True
        return False

    def register_key(self, meter_id: str, timestamp: str) -> None:
        """Record (meter_id, timestamp) in deduplication set."""
        self._seen_keys.add((meter_id, timestamp))

    def evaluate_arrival_timing(self, client_id: str, dt: datetime) -> bool:
        """
        Evaluate if reading timestamp constitutes a late event.
        A reading is late if it arrives significantly behind the client's current watermark.
        """
        current_watermark = self._client_watermarks.get(client_id)
        if current_watermark is None:
            self._client_watermarks[client_id] = dt
            return False

        late_cutoff = current_watermark - timedelta(hours=self.config.late_arrival_threshold_hours)
        is_late = dt < late_cutoff

        # Update watermark if newer
        if dt > current_watermark:
            self._client_watermarks[client_id] = dt

        return is_late

    def record_event(
        self,
        client_id: str,
        is_accepted: bool,
        is_duplicate: bool,
        is_missing: bool,
        is_late: bool,
        is_rejected: bool
    ) -> None:
        """Update metrics following an ingestion evaluation."""
        self.total_received += 1
        stats = self.client_stats[client_id]
        stats["received"] += 1

        if is_rejected:
            self.total_rejected += 1
            stats["rejected"] += 1
            return

        if is_duplicate:
            self.total_duplicates += 1
            stats["duplicates"] += 1
            return

        if is_accepted:
            self.total_accepted += 1
            stats["accepted"] += 1

        if is_missing:
            self.total_missing_readings += 1
            stats["missing"] += 1

        if is_late:
            self.total_late_arrivals += 1
            stats["late"] += 1

    def generate_report(self) -> DataQualityReport:
        """Synthesize comprehensive data quality audit report."""
        total_valid = self.total_accepted
        completion_pct = round(
            (max(0, total_valid - self.total_missing_readings) / max(1, total_valid)) * 100, 2
        )

        return DataQualityReport(
            disclaimer=self.config.disclaimer,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_received=self.total_received,
            total_accepted=self.total_accepted,
            total_duplicates=self.total_duplicates,
            total_missing_readings=self.total_missing_readings,
            total_late_arrivals=self.total_late_arrivals,
            total_rejected=self.total_rejected,
            overall_completion_rate_pct=completion_pct,
            client_breakdown=dict(self.client_stats)
        )
