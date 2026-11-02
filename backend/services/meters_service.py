"""
Smart Meter Summary Service.

STRICT PRIVACY INVARIANT:
-------------------------
Never expose individual smart meter time-series readings or consumer PII
in global responses. Only privacy-preserving aggregates (statistical moments,
anomaly counts, and risk distributions) are computed and returned.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, Any

from backend.database.adapter import get_database_adapter
from backend.schemas import MetersSummaryResponse, ConsumptionMetricsAggregate

logger = logging.getLogger("backend.services.meters")


class MetersService:
    """Service for computing and delivering aggregated smart meter telemetry."""

    def __init__(self, db=None):
        self.db = db or get_database_adapter()

    def get_meters_summary(self) -> MetersSummaryResponse:
        """
        Fetch aggregated smart meter summary statistics.
        Guarantees that raw time-series consumption traces are completely redacted.
        """
        doc = self.db.find_one("meter_summaries")
        if not doc:
            # Fallback default statistics if not yet seeded
            doc = {
                "total_meters": 1000,
                "total_readings_sampled": 720000,
                "sampling_frequency_minutes": 15,
                "theft_ratio": 0.082,
                "consumption_metrics": {
                    "mean_kwh": 21.98,
                    "std_kwh": 12.45,
                    "min_kwh": 0.0,
                    "max_kwh": 335.94,
                    "median_kwh": 18.25,
                    "p25_kwh": 8.40,
                    "p75_kwh": 31.60,
                    "total_consumed_kwh": 15825600.0,
                },
                "status_breakdown": {
                    "NORMAL": 695193,
                    "OUTAGE": 9000,
                    "COMM_FAILURE": 7077,
                    "TAMPER_ALERT": 6575,
                    "DISCONNECTED": 2155,
                },
                "event_breakdown": {
                    "NONE": 640704,
                    "TAMPER_EVENT": 23825,
                    "SUSTAINED_DROP": 13696,
                    "VACANCY": 11932,
                    "SOLAR_SELF_GEN": 11611,
                    "GRID_OUTAGE": 9000,
                    "COMM_FAILURE": 7077,
                    "DISCONNECTION": 2155,
                },
                "consumer_profiles_breakdown": {
                    "Residential": 550,
                    "Commercial": 250,
                    "Industrial": 120,
                    "Agricultural": 80,
                },
                "hardware_tamper_alerts": 6575,
                "estimated_unreported_theft_kwh": 462381.27,
                "data_source_label": "SIMULATED_DEMO_DATA",
                "demo_notice": (
                    "Aggregated demonstration smart meter statistics (1000 meters, 30 days). "
                    "Individual meter time-series traces and consumer PII are strictly redacted."
                ),
            }

        # Enforce that no raw arrays or timeseries keys are present
        forbidden_keys = {"readings", "consumption_kwh", "raw_series", "time_series", "pii", "user_id"}
        for k in forbidden_keys:
            doc.pop(k, None)

        metrics = doc.get("consumption_metrics", {})
        agg_metrics = ConsumptionMetricsAggregate(
            mean_kwh=float(metrics.get("mean_kwh", 21.98)),
            std_kwh=float(metrics.get("std_kwh", 12.45)),
            min_kwh=float(metrics.get("min_kwh", 0.0)),
            max_kwh=float(metrics.get("max_kwh", 335.94)),
            median_kwh=float(metrics.get("median_kwh", 18.25)),
            p25_kwh=float(metrics.get("p25_kwh", 8.40)),
            p75_kwh=float(metrics.get("p75_kwh", 31.60)),
            total_consumed_kwh=float(metrics.get("total_consumed_kwh", 15825600.0)),
        )

        return MetersSummaryResponse(
            total_meters=int(doc.get("total_meters", 1000)),
            total_readings_sampled=int(doc.get("total_readings_sampled", 720000)),
            sampling_frequency_minutes=int(doc.get("sampling_frequency_minutes", 15)),
            theft_ratio=float(doc.get("theft_ratio", 0.082)),
            consumption_metrics=agg_metrics,
            status_breakdown=doc.get("status_breakdown", {}),
            event_breakdown=doc.get("event_breakdown", {}),
            consumer_profiles_breakdown=doc.get("consumer_profiles_breakdown", {}),
            hardware_tamper_alerts=int(doc.get("hardware_tamper_alerts", 6575)),
            estimated_unreported_theft_kwh=float(doc.get("estimated_unreported_theft_kwh", 462381.27)),
            data_source_label=doc.get("data_source_label", "SIMULATED_DEMO_DATA"),
            demo_notice=doc.get(
                "demo_notice",
                "Demonstration smart meter simulation data. Confidentiality strictly enforced.",
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
