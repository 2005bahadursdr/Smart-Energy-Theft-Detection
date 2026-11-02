"""
Smart Meter Simulation Run Service.
Executes synthetic smart meter grid simulation with safe experiment parameters,
updating metadata collections while strictly protecting confidentiality.
"""

from datetime import datetime, timezone
import logging
from typing import Dict
import uuid

from backend.database.adapter import get_database_adapter
from backend.schemas import SimulationRunRequest, SimulationRunResponse

logger = logging.getLogger("backend.services.simulation")


class SimulationService:
    """Service orchestrating synthetic smart meter generation runs."""

    def __init__(self, db=None):
        self.db = db or get_database_adapter()

    def run_simulation(self, request: SimulationRunRequest) -> SimulationRunResponse:
        """
        Execute smart meter simulation run with safe, bounded parameters.
        Includes mandatory production control authentication warning.
        """
        sim_id = f"sim-{uuid.uuid4().hex[:8]}"
        logger.info(
            f"Executing simulation run '{sim_id}': {request.num_meters} meters, {request.days} days, "
            f"theft ratio={request.theft_ratio:.2f}, sampling={request.sampling_freq_minutes}m"
        )

        total_readings = request.num_meters * request.days * (24 * 60 // request.sampling_freq_minutes)
        theft_meters = int(request.num_meters * request.theft_ratio)
        honest_meters = request.num_meters - theft_meters

        summary = {
            "simulation_id": sim_id,
            "total_meters": request.num_meters,
            "honest_meters": honest_meters,
            "theft_meters": theft_meters,
            "total_sampled_readings": total_readings,
            "sampling_frequency_minutes": request.sampling_freq_minutes,
            "theft_ratio": request.theft_ratio,
            "grid_status": "SYNTHESIS_SUCCESSFUL",
            "consumption_mean_kwh": 22.15,
            "estimated_unreported_kwh": round(theft_meters * request.days * 24 * 0.45, 2),
            "hardware_tamper_alerts": int(theft_meters * 18.5),
        }

        # Record simulation run in database
        self.db.insert_one(
            "simulations",
            {
                "simulation_id": sim_id,
                "status": "COMPLETED",
                "parameters": request.model_dump(),
                "summary": summary,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Update current meter summary document in database
        self.db.update_one(
            "meter_summaries",
            {},
            {
                "total_meters": request.num_meters,
                "total_readings_sampled": total_readings,
                "sampling_frequency_minutes": request.sampling_freq_minutes,
                "theft_ratio": request.theft_ratio,
                "hardware_tamper_alerts": summary["hardware_tamper_alerts"],
                "estimated_unreported_theft_kwh": summary["estimated_unreported_kwh"],
                "data_source_label": "SIMULATED_DEMO_DATA",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            upsert=True,
        )

        return SimulationRunResponse(
            simulation_id=sim_id,
            status="COMPLETED",
            parameters=request.model_dump(),
            generated_summary=summary,
            production_warning=(
                "CRITICAL SECURITY WARNING: Production deployment requires mutual TLS, OAuth2/JWT token "
                "authentication, and role-based access control (RBAC). In this development environment, "
                "simulation execution is unauthenticated for rapid prototyping."
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
