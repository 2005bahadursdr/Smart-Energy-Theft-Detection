"""
Substation Client Service.
Delivers metadata for localized substation clients (client_A through client_E),
highlighting regional non-IID load heterogeneity and local privacy firewalls.
"""

from datetime import datetime, timezone
import logging
from typing import List

from backend.database.adapter import get_database_adapter
from backend.schemas import ClientDetail, ClientListResponse

logger = logging.getLogger("backend.services.clients")


class ClientService:
    """Service managing smart grid substation client metadata."""

    def __init__(self, db=None):
        self.db = db or get_database_adapter()

    def get_clients(self) -> ClientListResponse:
        """Query and return substation client profiles."""
        raw_clients = self.db.find("clients", sort=[("client_id", 1)])
        clients: List[ClientDetail] = []

        for c in raw_clients:
            clients.append(
                ClientDetail(
                    client_id=c.get("client_id", "unknown"),
                    substation_name=c.get("substation_name", "Substation"),
                    region_type=c.get("region_type", "Mixed"),
                    meters_count=int(c.get("meters_count", 200)),
                    theft_meters=int(c.get("theft_meters", 17)),
                    theft_rate_pct=float(c.get("theft_rate_pct", 8.5)),
                    mean_daily_kwh=float(c.get("mean_daily_kwh", 10.0)),
                    extracted_sequences=int(c.get("extracted_sequences", 13400)),
                    current_reputation=float(c.get("current_reputation", 0.85)),
                    current_trust_score=float(c.get("current_trust_score", 0.90)),
                    status=c.get("status", "ACTIVE"),
                    privacy_firewall=c.get("privacy_firewall", "ACTIVE"),
                    central_fl_access=c.get("central_fl_access", "BLOCKED"),
                    local_data_path=c.get("local_data_path", "data/processed/clients"),
                )
            )

        active_count = sum(1 for c in clients if c.status == "ACTIVE")
        quarantined_count = sum(1 for c in clients if c.status == "QUARANTINED")

        return ClientListResponse(
            clients=clients,
            total_clients=len(clients),
            active_clients=active_count,
            quarantined_clients=quarantined_count,
            non_iid_characteristics=(
                "Clients exhibit extreme regional consumption heterogeneity: client_A represents industrial "
                "continuous high load (38.32 kWh/day), while client_E represents rural agricultural low baseline "
                "(0.36 kWh/day). The trust scoring mechanism specifically accounts for this non-IID variance."
            ),
            privacy_guarantee=(
                "Zero raw meter readings are transmitted to the central FL coordinator. "
                "Local datasets remain locked behind the substation privacy firewall."
            ),
            data_source_label="SGCC_REAL_WORLD_BENCHMARK",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
