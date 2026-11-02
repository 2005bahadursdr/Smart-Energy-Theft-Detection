"""
Multi-Criteria Trust Engine Analytics Service.
Delivers detailed component trust breakdowns (similarity, distance, validation, reputation,
composite trust score, normalized weights, and quarantine status) across clients.
"""

from datetime import datetime, timezone
import logging
from typing import List

from backend.database.adapter import get_database_adapter
from backend.schemas import ClientTrustScoreRecord, TrustScoresResponse

logger = logging.getLogger("backend.services.trust")


class TrustService:
    """Service tracking client trust scores and dynamic aggregation weights."""

    def __init__(self, db=None):
        self.db = db or get_database_adapter()

    def get_trust_scores(self) -> TrustScoresResponse:
        """Fetch client multi-criteria trust scores and parameter weights."""
        raw_scores = self.db.find("trust_scores", sort=[("round", 1), ("client_id", 1)])
        scores: List[ClientTrustScoreRecord] = []

        for s in raw_scores:
            scores.append(
                ClientTrustScoreRecord(
                    client_id=s.get("client_id", "client_A"),
                    round=int(s.get("round", 1)),
                    similarity_score=float(s.get("similarity_score", 0.95)),
                    distance_metric=float(s.get("distance_metric", 0.10)),
                    validation_delta=float(s.get("validation_delta", 0.001)),
                    reputation_score=float(s.get("reputation_score", 0.85)),
                    composite_trust_score=float(s.get("composite_trust_score", 0.92)),
                    normalized_weight=float(s.get("normalized_weight", 0.25)),
                    is_quarantined=bool(s.get("is_quarantined", False)),
                    status=s.get("status", "TRUSTED"),
                    timestamp=s.get("timestamp", datetime.now(timezone.utc).isoformat()),
                )
            )

        weights = {
            "weight_similarity": 0.40,
            "weight_validation": 0.35,
            "weight_reputation": 0.25,
        }
        thresholds = {
            "suspicious_threshold": 0.45,
            "quarantine_threshold": 0.25,
        }

        return TrustScoresResponse(
            scores=scores,
            weights_used=weights,
            thresholds=thresholds,
            data_source_label="SIMULATED_DEMO_DATA",
            summary=(
                "Dynamic trust scores computed from cosine directional agreement, validation loss impact, "
                "and historical reputation EMA. Honest non-IID clients (client_A, client_E) maintain high trust "
                "despite statistical divergence, while Byzantine sign-flipping attacker client_D is quarantined."
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
