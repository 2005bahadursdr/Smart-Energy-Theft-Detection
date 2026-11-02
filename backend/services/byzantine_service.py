"""
Byzantine Attack & Anomaly Detection Service.
Serves detected and ground-truth Byzantine perturbation events across FL rounds,
and supports dynamic injection of adversarial attacks across the federated grid.
"""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List
import uuid

from backend.database.adapter import get_database_adapter
from backend.schemas import ByzantineAttackInjectRequest, ByzantineEventRecord, ByzantineEventsResponse

logger = logging.getLogger("backend.services.byzantine")


class ByzantineService:
    """Service tracking Byzantine perturbation events, dynamic threat injection, and defense mitigations."""

    def __init__(self, db=None):
        self.db = db or get_database_adapter()

    def get_byzantine_events(self) -> ByzantineEventsResponse:
        """Fetch detected and ground-truth Byzantine perturbation logs."""
        raw_events = self.db.find("byzantine_events", sort=[("round", 1)])
        events: List[ByzantineEventRecord] = []

        for e in raw_events:
            events.append(
                ByzantineEventRecord(
                    event_id=e.get("event_id", "byz-unknown"),
                    round=int(e.get("round", 1)),
                    client_id=e.get("client_id", "client_D"),
                    attack_type=e.get("attack_type", "sign_flipping"),
                    detected_by_trust=bool(e.get("detected_by_trust", True)),
                    anomaly_metric=e.get("anomaly_metric", "cosine_similarity_negative"),
                    perturbation_norm=float(e.get("perturbation_norm", 4.12)),
                    action_taken=e.get("action_taken", "QUARANTINED"),
                    ground_truth_label=e.get("ground_truth_label", "MALICIOUS_BYZANTINE"),
                    timestamp=e.get("timestamp", datetime.now(timezone.utc).isoformat()),
                )
            )

        detected_count = sum(1 for ev in events if ev.detected_by_trust)
        mitigation_counts: Dict[str, int] = {}
        for ev in events:
            mitigation_counts[ev.action_taken] = mitigation_counts.get(ev.action_taken, 0) + 1

        return ByzantineEventsResponse(
            events=events,
            total_events=len(events),
            detected_events=detected_count,
            mitigation_summary=mitigation_counts,
            data_source_label="SIMULATED_DEMO_DATA",
            disclaimer=(
                "RESEARCH NOTICE: An unusual, high-norm, or divergent gradient update is NOT definitive proof "
                "of malicious Byzantine intent. Extreme non-IID load heterogeneity in smart grids naturally "
                "creates legitimate statistical outliers. The trust engine dampens outliers while preserving non-IID variance."
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def inject_attack(self, request: ByzantineAttackInjectRequest) -> Dict[str, Any]:
        """
        Dynamically injects an adversarial attack into the specified substation node.
        Updates trust_scores, fl_rounds, byzantine_events, and blockchain_audit system-wide.
        """
        all_clients = ["client_A", "client_B", "client_C", "client_D", "client_E"]
        target = request.client_id
        attack = request.attack_type
        mag = float(request.magnitude)
        is_attack_active = attack != "none"

        logger.info(f"Injecting Byzantine attack '{attack}' on '{target}' (magnitude={mag}) system-wide...")

        # 1. Update Trust Scores collection
        self.db.delete_many("trust_scores")
        scores = []
        for cid in all_clients:
            if cid == target and is_attack_active:
                cos_sim = -0.984 if attack == "sign_flipping" else (0.045 if attack == "gaussian_poisoning" else (0.720 if attack == "scaling_poison" else 0.00))
                dist = round(4.25 * (mag / 3.0), 2)
                val_delta = round(0.450 * (mag / 3.0), 3)
                scores.append({
                    "client_id": cid,
                    "round": 2,
                    "similarity_score": cos_sim,
                    "distance_metric": dist,
                    "validation_delta": val_delta,
                    "reputation_score": 0.20,
                    "composite_trust_score": 0.000,
                    "normalized_weight": 0.000,
                    "is_quarantined": True,
                    "status": "QUARANTINED",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            else:
                sim = 0.985 if cid == "client_A" else (0.972 if cid == "client_B" else (0.991 if cid == "client_C" else (0.935 if cid == "client_E" else 0.940)))
                weight = 0.25 if is_attack_active else 0.20
                scores.append({
                    "client_id": cid,
                    "round": 2,
                    "similarity_score": sim,
                    "distance_metric": 0.12,
                    "validation_delta": 0.002,
                    "reputation_score": 0.88,
                    "composite_trust_score": 0.942,
                    "normalized_weight": weight,
                    "is_quarantined": False,
                    "status": "TRUSTED",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        self.db.insert_many("trust_scores", scores)

        # 2. Update FL Rounds collection
        quarantined = [target] if is_attack_active else []
        participating = [c for c in all_clients if c not in quarantined]

        rounds = self.db.find("fl_rounds")
        if rounds:
            for r in rounds:
                self.db.update_one(
                    "fl_rounds",
                    {"round_number": r.get("round_number")},
                    {"$set": {
                        "quarantined_clients": quarantined,
                        "participating_clients": participating,
                    }}
                )

        # 2.5 Update Substation Clients collection status
        for cid in all_clients:
            if is_attack_active and cid == target:
                self.db.update_one(
                    "clients",
                    {"client_id": cid},
                    {"$set": {
                        "status": "QUARANTINED",
                        "current_trust_score": 0.08,
                        "current_reputation": 0.20,
                    }}
                )
            else:
                self.db.update_one(
                    "clients",
                    {"client_id": cid},
                    {"$set": {
                        "status": "ACTIVE",
                        "current_trust_score": 0.94,
                        "current_reputation": 0.88,
                    }}
                )

        # 3. Prepend into Byzantine Events collection
        if is_attack_active:
            new_evt = {
                "event_id": f"byz-{target}-{int(time.time())}",
                "round": 2,
                "client_id": target,
                "attack_type": attack,
                "detected_by_trust": True,
                "anomaly_metric": f"cosine_similarity_negative (-0.984)" if attack == "sign_flipping" else f"{attack}_anomaly",
                "perturbation_norm": round(4.12 * (mag / 3.0), 2),
                "action_taken": "QUARANTINED",
                "ground_truth_label": "MALICIOUS_BYZANTINE",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.db.insert_one("byzantine_events", new_evt)

        # 4. On-Chain Blockchain Audit Invariant:
        # Immutable on-chain audit records are committed exclusively by the FL coordinator
        # and smart contract transactions. Dummy records with random UUIDs are never inserted.

        return {
            "status": "SUCCESS",
            "quarantined_client": target if is_attack_active else None,
            "attack_type": attack,
            "quarantined_clients": quarantined,
            "participating_clients": participating,
            "message": f"Successfully injected {attack} attack on {target}. Quarantine applied system-wide.",
        }

    def reset_attacks(self) -> Dict[str, Any]:
        """Reset default state with client_D as benchmark adversary."""
        req = ByzantineAttackInjectRequest(client_id="client_D", attack_type="sign_flipping", magnitude=3.5)
        return self.inject_attack(req)
