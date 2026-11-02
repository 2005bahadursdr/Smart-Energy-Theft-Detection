"""
Historical Reputation Engine for Federated Multi-Round Trust Scoring.

Key Invariants:
1. "Reputation must not unfairly punish non-IID clients": Smart grid substations with
   unusual regional consumption patterns must not be permanently stigmatized. The engine
   includes minimum reputation floors, non-punitive priors, and forgiveness boosts.
2. Missing history tolerance: Handles new clients entering late and unselected clients
   without zeroing out or crashing.
3. Exponential Moving Average (EMA) smoothing prevents single-round statistical noise
   from dominating longitudinal assessment.
"""

import copy
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("trust_reputation")


class HistoricalReputationEngine:
    """
    Tracks and updates multi-round reputation trajectories for smart grid clients.
    """

    def __init__(
        self,
        initial_reputation: float = 0.70,
        decay_alpha: float = 0.80,
        unseen_retention: float = 0.95,
        min_reputation_floor: float = 0.10,
        forgiveness_boost: float = 0.05,
        history_window: int = 10,
    ):
        self.initial_reputation = float(np_clip(initial_reputation, 0.0, 1.0))
        self.decay_alpha = float(np_clip(decay_alpha, 0.0, 1.0))
        self.unseen_retention = float(np_clip(unseen_retention, 0.0, 1.0))
        self.min_reputation_floor = float(np_clip(min_reputation_floor, 0.0, 0.5))
        self.forgiveness_boost = float(forgiveness_boost)
        self.history_window = history_window

        # Map of client_id -> current reputation float in [0.0, 1.0]
        self._reputations: Dict[str, float] = {}
        # Map of client_id -> consecutive honest round count
        self._consecutive_honest: Dict[str, int] = {}
        # Audit history: list of round records
        self._round_history: List[Dict[str, Any]] = []

    def get_reputation(self, client_id: str) -> float:
        """
        Get current reputation for client. If unseen, returns initial_reputation.
        """
        return self._reputations.get(client_id, self.initial_reputation)

    def get_all_reputations(self) -> Dict[str, float]:
        """Return a copy of all tracked client reputations."""
        return {k: round(v, 4) for k, v in self._reputations.items()}

    def update_reputations(
        self,
        round_trust_scores: Dict[str, float],
        all_known_client_ids: Optional[List[str]] = None,
        server_round: int = 1,
    ) -> Dict[str, float]:
        """
        Update client reputations following completion of a federated round.

        Args:
            round_trust_scores: Dict of client_id -> round trust score T_round in [0, 1]
                               for clients that participated in the round.
            all_known_client_ids: Optional complete list of all registered clients.
            server_round: Current federated learning round number.

        Returns:
            Dict mapping each known client_id to updated reputation in [0.0, 1.0].
        """
        all_cids = set(self._reputations.keys())
        all_cids.update(round_trust_scores.keys())
        if all_known_client_ids:
            all_cids.update(all_known_client_ids)

        updated: Dict[str, float] = {}
        round_audit_entries = {}

        for cid in sorted(all_cids):
            prev_rep = self._reputations.get(cid, self.initial_reputation)

            if cid in round_trust_scores:
                t_round = float(np_clip(round_trust_scores[cid], 0.0, 1.0))
                # EMA Update: R_t = alpha * R_{t-1} + (1 - alpha) * T_round
                new_rep = self.decay_alpha * prev_rep + (1.0 - self.decay_alpha) * t_round

                # Forgiveness & Recovery for non-IID clients:
                # If round trust score is high (>= 0.70), increment consecutive honest counter
                if t_round >= 0.70:
                    cons = self._consecutive_honest.get(cid, 0) + 1
                    self._consecutive_honest[cid] = cons
                    # Apply forgiveness boost if recovering
                    if cons >= 2 and new_rep < 0.90:
                        new_rep = min(1.0, new_rep + self.forgiveness_boost)
                else:
                    self._consecutive_honest[cid] = 0

                # Enforce minimum reputation floor
                new_rep = max(self.min_reputation_floor, new_rep)
                new_rep = min(1.0, new_rep)

                round_audit_entries[cid] = {
                    "participated": True,
                    "round_score": round(t_round, 4),
                    "previous_reputation": round(prev_rep, 4),
                    "new_reputation": round(new_rep, 4),
                }
            else:
                # Client not participating in this round: passive retention
                new_rep = max(self.min_reputation_floor, prev_rep * self.unseen_retention)
                round_audit_entries[cid] = {
                    "participated": False,
                    "round_score": None,
                    "previous_reputation": round(prev_rep, 4),
                    "new_reputation": round(new_rep, 4),
                }

            self._reputations[cid] = new_rep
            updated[cid] = round(new_rep, 4)

        # Record round audit
        self._round_history.append({
            "round": server_round,
            "reputations": copy.deepcopy(updated),
            "details": round_audit_entries,
        })
        if len(self._round_history) > self.history_window:
            self._round_history.pop(0)

        return updated

    def get_history(self) -> List[Dict[str, Any]]:
        """Return history audit trail."""
        return copy.deepcopy(self._round_history)


def np_clip(val: float, low: float, high: float) -> float:
    """Helper to clip float value safely."""
    return max(low, min(high, float(val)))
