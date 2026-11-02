"""
Composite Multi-Criteria Trust Engine for Byzantine-Resilient Federated Learning.

Combines:
1. Cosine Directional Agreement (S_sim)
2. Euclidean Distance / Norm Anomaly (S_dist)
3. Server Benchmark Validation Delta (S_val)
4. Historical Multi-Round Reputation (R_prev)

Invariants:
- Factor weights strictly normalize to 1.0.
- Aggregation weights are strictly non-negative and sum to 1.0.
- Safe fallbacks for zero-weight and all-quarantined edge cases.
- Ground-truth labels are NEVER accessed during scoring; they may only be passed
  optionally to post-hoc diagnostic functions.
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import yaml

from trust.reputation.engine import HistoricalReputationEngine
from trust.similarity.cosine_distance import (
    compute_parameter_delta,
    evaluate_similarity_and_distance,
)
from trust.validation.evaluator import ClientUpdateValidationEvaluator

logger = logging.getLogger("trust_engine")


@dataclass
class ClientTrustEvaluation:
    """Detailed trust evaluation for a single client update."""
    client_id: str
    num_samples: int
    similarity_score: float
    distance_score: float
    validation_score: float
    reputation_score: float
    composite_trust_score: float
    is_suspicious: bool
    raw_cosine: float
    euclidean_distance: float
    update_norm: float
    aggregation_weight: float
    quarantined: bool


class TrustEngine:
    """
    Multi-criteria trust scoring and weight allocation engine.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        validation_evaluator: Optional[ClientUpdateValidationEvaluator] = None,
        reputation_engine: Optional[HistoricalReputationEngine] = None,
    ):
        # 1. Load configuration
        self.config = self._load_config(config_path, config_dict)
        trust_cfg = self.config.get("trust_system", {})

        # 2. Extract and normalize factor weights
        raw_weights = trust_cfg.get("weights", {})
        w_sim = float(raw_weights.get("similarity", 0.30))
        w_dist = float(raw_weights.get("update_distance", 0.20))
        w_val = float(raw_weights.get("validation", 0.25))
        w_rep = float(raw_weights.get("reputation", 0.25))

        total_w = w_sim + w_dist + w_val + w_rep
        if total_w <= 0.0:
            total_w = 1.0
            w_sim, w_dist, w_val, w_rep = 0.25, 0.25, 0.25, 0.25

        self.w_sim = w_sim / total_w
        self.w_dist = w_dist / total_w
        self.w_val = w_val / total_w
        self.w_rep = w_rep / total_w

        # 3. Parameters
        agg_cfg = trust_cfg.get("aggregation", {})
        self.suspicious_threshold = float(agg_cfg.get("suspicious_threshold", 0.40))
        self.policy = str(agg_cfg.get("policy", "quarantine"))
        self.penalty_exponent = float(agg_cfg.get("penalty_exponent", 2.0))
        self.zero_weight_fallback = str(agg_cfg.get("zero_weight_fallback", "coordinate_median"))

        # Distance scaling
        dist_cfg = trust_cfg.get("distance", {})
        self.distance_scale = float(dist_cfg.get("distance_scale", 2.0))

        # Subsystems
        self.validation_evaluator = validation_evaluator
        rep_cfg = trust_cfg.get("reputation", {})
        self.reputation_engine = reputation_engine or HistoricalReputationEngine(
            initial_reputation=float(rep_cfg.get("initial_reputation", 0.70)),
            decay_alpha=float(rep_cfg.get("decay_alpha", 0.80)),
            unseen_retention=float(rep_cfg.get("unseen_client_retention", 0.95)),
            min_reputation_floor=float(rep_cfg.get("min_reputation_floor", 0.10)),
            forgiveness_boost=float(rep_cfg.get("forgiveness_boost", 0.05)),
            history_window=int(rep_cfg.get("history_window", 10)),
        )

        logger.info(
            f"Initialized TrustEngine | Weights -> Sim: {self.w_sim:.2f}, Dist: {self.w_dist:.2f}, "
            f"Val: {self.w_val:.2f}, Rep: {self.w_rep:.2f} | "
            f"Threshold: {self.suspicious_threshold:.2f} | Policy: {self.policy}"
        )

    def _load_config(
        self,
        config_path: Optional[Union[str, Path]],
        config_dict: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Load configuration from dictionary or YAML file."""
        if config_dict is not None:
            return config_dict

        p = Path(config_path) if config_path else Path("config/trust_config.yaml")
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Could not load trust config from {p}: {e}")
        return {}

    def evaluate_and_weight_updates(
        self,
        current_global_parameters: List[np.ndarray],
        client_updates: Dict[str, List[np.ndarray]],
        client_sample_counts: Dict[str, int],
        server_round: int = 1,
    ) -> Tuple[Dict[str, float], Dict[str, ClientTrustEvaluation], bool]:
        """
        Evaluate client updates, score trust, and compute normalized aggregation weights.

        Args:
            current_global_parameters: Current global model parameters (numpy arrays).
            client_updates: Dict of client_id -> client's local parameters w_local.
            client_sample_counts: Dict of client_id -> number of training examples n_i.
            server_round: Current round number.

        Returns:
            Tuple of:
            - aggregation_weights: Dict[client_id, float] normalized non-negative weights summing to 1.0.
            - evaluations: Dict[client_id, ClientTrustEvaluation] with detailed metric breakdowns.
            - fallback_triggered: bool indicating if zero-weight fallback was triggered.
        """
        if not client_updates:
            return {}, {}, False

        client_ids = list(client_updates.keys())

        # 1. Compute delta updates: Delta w_i = w_local - w_global
        client_deltas = {
            cid: compute_parameter_delta(client_updates[cid], current_global_parameters)
            for cid in client_ids
        }

        # 2. Similarity and distance metrics
        sim_metrics = evaluate_similarity_and_distance(
            client_deltas=client_deltas,
            distance_scale=self.distance_scale,
        )

        # 3. Validation evaluation on server benchmark data
        if self.validation_evaluator is not None:
            val_metrics = self.validation_evaluator.evaluate_client_updates(
                current_global_parameters=current_global_parameters,
                client_updates=client_updates,
            )
        else:
            val_metrics = {
                cid: {"validation_score": 0.70, "loss_delta": 0.0, "candidate_loss": 0.0}
                for cid in client_ids
            }

        # 4. Reputation retrieval and composite score calculation
        evaluations: Dict[str, ClientTrustEvaluation] = {}
        round_trust_scores: Dict[str, float] = {}

        for cid in client_ids:
            s_sim = sim_metrics[cid]["similarity_score"]
            s_dist = sim_metrics[cid]["distance_score"]
            s_val = val_metrics[cid]["validation_score"]
            r_prev = self.reputation_engine.get_reputation(cid)

            # Composite Score: weighted sum of factors
            composite = (
                self.w_sim * s_sim
                + self.w_dist * s_dist
                + self.w_val * s_val
                + self.w_rep * r_prev
            )
            composite = float(np.clip(composite, 0.0, 1.0))
            round_trust_scores[cid] = composite

            is_suspicious = bool(composite < self.suspicious_threshold)

            evaluations[cid] = ClientTrustEvaluation(
                client_id=cid,
                num_samples=client_sample_counts.get(cid, 1),
                similarity_score=round(s_sim, 4),
                distance_score=round(s_dist, 4),
                validation_score=round(s_val, 4),
                reputation_score=round(r_prev, 4),
                composite_trust_score=round(composite, 4),
                is_suspicious=is_suspicious,
                raw_cosine=sim_metrics[cid]["raw_cosine"],
                euclidean_distance=sim_metrics[cid]["euclidean_distance"],
                update_norm=sim_metrics[cid]["norm"],
                aggregation_weight=0.0, # Populated below
                quarantined=False,
            )

        # 5. Update historical reputations for next rounds
        self.reputation_engine.update_reputations(
            round_trust_scores=round_trust_scores,
            all_known_client_ids=client_ids,
            server_round=server_round,
        )

        # 6. Compute non-negative aggregation weights
        raw_weights: Dict[str, float] = {}
        for cid in client_ids:
            eval_res = evaluations[cid]
            n_samples = max(1, client_sample_counts.get(cid, 1))
            t_score = eval_res.composite_trust_score

            if eval_res.is_suspicious:
                if self.policy == "quarantine":
                    w = 0.0
                    eval_res.quarantined = True
                elif self.policy == "soft_clip":
                    # Soft polynomial decay
                    decayed_t = max(0.01, t_score) ** self.penalty_exponent
                    w = float(n_samples * decayed_t)
                else: # scaled
                    w = float(n_samples * t_score)
            else:
                w = float(n_samples * t_score)

            raw_weights[cid] = max(0.0, w)

        total_weight = sum(raw_weights.values())
        fallback_triggered = False

        # Edge case: all clients quarantined or zero total weight
        if total_weight <= 1e-9:
            fallback_triggered = True
            logger.warning(
                f"[ROUND {server_round}] ZERO WEIGHT EMERGENCY: All clients below trust threshold "
                f"({self.suspicious_threshold}). Triggering fallback policy: '{self.zero_weight_fallback}'."
            )
            # Assign equal weights among all available clients as safe uniform baseline
            norm_weight = 1.0 / len(client_ids)
            final_weights = {cid: norm_weight for cid in client_ids}
        else:
            final_weights = {cid: raw_weights[cid] / total_weight for cid in client_ids}

        # Store assigned weights in evaluation records
        for cid in client_ids:
            evaluations[cid].aggregation_weight = round(final_weights[cid], 4)

        return final_weights, evaluations, fallback_triggered

    @staticmethod
    def calculate_detection_metrics(
        evaluations: Dict[str, ClientTrustEvaluation],
        ground_truth_attackers: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate Byzantine detection performance against post-hoc ground truth.
        STRICT NOTICE: This is an evaluation-only diagnostic. Ground-truth labels
        are NEVER exposed to the aggregator or trust engine during optimization.
        """
        tp = 0 # Malicious correctly flagged as suspicious
        fp = 0 # Honest falsely flagged as suspicious
        tn = 0 # Honest correctly identified as trusted
        fn = 0 # Malicious falsely identified as trusted

        gt_set = set(ground_truth_attackers)

        for cid, ev in evaluations.items():
            is_malicious = cid in gt_set
            flagged = ev.is_suspicious

            if is_malicious and flagged:
                tp += 1
            elif not is_malicious and flagged:
                fp += 1
            elif not is_malicious and not flagged:
                tn += 1
            elif is_malicious and not flagged:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0 if tp == 0 and fp == 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0 if tp == 0 and fn == 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Calculate malicious influence: fraction of aggregate weight assigned to Byzantine nodes
        total_weight = sum(ev.aggregation_weight for ev in evaluations.values())
        malicious_weight = sum(ev.aggregation_weight for cid, ev in evaluations.items() if cid in gt_set)
        malicious_influence = malicious_weight / total_weight if total_weight > 0 else 0.0

        return {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4),
            "malicious_influence": round(float(malicious_influence), 4),
        }
