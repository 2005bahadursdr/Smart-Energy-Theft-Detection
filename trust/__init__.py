"""
Trust evaluation and reputation subsystem for Byzantine-resilient Federated Learning.
"""

from trust.reputation.engine import HistoricalReputationEngine
from trust.similarity.cosine_distance import (
    compute_coordinate_median,
    compute_cosine_similarity,
    compute_euclidean_distance,
    compute_parameter_delta,
    evaluate_similarity_and_distance,
)
from trust.trust_engine import ClientTrustEvaluation, TrustEngine
from trust.validation.evaluator import ClientUpdateValidationEvaluator

__all__ = [
    "ClientTrustEvaluation",
    "ClientUpdateValidationEvaluator",
    "HistoricalReputationEngine",
    "TrustEngine",
    "compute_coordinate_median",
    "compute_cosine_similarity",
    "compute_euclidean_distance",
    "compute_parameter_delta",
    "evaluate_similarity_and_distance",
]
