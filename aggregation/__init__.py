"""
Aggregation strategies for Byzantine-Resilient Federated Learning.
"""

from aggregation.coordinate_median import (
    CoordinateMedianStrategy,
    aggregate_coordinate_median,
)
from aggregation.trust_aware_fedavg import TrustAwareFedAvg

__all__ = [
    "CoordinateMedianStrategy",
    "TrustAwareFedAvg",
    "aggregate_coordinate_median",
]
