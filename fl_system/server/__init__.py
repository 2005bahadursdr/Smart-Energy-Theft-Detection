"""
fl_system.server package.
Blockchain-Based Byzantine-Resilient Federated Learning for Smart Grid Energy Theft Detection.

Exposes SmartGridFedAvg strategy, server runner, and metrics aggregator.
"""

from fl_system.server.metrics_aggregator import aggregate_metrics_weighted
from fl_system.server.server_runner import run_fl_server
from fl_system.server.strategy import SmartGridFedAvg

__all__ = [
    "SmartGridFedAvg",
    "run_fl_server",
    "aggregate_metrics_weighted",
]
