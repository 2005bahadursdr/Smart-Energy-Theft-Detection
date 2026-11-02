"""
fl_system package.
Blockchain-Based Byzantine-Resilient Federated Learning for Smart Grid Energy Theft Detection.

Modules:
- models: PyTorch neural networks (LSTM, CNN-LSTM), preprocessing, and evaluation.
- clients: Flower NumPyClient implementations for smart grid substations.
- server: SmartGridFedAvg strategy, checkpointing, and communication logger.
- coordinator: Multi-client simulation runner and experiment orchestrator.
"""

from fl_system.clients.base_client import (
    FL_PRIVACY_BOUNDARY_DISCLAIMER,
    SmartGridFlowerClient,
)
from fl_system.clients.factory import (
    SUBSTATION_METADATA,
    create_all_clients,
    create_client,
)
from fl_system.coordinator import FLSimulationCoordinator
from fl_system.server.strategy import SmartGridFedAvg
from fl_system.server.server_runner import run_fl_server

__all__ = [
    "SmartGridFlowerClient",
    "SmartGridFedAvg",
    "FLSimulationCoordinator",
    "create_client",
    "create_all_clients",
    "run_fl_server",
    "SUBSTATION_METADATA",
    "FL_PRIVACY_BOUNDARY_DISCLAIMER",
]
