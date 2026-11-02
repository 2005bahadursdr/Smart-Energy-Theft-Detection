"""
fl_system.clients package.
Blockchain-Based Byzantine-Resilient Federated Learning for Smart Grid Energy Theft Detection.

Exposes SmartGridFlowerClient and client factory for substations client_A through client_E.
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

__all__ = [
    "SmartGridFlowerClient",
    "create_client",
    "create_all_clients",
    "SUBSTATION_METADATA",
    "FL_PRIVACY_BOUNDARY_DISCLAIMER",
]
