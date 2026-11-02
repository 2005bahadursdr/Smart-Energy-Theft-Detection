"""
Substation Client Factory and Registry.
Initializes configured SmartGridFlowerClient instances with appropriate
regional grid topology attributes and partition vault directories.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import pandas as pd

from fl_system.clients.base_client import SmartGridFlowerClient

logger = logging.getLogger("fl_client_factory")

SUBSTATION_METADATA: Dict[str, Dict[str, str]] = {
    "client_A": {
        "name": "Industrial Feeder Substation A",
        "zone_type": "Industrial",
        "characteristics": "High mean baseline, extreme diurnal shifts, continuous process loads",
    },
    "client_B": {
        "name": "Commercial District Substation B",
        "zone_type": "Commercial",
        "characteristics": "High business hour peak, significant weekend drops",
    },
    "client_C": {
        "name": "Dense Urban Residential Substation C",
        "zone_type": "Urban Residential",
        "characteristics": "Sharp evening peaks, moderate daytime consumption, weather sensitivity",
    },
    "client_D": {
        "name": "Suburban Residential Substation D",
        "zone_type": "Suburban Residential",
        "characteristics": "Morning and evening dual peaks, variable seasonal air conditioning",
    },
    "client_E": {
        "name": "Rural & Agricultural Substation E",
        "zone_type": "Rural Mixed",
        "characteristics": "Low density, intermittent high pump irrigation loads, seasonal volatility",
    },
}


def create_client(
    client_id: str,
    model_name: str = "cnn_lstm",
    base_data_dir: Union[str, Path] = "data/processed/clients",
    df: Optional[pd.DataFrame] = None,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    local_epochs: int = 1,
    device: str = "cpu",
    seed: int = 42,
) -> SmartGridFlowerClient:
    """
    Instantiate a configured SmartGridFlowerClient for the specified substation.
    """
    if client_id not in SUBSTATION_METADATA and not client_id.startswith("client_"):
        logger.warning(f"Unregistered client ID '{client_id}'. Using standard configuration.")

    vault_dir = Path(base_data_dir) / client_id if df is None else None

    client = SmartGridFlowerClient(
        client_id=client_id,
        model_name=model_name,
        data_dir=vault_dir,
        df=df,
        batch_size=batch_size,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        device=device,
        seed=seed,
    )
    return client


def create_all_clients(
    client_ids: Optional[List[str]] = None,
    model_name: str = "cnn_lstm",
    base_data_dir: Union[str, Path] = "data/processed/clients",
    batch_size: int = 32,
    learning_rate: float = 0.001,
    local_epochs: int = 1,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, SmartGridFlowerClient]:
    """
    Instantiate all specified substation clients (defaults to client_A through client_E).
    """
    client_ids = client_ids or ["client_A", "client_B", "client_C", "client_D", "client_E"]
    clients = {}
    for cid in client_ids:
        clients[cid] = create_client(
            client_id=cid,
            model_name=model_name,
            base_data_dir=base_data_dir,
            batch_size=batch_size,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
            device=device,
            seed=seed,
        )
    return clients
