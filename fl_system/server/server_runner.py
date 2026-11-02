"""
Flower gRPC Server Runner for Federated Learning.
Starts central orchestrator listening for substation client connections.
"""

import logging
from pathlib import Path
from typing import Optional, Union
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Parameters
from fl_system.server.strategy import SmartGridFedAvg

logger = logging.getLogger("fl_server_runner")


def run_fl_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    num_rounds: int = 5,
    min_fit_clients: int = 2,
    min_available_clients: int = 2,
    model_name: str = "cnn_lstm",
    checkpoint_dir: Union[str, Path] = "fl_system/checkpoints",
    benchmark_loader: Optional[DataLoader] = None,
    local_epochs: int = 1,
    learning_rate: float = 0.001,
) -> SmartGridFedAvg:
    """
    Launch Flower gRPC server configured with SmartGridFedAvg strategy.
    """
    server_address = f"{host}:{port}"
    logger.info(f"Starting Smart Grid FL Server on {server_address} for {num_rounds} rounds...")

    strategy = SmartGridFedAvg(
        model_name=model_name,
        benchmark_loader=benchmark_loader,
        checkpoint_dir=checkpoint_dir,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        min_evaluate_clients=min_fit_clients,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )

    config = fl.server.ServerConfig(num_rounds=num_rounds)

    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )

    logger.info(f"FL Server completed {num_rounds} rounds successfully.")
    return strategy
