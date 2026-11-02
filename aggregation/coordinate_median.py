"""
Coordinate-Wise Median Robust Aggregation Strategy.
Serves as the empirical robust baseline for Byzantine resilience benchmarks.

Mathematical Properties:
- Coordinate-wise median computes the median value along each individual scalar coordinate:
    w_{agg, l}[k] = median({ w_{i, l}[k] for i in 1..N })
- Tolerates up to 50% Byzantine adversaries without arbitrary breakdown under coordinate
  independence assumptions (Blanchard et al., Yin et al.).
- Preserves identical smart grid telemetry tracking, dual evaluation, and checkpointing.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from fl_system.server.strategy import SmartGridFedAvg

logger = logging.getLogger("coordinate_median_strategy")


def aggregate_coordinate_median(
    results: List[Tuple[ClientProxy, FitRes]],
) -> List[np.ndarray]:
    """
    Compute coordinate-wise median across all client parameter arrays.

    Args:
        results: List of (ClientProxy, FitRes) tuples from Flower clients.

    Returns:
        List of aggregated numpy parameter arrays.
    """
    if not results:
        return []

    # Extract parameter arrays from all clients
    # client_params[i] = list of layer ndarrays for client i
    all_client_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
    num_layers = len(all_client_ndarrays[0])

    aggregated_layers = []
    for layer_idx in range(num_layers):
        layer_stack = np.stack(
            [client_ndarrays[layer_idx] for client_ndarrays in all_client_ndarrays],
            axis=0,
        ) # Shape: (N, dim_1, dim_2, ...)
        med = np.median(layer_stack, axis=0).astype(all_client_ndarrays[0][layer_idx].dtype)
        aggregated_layers.append(med)

    return aggregated_layers


class CoordinateMedianStrategy(SmartGridFedAvg):
    """
    Byzantine-robust Federated Learning Strategy utilizing Coordinate-Wise Median.
    """

    def __init__(
        self,
        *,
        model_name: str = "cnn_lstm",
        input_size: int = 11,
        benchmark_loader: Optional[DataLoader] = None,
        checkpoint_dir: Union[str, Path] = "fl_system/checkpoints/median",
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        pos_weight: float = 5.0,
        device: str = "cpu",
    ):
        super().__init__(
            model_name=model_name,
            input_size=input_size,
            benchmark_loader=benchmark_loader,
            checkpoint_dir=checkpoint_dir,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            pos_weight=pos_weight,
            device=device,
        )
        logger.info("Initialized CoordinateMedianStrategy (Robust Baseline Aggregator).")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client updates using coordinate-wise median instead of sample-weighted mean.
        """
        if not results:
            return None, {}

        # 1. Telemetry and Client Metadata Tracking
        client_metrics: Dict[str, Dict[str, Any]] = {}
        total_uplink_bytes = 0
        total_samples = 0
        participating_clients = []

        for client_proxy, fit_res in results:
            cid = fit_res.metrics.get("client_id", getattr(client_proxy, "cid", "unknown"))
            participating_clients.append(str(cid))
            num_samples = fit_res.num_examples
            total_samples += num_samples

            client_arrays = parameters_to_ndarrays(fit_res.parameters)
            uplink_bytes = sum(arr.nbytes for arr in client_arrays)
            total_uplink_bytes += uplink_bytes

            client_metrics[str(cid)] = {
                "num_samples": num_samples,
                "train_loss": float(fit_res.metrics.get("train_loss", 0.0)),
                "grad_norm": float(fit_res.metrics.get("grad_norm", 0.0)),
                "duration_sec": float(fit_res.metrics.get("duration_sec", 0.0)),
                "uplink_bytes": uplink_bytes,
            }

        downlink_per_client = sum(p.nbytes for p in self.latest_ndarrays) if self.latest_ndarrays else 0
        total_downlink_bytes = downlink_per_client * len(results)

        # 2. Execute Coordinate-Wise Median Aggregation
        aggregated_ndarrays = aggregate_coordinate_median(results)
        self.latest_ndarrays = aggregated_ndarrays
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # 3. Synchronize reference PyTorch model
        from fl_system.models.model_utils import save_checkpoint, set_parameters_from_numpy
        set_parameters_from_numpy(self.reference_model, self.latest_ndarrays)

        # 4. Save round checkpoint
        ckpt_filename = f"global_model_round_{server_round}.pt"
        ckpt_path = self.checkpoint_dir / ckpt_filename
        save_checkpoint(
            model=self.reference_model,
            optimizer=None,
            epoch=server_round,
            metrics={"aggregation": "coordinate_median"},
            checkpoint_path=ckpt_path,
            metadata={"model_name": f"median_{self.model_name}"},
        )

        # 5. Centralized Benchmark Evaluation Protocol
        benchmark_metrics = self._evaluate_server_benchmark()

        # Update best checkpoint based on F1
        saved_best = False
        if "f1_score" in benchmark_metrics and aggregated_parameters is not None:
            current_f1 = benchmark_metrics["f1_score"]
            if current_f1 >= self.best_f1_score:
                self.best_f1_score = current_f1
                best_ckpt_path = self.checkpoint_dir / "global_model_best.pt"
                save_checkpoint(
                    model=self.reference_model,
                    optimizer=None,
                    epoch=server_round,
                    metrics={"best_f1": current_f1, **benchmark_metrics},
                    checkpoint_path=best_ckpt_path,
                    metadata={"model_name": f"median_{self.model_name}_best"},
                )
                saved_best = True

        # 6. Structured Round Audit Record
        round_record = {
            "round": server_round,
            "aggregation_strategy": "coordinate_wise_median",
            "participating_clients": participating_clients,
            "client_count": len(participating_clients),
            "total_samples": total_samples,
            "client_fit_metrics": client_metrics,
            "server_benchmark_metrics": benchmark_metrics,
            "communication": {
                "total_uplink_bytes": total_uplink_bytes,
                "total_downlink_bytes": total_downlink_bytes,
                "total_bytes": total_uplink_bytes + total_downlink_bytes,
                "round_total_mb": round((total_uplink_bytes + total_downlink_bytes) / (1024 * 1024), 3),
            },
            "saved_best_checkpoint": saved_best,
            "best_f1_to_date": round(self.best_f1_score, 4),
        }
        self.history.append(round_record)
        self._flush_history_to_disk()

        logger.info(
            f"[ROUND {server_round}] CoordinateMedian Aggregation complete. "
            f"Clients: {len(participating_clients)} | Total Samples: {total_samples} "
            f"| Best F1: {self.best_f1_score:.4f}"
        )

        # Fit metrics aggregation (mean of training loss)
        avg_train_loss = float(np.mean([m["train_loss"] for m in client_metrics.values()]))
        return aggregated_parameters, {"train_loss": avg_train_loss, "aggregation": "coordinate_median"}
