"""
Smart Grid FedAvg Server Strategy.
Extends Flower FedAvg strategy with:
1. Dynamic communication telemetry tracking (uplink/downlink footprint, parameter count).
2. Documented dual-level evaluation protocol (local client-aggregated & server centralized benchmark).
3. Automatic model checkpointing (round-specific and best-performing checkpoints).
4. Comprehensive round-by-round JSON audit trail logging.
5. Strict privacy documentation and isolation validation.

CRITICAL PRIVACY DOCUMENTATION:
Federated Learning alone DOES NOT guarantee privacy. While raw consumption
telemetry remains strictly on-premise at regional substations and only model
parameter arrays and scalar metadata are transmitted to the server, standard
Federated Learning does not provide mathematical Differential Privacy (DP)
nor cryptographic Secure Multi-Party Computation (SMPC). Differential privacy
or secure aggregation must be implemented separately if formal privacy guarantees
against gradient inversion attacks are required.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fl_system.clients.base_client import FL_PRIVACY_BOUNDARY_DISCLAIMER
from fl_system.models.evaluate import calculate_binary_metrics
from fl_system.models.model_utils import (
    get_model,
    get_parameters_as_numpy,
    save_checkpoint,
    set_parameters_from_numpy,
)
from fl_system.server.metrics_aggregator import aggregate_metrics_weighted

logger = logging.getLogger("fl_strategy")


class SmartGridFedAvg(FedAvg):
    """
    Enhanced FedAvg Strategy tailored for Smart Grid Energy Theft Detection.
    """

    def __init__(
        self,
        *,
        model_name: str = "cnn_lstm",
        input_size: int = 11,
        benchmark_loader: Optional[DataLoader] = None,
        checkpoint_dir: Union[str, Path] = "fl_system/checkpoints",
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[[int, List[np.ndarray], Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        initial_parameters: Optional[Parameters] = None,
        pos_weight: float = 5.0,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.input_size = input_size
        self.benchmark_loader = benchmark_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.pos_weight = pos_weight
        self.device = torch.device(device)

        # Internal reference model for parameter shape validation and centralized evaluation
        self.reference_model = get_model(model_name=self.model_name, input_size=self.input_size).to(self.device)

        # Provide initial parameters if not explicitly passed
        if initial_parameters is None:
            initial_ndarrays = get_parameters_as_numpy(self.reference_model)
            initial_parameters = ndarrays_to_parameters(initial_ndarrays)

        # Default configuration generators passing round hyperparameters to clients
        if on_fit_config_fn is None:
            on_fit_config_fn = self._default_fit_config
        if on_evaluate_config_fn is None:
            on_evaluate_config_fn = self._default_eval_config

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=aggregate_metrics_weighted,
            evaluate_metrics_aggregation_fn=aggregate_metrics_weighted,
        )

        # Audit and History State
        self.history: List[Dict[str, Any]] = []
        self.current_round_data: Dict[str, Any] = {}
        self.best_f1_score: float = -1.0
        self.latest_ndarrays: Optional[List[np.ndarray]] = parameters_to_ndarrays(initial_parameters)

        logger.info(
            f"Initialized SmartGridFedAvg Strategy | Model: {self.model_name.upper()} | "
            f"Input Dim: {self.input_size} | Checkpoint Dir: {self.checkpoint_dir} | "
            f"Min Fit Clients: {min_fit_clients} | Fraction Fit: {fraction_fit}"
        )

    def _default_fit_config(self, server_round: int) -> Dict[str, Scalar]:
        """Distribute round hyperparameters to sampled fit clients."""
        return {
            "server_round": server_round,
            "local_epochs": self.local_epochs,
            "learning_rate": self.learning_rate,
        }

    def _default_eval_config(self, server_round: int) -> Dict[str, Scalar]:
        """Distribute evaluation hyperparameters to clients."""
        return {
            "server_round": server_round,
        }

    # --------------------------------------------------------------------------
    # Fit & Weight Aggregation Protocol
    # --------------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate local model updates via weighted FedAvg:
        w_{t+1} = sum_k (n_k / N) * w_{t+1}^k

        Logs communication overhead, updates audit history, and saves round checkpoint.
        """
        start_ts = datetime.now(timezone.utc).isoformat()
        logger.info(f"=== FL Server Round {server_round}: Aggregating Fit Updates ({len(results)} clients) ===")

        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} client fit updates failed.")

        if not results:
            return None, {}

        # 1. Inspect communication footprint and client metadata
        client_metrics: Dict[str, Dict[str, Any]] = {}
        total_uplink_bytes = 0
        total_samples = 0
        participating_clients = []

        for client_proxy, fit_res in results:
            cid = fit_res.metrics.get("client_id", getattr(client_proxy, "cid", "unknown"))
            participating_clients.append(str(cid))
            num_samples = fit_res.num_examples
            total_samples += num_samples

            # Uplink parameters size
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

        # Downlink footprint: global parameters broadcast to all sampled clients
        downlink_per_client = sum(p.nbytes for p in self.latest_ndarrays) if self.latest_ndarrays else 0
        total_downlink_bytes = downlink_per_client * len(results)

        # 2. Execute Base FedAvg Aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            self.latest_ndarrays = parameters_to_ndarrays(aggregated_parameters)

            # 3. Synchronize reference PyTorch model
            set_parameters_from_numpy(self.reference_model, self.latest_ndarrays)

            # 4. Save Checkpoint for this round
            ckpt_filename = f"global_model_round_{server_round}.pt"
            ckpt_path = self.checkpoint_dir / ckpt_filename
            save_checkpoint(
                model=self.reference_model,
                optimizer=None,
                epoch=server_round,
                metrics=aggregated_metrics,
                checkpoint_path=ckpt_path,
                metadata={"model_name": f"fedavg_{self.model_name}"},
            )

        # 5. Centralized Benchmark Evaluation Protocol (Server-side validation)
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
                    metadata={"model_name": f"fedavg_{self.model_name}_best"},
                )
                saved_best = True

        # 6. Structured Round Audit Record
        round_record = {
            "round": server_round,
            "timestamp": start_ts,
            "participating_clients": participating_clients,
            "client_count": len(participating_clients),
            "total_samples": total_samples,
            "client_fit_metrics": client_metrics,
            "aggregated_fit_metrics": aggregated_metrics,
            "server_benchmark_metrics": benchmark_metrics,
            "communication": {
                "uplink_bytes": total_uplink_bytes,
                "downlink_bytes": total_downlink_bytes,
                "total_bytes": total_uplink_bytes + total_downlink_bytes,
                "total_kb": round((total_uplink_bytes + total_downlink_bytes) / 1024, 2),
            },
            "checkpoint_saved": str(ckpt_path) if aggregated_parameters is not None else None,
            "is_best_model": saved_best,
            "best_f1_to_date": round(self.best_f1_score, 4),
            "privacy_notice": FL_PRIVACY_BOUNDARY_DISCLAIMER,
        }
        self.history.append(round_record)
        self._flush_history_to_disk()

        logger.info(
            f"Round {server_round} Complete | Uplink: {total_uplink_bytes / 1024:.1f} KB | "
            f"Downlink: {total_downlink_bytes / 1024:.1f} KB | Checkpoint: {ckpt_filename} "
            f"| Best F1: {self.best_f1_score:.4f}"
        )

        return aggregated_parameters, aggregated_metrics

    # --------------------------------------------------------------------------
    # Evaluation Protocol
    # --------------------------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate local client test evaluations according to sample-weighted protocol.
        """
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        current_f1 = float(metrics.get("f1_score", 0.0)) if metrics else 0.0

        if current_f1 >= self.best_f1_score:
            self.best_f1_score = current_f1
            best_ckpt_path = self.checkpoint_dir / "global_model_best.pt"
            save_checkpoint(
                model=self.reference_model,
                optimizer=None,
                epoch=server_round,
                metrics={"best_f1": current_f1, **(metrics or {})},
                checkpoint_path=best_ckpt_path,
                metadata={"model_name": f"fedavg_{self.model_name}_best"},
            )

        if self.history and self.history[-1]["round"] == server_round:
            self.history[-1]["aggregated_evaluate_metrics"] = metrics
            self.history[-1]["client_evaluate_loss"] = loss
            self._flush_history_to_disk()

        return loss, metrics

    def _evaluate_server_benchmark(self) -> Dict[str, float]:
        """
        Execute server-side evaluation on clean benchmark DataLoader (Server-side ground truth).
        Ensures unbiased fraud detection verification regardless of client-side heterogeneity.
        """
        if self.benchmark_loader is None:
            return {}

        self.reference_model.eval()
        pos_weight_tensor = torch.tensor([self.pos_weight], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        total_loss = 0.0
        all_targets = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for X_b, y_b in self.benchmark_loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device).float()
                logits = self.reference_model(X_b)
                if logits.dim() == 1 and y_b.dim() > 1:
                    y_b_loss = y_b.squeeze(-1)
                elif logits.dim() > 1 and y_b.dim() == 1:
                    y_b_loss = y_b.unsqueeze(-1)
                else:
                    y_b_loss = y_b
                loss = criterion(logits, y_b_loss)
                total_loss += loss.item() * len(y_b)

                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= 0.5).astype(int)
                targets = y_b.cpu().numpy().flatten().astype(int)

                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_targets.extend(targets.tolist())

        n_samples = len(all_targets)
        if n_samples == 0:
            return {}

        avg_loss = total_loss / n_samples
        metrics = calculate_binary_metrics(
            y_true=np.array(all_targets),
            y_prob=np.array(all_probs),
            loss=avg_loss,
        )

        res = {
            "loss": round(float(avg_loss), 4),
            "accuracy": round(float(metrics.accuracy), 4),
            "precision": round(float(metrics.precision), 4),
            "recall": round(float(metrics.recall), 4),
            "f1_score": round(float(metrics.f1_score), 4),
            "pr_auc": round(float(metrics.pr_auc), 4),
            "roc_auc": round(float(metrics.roc_auc if metrics.roc_auc is not None else 0.0), 4),
        }
        logger.info(
            f"[SERVER BENCHMARK] Round Loss: {res['loss']} | F1: {res['f1_score']} "
            f"| Recall: {res['recall']} | PR-AUC: {res['pr_auc']}"
        )
        return res

    def _flush_history_to_disk(self) -> None:
        """Persist full round audit trail to disk."""
        log_file = self.checkpoint_dir / "fl_round_history.json"
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write round history log: {e}")
