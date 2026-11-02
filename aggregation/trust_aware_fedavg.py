"""
Trust-Aware FedAvg Strategy for Smart Grid Energy Theft Detection.

Integrates Multi-Criteria TrustEngine to dynamically filter, downweight, or quarantine
Byzantine adversaries while protecting non-IID honest clients.

Key Properties:
1. Normalized, non-negative aggregation weights: omega_i >= 0, sum(omega_i) = 1.0.
2. Configurable suspicious thresholds and weight reduction policies (quarantine / soft-clip).
3. Graceful fallback to Coordinate-Wise Median if all updates are flagged.
4. Detailed round telemetry including component trust scores and malicious influence metrics.
"""

from dataclasses import asdict
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

from aggregation.coordinate_median import aggregate_coordinate_median
from fl_system.models.model_utils import save_checkpoint, set_parameters_from_numpy
from fl_system.server.strategy import SmartGridFedAvg
from trust.trust_engine import ClientTrustEvaluation, TrustEngine
from trust.validation.evaluator import ClientUpdateValidationEvaluator

logger = logging.getLogger("trust_aware_fedavg")


class TrustAwareFedAvg(SmartGridFedAvg):
    """
    Research Trust-Aware FedAvg strategy with multi-criteria Byzantine defense.
    """

    def __init__(
        self,
        *,
        model_name: str = "cnn_lstm",
        input_size: int = 11,
        benchmark_loader: Optional[DataLoader] = None,
        checkpoint_dir: Union[str, Path] = "fl_system/checkpoints/trust_aware",
        trust_engine: Optional[TrustEngine] = None,
        config_path: Optional[Union[str, Path]] = None,
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

        # Initialize TrustEngine if not provided
        if trust_engine is not None:
            self.trust_engine = trust_engine
        else:
            val_evaluator = None
            if self.benchmark_loader is not None:
                val_evaluator = ClientUpdateValidationEvaluator(
                    reference_model=self.reference_model,
                    benchmark_loader=self.benchmark_loader,
                    pos_weight=self.pos_weight,
                    device=device,
                )
            self.trust_engine = TrustEngine(
                config_path=config_path,
                validation_evaluator=val_evaluator,
            )

        # Evaluation-only ground-truth attacker labels (STRICT NOTICE: post-hoc only)
        self._ground_truth_attackers: List[str] = []

        logger.info(
            f"Initialized TrustAwareFedAvg Strategy | Checkpoint Dir: {self.checkpoint_dir} | "
            f"Policy: {self.trust_engine.policy} | Suspicious Threshold: {self.trust_engine.suspicious_threshold}"
        )

    def set_ground_truth_for_evaluation(self, attacker_client_ids: List[str]) -> None:
        """
        Record ground-truth attacker labels strictly for diagnostic evaluation.
        The TrustEngine never accesses this attribute during parameter weighting.
        """
        self._ground_truth_attackers = list(attacker_client_ids)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client updates using multi-criteria trust weighting.
        """
        if not results:
            return None, {}

        # 1. Telemetry and Client Parameter Extraction
        client_updates: Dict[str, List[np.ndarray]] = {}
        client_sample_counts: Dict[str, int] = {}
        client_metrics: Dict[str, Dict[str, Any]] = {}
        total_uplink_bytes = 0
        total_samples = 0
        participating_clients = []

        for client_proxy, fit_res in results:
            cid = str(fit_res.metrics.get("client_id", getattr(client_proxy, "cid", "unknown")))
            participating_clients.append(cid)
            num_samples = fit_res.num_examples
            total_samples += num_samples

            client_arrays = parameters_to_ndarrays(fit_res.parameters)
            client_updates[cid] = client_arrays
            client_sample_counts[cid] = num_samples

            uplink_bytes = sum(arr.nbytes for arr in client_arrays)
            total_uplink_bytes += uplink_bytes

            client_metrics[cid] = {
                "num_samples": num_samples,
                "train_loss": float(fit_res.metrics.get("train_loss", 0.0)),
                "grad_norm": float(fit_res.metrics.get("grad_norm", 0.0)),
                "duration_sec": float(fit_res.metrics.get("duration_sec", 0.0)),
                "uplink_bytes": uplink_bytes,
            }

        downlink_per_client = sum(p.nbytes for p in self.latest_ndarrays) if self.latest_ndarrays else 0
        total_downlink_bytes = downlink_per_client * len(results)

        # 2. Trust Evaluation and Weight Computation
        current_global = [arr.copy() for arr in self.latest_ndarrays] if self.latest_ndarrays else client_updates[participating_clients[0]]

        weights, evaluations, fallback_triggered = self.trust_engine.evaluate_and_weight_updates(
            current_global_parameters=current_global,
            client_updates=client_updates,
            client_sample_counts=client_sample_counts,
            server_round=server_round,
        )

        # 3. Model Parameter Aggregation
        if fallback_triggered and self.trust_engine.zero_weight_fallback == "coordinate_median":
            logger.warning(f"[ROUND {server_round}] Executing Coordinate-Wise Median fallback aggregation.")
            aggregated_ndarrays = aggregate_coordinate_median(results)
        else:
            # Weighted average: w_global = sum(omega_i * w_i)
            num_layers = len(current_global)
            aggregated_ndarrays = []
            for layer_idx in range(num_layers):
                weighted_layer = np.zeros_like(current_global[layer_idx], dtype=np.float64)
                for cid in participating_clients:
                    w_i = weights.get(cid, 0.0)
                    layer_arr = client_updates[cid][layer_idx].astype(np.float64)
                    weighted_layer += w_i * layer_arr
                aggregated_ndarrays.append(weighted_layer.astype(current_global[layer_idx].dtype))

        self.latest_ndarrays = aggregated_ndarrays
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # 4. Synchronize reference PyTorch model
        set_parameters_from_numpy(self.reference_model, self.latest_ndarrays)

        # 5. Save round checkpoint
        ckpt_filename = f"global_model_round_{server_round}.pt"
        ckpt_path = self.checkpoint_dir / ckpt_filename
        save_checkpoint(
            model=self.reference_model,
            optimizer=None,
            epoch=server_round,
            metrics={"aggregation": "trust_aware_fedavg"},
            checkpoint_path=ckpt_path,
            metadata={"model_name": f"trust_fedavg_{self.model_name}"},
        )

        # 6. Centralized Benchmark Evaluation Protocol
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
                    metadata={"model_name": f"trust_fedavg_{self.model_name}_best"},
                )
                saved_best = True

        # 7. Post-Hoc Diagnostic Detection Metrics (Evaluation only)
        detection_metrics = {}
        if self._ground_truth_attackers:
            detection_metrics = TrustEngine.calculate_detection_metrics(
                evaluations=evaluations,
                ground_truth_attackers=self._ground_truth_attackers,
            )

        # 8. Structured Round Audit Record
        round_record = {
            "round": server_round,
            "aggregation_strategy": "trust_aware_fedavg",
            "participating_clients": participating_clients,
            "client_count": len(participating_clients),
            "total_samples": total_samples,
            "trust_evaluations": {cid: asdict(ev) for cid, ev in evaluations.items()},
            "aggregation_weights": {cid: round(w, 4) for cid, w in weights.items()},
            "fallback_triggered": fallback_triggered,
            "detection_diagnostics": detection_metrics,
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

        quarantined_cids = [cid for cid, ev in evaluations.items() if ev.quarantined]
        logger.info(
            f"[ROUND {server_round}] Trust-Aware FedAvg complete. "
            f"Clients: {len(participating_clients)} | Quarantined: {quarantined_cids} "
            f"| Best F1: {self.best_f1_score:.4f}"
        )

        avg_train_loss = float(np.mean([m["train_loss"] for m in client_metrics.values()]))
        return aggregated_parameters, {
            "train_loss": avg_train_loss,
            "quarantined_count": len(quarantined_cids),
            "aggregation": "trust_aware_fedavg",
        }
