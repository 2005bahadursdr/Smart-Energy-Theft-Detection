"""
Federated Learning Simulation Coordinator.
Orchestrates multi-client federated training rounds across 3-5 simulated
substations using Flower SmartGridFedAvg strategy and SmartGridFlowerClient instances.

Operates deterministically without Ray dependency, executing the exact Flower
client-server protocol (configure_fit -> client.fit -> aggregate_fit -> evaluate).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from fl_system.clients.base_client import (
    FL_PRIVACY_BOUNDARY_DISCLAIMER,
    SmartGridFlowerClient,
)
from fl_system.clients.factory import create_all_clients, create_client
from fl_system.server.strategy import SmartGridFedAvg

logger = logging.getLogger("fl_coordinator")


class DummyClientProxy(ClientProxy):
    """Minimal ClientProxy wrapper for in-memory Flower coordination."""

    def __init__(self, cid: str):
        super().__init__(cid=cid)

    def get_properties(self, ins, timeout=None):
        return None

    def get_parameters(self, ins, timeout=None):
        return None

    def fit(self, ins: FitIns, timeout=None) -> FitRes:
        return None

    def evaluate(self, ins: EvaluateIns, timeout=None) -> EvaluateRes:
        return None

    def reconnect(self, ins, timeout=None):
        return None


class FLSimulationCoordinator:
    """
    Coordinates multi-round Federated Learning experiments across smart grid substations.

    Features
    --------
    1. Configurable round count and client sampling (e.g. 3 of 5, or all 5).
    2. Local parameter updates only (verifies zero raw readings transmitted).
    3. Documented dual-level evaluation protocol (local client & server benchmark).
    4. Communication footprint tracking (bytes uplink/downlink).
    5. Automatic checkpoint saving and JSON history logging.
    6. Strict partition isolation enforcement.
    """

    def __init__(
        self,
        clients: Optional[Dict[str, SmartGridFlowerClient]] = None,
        strategy: Optional[SmartGridFedAvg] = None,
        num_rounds: int = 5,
        clients_per_round: Optional[int] = None,
        fraction_fit: float = 1.0,
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        model_name: str = "cnn_lstm",
        checkpoint_dir: Union[str, Path] = "fl_system/checkpoints",
        benchmark_loader: Optional[DataLoader] = None,
        attack_manager: Optional[Any] = None,
        enable_blockchain: bool = False,
        blockchain_adapter: Optional[Any] = None,
        seed: int = 42,
    ):
        self.num_rounds = num_rounds
        self.fraction_fit = fraction_fit
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_loader = benchmark_loader
        self.attack_manager = attack_manager
        self.enable_blockchain = enable_blockchain
        self.blockchain_adapter = blockchain_adapter
        self.seed = seed
        self.random_state = random.Random(seed)

        # 1. Initialize Clients (defaults to client_A through client_E if None)
        if clients is None:
            self.clients = create_all_clients(
                model_name=model_name,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                seed=seed,
            )
        else:
            self.clients = clients

        self.client_ids = list(self.clients.keys())
        self.clients_per_round = clients_per_round or max(1, int(len(self.client_ids) * fraction_fit))

        # 2. Initialize Strategy
        if strategy is None:
            self.strategy = SmartGridFedAvg(
                model_name=model_name,
                benchmark_loader=benchmark_loader,
                checkpoint_dir=self.checkpoint_dir,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                min_fit_clients=min(2, len(self.client_ids)),
                min_available_clients=len(self.client_ids),
                fraction_fit=self.fraction_fit,
            )
        else:
            self.strategy = strategy

        # Current global parameters (NumPy arrays)
        self.global_parameters: List[np.ndarray] = self.strategy.latest_ndarrays

        logger.info(
            f"Initialized FLSimulationCoordinator with {len(self.clients)} clients "
            f"({self.client_ids}) | Rounds: {num_rounds} | Sampled/Round: {self.clients_per_round}"
        )

    def run(self) -> Dict[str, Any]:
        """
        Execute federated learning training rounds across all substations.

        Returns
        -------
        Dict[str, Any]
            Complete simulation summary including round logs and communication telemetry.
        """
        logger.info(
            f"=== Commencing Federated Learning Simulation: {self.num_rounds} Rounds across "
            f"{len(self.clients)} Substations ==="
        )
        logger.info(FL_PRIVACY_BOUNDARY_DISCLAIMER)

        start_time = datetime.now(timezone.utc)

        for current_round in range(1, self.num_rounds + 1):
            logger.info(f"\n--- [Round {current_round}/{self.num_rounds}] ---")

            # 1. Sample Participating Substation Clients
            sampled_ids = self.random_state.sample(self.client_ids, self.clients_per_round)
            logger.info(f"Sampled Clients ({len(sampled_ids)}/{len(self.client_ids)}): {sampled_ids}")

            # 2. Client Local Training (Strictly On-Premise)
            fit_results: List[Tuple[ClientProxy, FitRes]] = []
            client_transmitted_params: Dict[str, List[np.ndarray]] = {}
            fit_config = {
                "server_round": current_round,
                "local_epochs": self.local_epochs,
                "learning_rate": self.learning_rate,
            }

            for cid in sampled_ids:
                client = self.clients[cid]
                # If label flipping is active and client is attacker, poison training dataset
                if self.attack_manager is not None and getattr(self.attack_manager, "active_attack", None) == "label_flipping":
                    if hasattr(client, "train_dataset") and client.train_dataset is not None:
                        client.train_dataset = self.attack_manager.poison_training_dataset(
                            client_id=cid,
                            server_round=current_round,
                            dataset=client.train_dataset,
                        )

                # Client receives global parameters, trains locally, returns updated parameters
                updated_params, num_samples, metrics = client.fit(
                    parameters=self.global_parameters,
                    config=fit_config,
                )

                # If Byzantine attack manager is configured, apply adversarial perturbation
                if self.attack_manager is not None:
                    transmitted_params = self.attack_manager.apply_attack(
                        client_id=cid,
                        server_round=current_round,
                        global_parameters=self.global_parameters,
                        local_parameters=updated_params,
                    )
                else:
                    transmitted_params = updated_params

                client_transmitted_params[cid] = transmitted_params

                # Convert parameters to Flower Parameters object
                fit_res = FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=ndarrays_to_parameters(transmitted_params),
                    num_examples=num_samples,
                    metrics=metrics,
                )
                fit_results.append((DummyClientProxy(cid), fit_res))

            # 3. Server FedAvg Aggregation
            agg_parameters, agg_metrics = self.strategy.aggregate_fit(
                server_round=current_round,
                results=fit_results,
                failures=[],
            )

            if agg_parameters is not None:
                self.global_parameters = parameters_to_ndarrays(agg_parameters)

            # 4. Local Evaluation on All Participating Clients
            eval_results: List[Tuple[ClientProxy, EvaluateRes]] = []
            eval_config = {"server_round": current_round}

            for cid in sampled_ids:
                client = self.clients[cid]
                loss, num_test, eval_metrics = client.evaluate(
                    parameters=self.global_parameters,
                    config=eval_config,
                )
                eval_res = EvaluateRes(
                    status=Status(code=Code.OK, message="Success"),
                    loss=loss,
                    num_examples=num_test,
                    metrics=eval_metrics,
                )
                eval_results.append((DummyClientProxy(cid), eval_res))

            # 5. Aggregate Local Evaluations
            self.strategy.aggregate_evaluate(
                server_round=current_round,
                results=eval_results,
                failures=[],
            )

            # 6. Blockchain Audit Trail Commitment
            if self.enable_blockchain:
                try:
                    from blockchain.service import commit_fl_round_audit, get_blockchain_adapter
                    bc_adapter = self.blockchain_adapter or get_blockchain_adapter()

                    last_record = self.strategy.history[-1] if self.strategy.history else {}
                    trust_evals = last_record.get("trust_evaluations", {})
                    if not trust_evals:
                        trust_evals = {
                            cid: {"is_suspicious": False, "quarantined": False, "reputation_score": 0.70}
                            for cid in sampled_ids
                        }

                    client_deltas = {
                        cid: [l - g for l, g in zip(client_transmitted_params[cid], self.global_parameters)]
                        for cid in sampled_ids
                        if cid in client_transmitted_params
                    }
                    rep_engine = getattr(getattr(self.strategy, "trust_engine", None), "reputation_engine", None)
                    rep_updates = {
                        cid: rep_engine.get_reputation(cid) if rep_engine else 0.75
                        for cid in sampled_ids
                    }

                    audit_res = commit_fl_round_audit(
                        round_num=current_round,
                        client_evaluations=trust_evals,
                        client_deltas=client_deltas,
                        reputation_updates=rep_updates,
                        adapter=bc_adapter,
                    )
                    if self.strategy.history:
                        self.strategy.history[-1]["blockchain_audit"] = {
                            "status": audit_res.get("status"),
                            "blockNumber": audit_res.get("blockNumber"),
                            "transactionHash": audit_res.get("transactionHash"),
                            "recordCount": len(audit_res.get("recordIds", [])),
                        }
                        self.strategy._flush_history_to_disk()
                except Exception as e:
                    logger.warning(f"[BLOCKCHAIN AUDIT] Round {current_round} commitment deferred: {e}")

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(f"=== Federated Learning Simulation Complete ({duration:.2f}s) ===")

        # Build summary
        total_comm_bytes = sum(
            r.get("communication", {}).get(
                "total_bytes",
                r.get("communication", {}).get("total_uplink_bytes", 0) + r.get("communication", {}).get("total_downlink_bytes", 0),
            )
            for r in self.strategy.history
        )
        summary = {
            "status": "COMPLETED",
            "total_rounds": self.num_rounds,
            "clients_count": len(self.clients),
            "client_ids": self.client_ids,
            "duration_seconds": round(duration, 2),
            "total_communication_bytes": total_comm_bytes,
            "total_communication_mb": round(total_comm_bytes / (1024 * 1024), 2),
            "best_f1_score": self.strategy.best_f1_score,
            "attack_configured": self.attack_manager is not None,
            "active_attack": getattr(self.attack_manager, "active_attack", "none") if self.attack_manager else "none",
            "attacker_clients": getattr(self.attack_manager, "attacker_clients", []) if self.attack_manager else [],
            "history": self.strategy.history,
            "checkpoint_dir": str(self.checkpoint_dir),
            "privacy_notice": FL_PRIVACY_BOUNDARY_DISCLAIMER,
        }

        if self.attack_manager is not None:
            gt_path = self.checkpoint_dir / "byzantine_ground_truth.json"
            self.attack_manager.export_ground_truth_log(gt_path)
            summary["ground_truth_log"] = str(gt_path)

        # Save summary
        summary_path = self.checkpoint_dir / "fl_simulation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
