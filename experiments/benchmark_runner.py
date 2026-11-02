"""
Byzantine Federated Learning Benchmark Runner.
Executes reproducible comparative experiments:
1. Clean Federated Learning (Baseline FedAvg without attacks)
2. Byzantine Attacked Federated Learning (FedAvg under Sign-Flipping)
3. Byzantine Attacked Federated Learning (FedAvg under Model Poisoning)

PRESERVES STRICT RESEARCH SCIENTIFIC INTEGRITY:
1. Identical initial global model weights across all comparative arms.
2. Identical substation data splits and consumer partitions.
3. Ground-truth attacker identities recorded strictly for post-hoc evaluation;
   server aggregator/detector never accesses these labels.
4. An unusual update is not proof of malicious intent (epistemic disclaimer recorded).
5. All results saved to experiments/results/.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from byzantine.attack_manager import (
    ANOMALY_EPISTEMIC_DISCLAIMER,
    ByzantineAttackManager,
)
from fl_system.clients.base_client import FL_PRIVACY_BOUNDARY_DISCLAIMER
from fl_system.clients.factory import create_all_clients
from fl_system.coordinator import FLSimulationCoordinator
from fl_system.models.model_utils import (
    get_model,
    get_parameters_as_numpy,
    set_parameters_from_numpy,
    set_reproducible_seed,
)
from fl_system.server.strategy import SmartGridFedAvg
from flwr.common import ndarrays_to_parameters

logger = logging.getLogger("byzantine_benchmark")


def run_comparative_benchmark(
    num_rounds: int = 3,
    local_epochs: int = 1,
    learning_rate: float = 0.001,
    model_name: str = "cnn_lstm",
    results_dir: Union[str, Path] = "experiments/results",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run baseline FedAvg clean vs under Byzantine attacks (sign-flipping and model poisoning).
    Ensures identical initial weights and client partitions.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_reproducible_seed(seed)
    logger.info("=== Initializing Byzantine FL Comparative Benchmark ===")
    logger.info(ANOMALY_EPISTEMIC_DISCLAIMER)

    # 1. Generate Deterministic Initial Model Parameters (Identical for ALL Arms)
    init_model = get_model(model_name=model_name, input_size=11)
    common_initial_ndarrays = get_parameters_as_numpy(init_model)

    experiment_arms = [
        {
            "name": "fedavg_clean",
            "description": "Baseline FedAvg without Byzantine attacks (0% malicious nodes)",
            "attack_type": None,
            "attackers": [],
        },
        {
            "name": "fedavg_sign_flipping",
            "description": "FedAvg under Sign-Flipping attack (20% malicious: client_D inverting gradients)",
            "attack_type": "sign_flipping",
            "attackers": ["client_D"],
        },
        {
            "name": "fedavg_model_poisoning",
            "description": "FedAvg under Model Poisoning attack (20% malicious: client_D injecting adversarial noise)",
            "attack_type": "model_poisoning",
            "attackers": ["client_D"],
        },
    ]

    all_arm_summaries = {}

    for arm in experiment_arms:
        arm_name = arm["name"]
        logger.info(f"\n=======================================================")
        logger.info(f"Executing Benchmark Arm: {arm_name.upper()}")
        logger.info(f"Description: {arm['description']}")
        logger.info(f"=======================================================")

        # Identical initial parameters
        initial_params = ndarrays_to_parameters([arr.copy() for arr in common_initial_ndarrays])

        # Arm-specific checkpoint directory
        arm_ckpt_dir = out_dir / arm_name
        arm_ckpt_dir.mkdir(parents=True, exist_ok=True)

        strategy = SmartGridFedAvg(
            model_name=model_name,
            input_size=11,
            checkpoint_dir=arm_ckpt_dir,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            min_fit_clients=5,
            min_available_clients=5,
            initial_parameters=initial_params,
        )

        # Clients with identical data partitions
        clients = create_all_clients(
            model_name=model_name,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            seed=seed,
        )

        # Configure Attack Manager if arm has attacks
        attack_mgr = None
        if arm["attack_type"] is not None:
            attack_mgr = ByzantineAttackManager(
                active_attack=arm["attack_type"],
                attacker_clients=arm["attackers"],
                all_client_ids=list(clients.keys()),
                seed=seed,
            )

        coordinator = FLSimulationCoordinator(
            clients=clients,
            strategy=strategy,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            model_name=model_name,
            checkpoint_dir=arm_ckpt_dir,
            attack_manager=attack_mgr,
            seed=seed,
        )

        # Execute FL Simulation
        arm_summary = coordinator.run()
        arm_summary["arm_description"] = arm["description"]
        arm_summary["attack_type"] = arm["attack_type"]
        arm_summary["attackers"] = arm["attackers"]

        # Save arm-specific result
        arm_result_file = out_dir / f"{arm_name}_results.json"
        with open(arm_result_file, "w", encoding="utf-8") as f:
            json.dump(arm_summary, f, indent=2)

        all_arm_summaries[arm_name] = arm_summary
        logger.info(f"Arm '{arm_name}' completed. Results saved to: {arm_result_file}")

    # Generate Comparative Report
    comparative_report = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_rounds": num_rounds,
        "local_epochs": local_epochs,
        "model_architecture": model_name.upper(),
        "epistemic_disclaimer": ANOMALY_EPISTEMIC_DISCLAIMER,
        "privacy_notice": FL_PRIVACY_BOUNDARY_DISCLAIMER,
        "comparison": {},
    }

    for arm_name, summary in all_arm_summaries.items():
        rounds_history = summary.get("history", [])
        final_round = rounds_history[-1] if rounds_history else {}
        final_eval = final_round.get("aggregated_evaluate_metrics", {})
        final_fit = final_round.get("aggregated_fit_metrics", {})

        comparative_report["comparison"][arm_name] = {
            "attack_type": summary.get("active_attack", "none"),
            "attackers": summary.get("attacker_clients", []),
            "final_train_loss": final_fit.get("train_loss", None),
            "final_test_loss": final_round.get("client_evaluate_loss", None),
            "final_accuracy": final_eval.get("accuracy", None),
            "final_f1_score": final_eval.get("f1_score", None),
            "final_pr_auc": final_eval.get("pr_auc", None),
            "total_communication_mb": summary.get("total_communication_mb", None),
            "best_f1_score": summary.get("best_f1_score", None),
        }

    report_path = out_dir / "byzantine_comparative_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparative_report, f, indent=2)

    logger.info(f"\n=== Full Comparative Benchmark Complete ===")
    logger.info(f"Summary Report generated at: {report_path}")
    for arm_name, metrics in comparative_report["comparison"].items():
        logger.info(
            f"  [{arm_name}] Attack: {metrics['attack_type']} | F1: {metrics['final_f1_score']} "
            f"| Loss: {metrics['final_test_loss']} | PR-AUC: {metrics['final_pr_auc']}"
        )

    return comparative_report


if __name__ == "__main__":
    run_comparative_benchmark(num_rounds=2, local_epochs=1)
