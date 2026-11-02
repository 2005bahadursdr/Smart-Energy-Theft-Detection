"""
Byzantine Attack Manager.
Coordinates malicious substation client designations, attack scheduling,
perturbation dispatching, and post-hoc evaluation ground-truth tracking.

CRITICAL EPISTEMIC & SECURITY PRINCIPLES:
1. An unusual or divergent update is NOT proof of malicious intent. Legitimate
   smart grid domain heterogeneity (e.g., industrial arc furnaces in client_A or
   agricultural pump loads in client_E) naturally yields high-variance parameter updates.
2. Ground-truth attacker identities are recorded exclusively for post-hoc research
   evaluation (calculating True Positive Rate, False Positive Rate, and F1 of defenses).
   The central FL aggregation server and trust detectors MUST NEVER access these ground-truth labels.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import yaml

from byzantine.attacks.base_attack import BaseByzantineAttack
from byzantine.attacks.label_flipping import LabelFlippingAttack
from byzantine.attacks.model_poisoning import ModelPoisoningAttack
from byzantine.attacks.random_update import RandomUpdateAttack
from byzantine.attacks.sign_flipping import SignFlippingAttack
from fl_system.models.dataset import MeterSequenceDataset

logger = logging.getLogger("byzantine_manager")

ANOMALY_EPISTEMIC_DISCLAIMER: str = (
    "RESEARCH & OPERATIONAL NOTICE: An unusual, high-norm, or directionally divergent "
    "gradient update is NOT definitive proof of malicious Byzantine intent. "
    "In smart distribution grids, regional non-IID energy consumption (e.g. continuous industrial "
    "processes in client_A or seasonal agricultural irrigation in client_E) naturally produces "
    "legitimate statistical outliers. Trust scoring systems must balance outlier dampening "
    "with domain-specific variance tolerance."
)


class ByzantineAttackManager:
    """
    Orchestrates Byzantine threat simulation across smart grid substation clients.
    """

    EPISTEMIC_NOTICE = ANOMALY_EPISTEMIC_DISCLAIMER

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = "config/attack_config.yaml",
        attacker_clients: Optional[List[str]] = None,
        malicious_fraction: Optional[float] = None,
        all_client_ids: Optional[List[str]] = None,
        active_attack: str = "sign_flipping",
        attack_rounds: Optional[List[int]] = None,
        seed: int = 42,
    ):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.all_client_ids = all_client_ids or ["client_A", "client_B", "client_C", "client_D", "client_E"]

        # 1. Load YAML Configuration if provided
        self.raw_config: Dict[str, Any] = {}
        if config_path is not None and Path(config_path).exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.raw_config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Could not load attack config from {config_path}: {e}")

        sim_cfg = self.raw_config.get("byzantine_simulation", {})
        attacks_cfg = self.raw_config.get("attacks", {})

        # 2. Determine Attacker Clients
        if attacker_clients is not None:
            self.attacker_clients: List[str] = list(attacker_clients)
        elif malicious_fraction is not None:
            num_attackers = max(1, int(round(len(self.all_client_ids) * malicious_fraction)))
            self.attacker_clients = self.rng.choice(self.all_client_ids, size=num_attackers, replace=False).tolist()
        else:
            self.attacker_clients = sim_cfg.get("attacker_clients", ["client_D"])

        self.attacker_set: Set[str] = set(self.attacker_clients)
        self.active_attack: str = active_attack or sim_cfg.get("active_attack", "sign_flipping")
        self.attack_rounds: Optional[Set[int]] = set(attack_rounds) if attack_rounds is not None else None

        # 3. Instantiate Attack Engines
        self.attack_catalog: Dict[str, BaseByzantineAttack] = {
            "sign_flipping": SignFlippingAttack(
                config=attacks_cfg.get("sign_flipping", {}), seed=seed
            ),
            "model_poisoning": ModelPoisoningAttack(
                config=attacks_cfg.get("model_poisoning", {}), seed=seed
            ),
            "random_update": RandomUpdateAttack(
                config=attacks_cfg.get("random_update", {}), seed=seed
            ),
            "label_flipping": LabelFlippingAttack(
                config=attacks_cfg.get("label_flipping", {}), seed=seed
            ),
        }

        # 4. Strictly Isolated Ground-Truth Log (Accessible ONLY for offline research evaluation)
        self._ground_truth_audit: List[Dict[str, Any]] = []

        logger.info(
            f"Initialized ByzantineAttackManager | Active Attack: {self.active_attack} | "
            f"Attacker Nodes: {self.attacker_clients} ({len(self.attacker_clients)}/{len(self.all_client_ids)}) | "
            f"Attack Rounds: {self.attack_rounds or 'All'}"
        )
        logger.info(self.EPISTEMIC_NOTICE)

    def is_attacker(self, client_id: str, server_round: int) -> bool:
        """
        Check if a given client is scheduled to execute an attack in this round.
        """
        if client_id not in self.attacker_set:
            return False
        if self.attack_rounds is not None and server_round not in self.attack_rounds:
            return False
        return True

    def apply_attack(
        self,
        client_id: str,
        server_round: int,
        global_parameters: List[np.ndarray],
        local_parameters: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Apply attack perturbation if client is an active Byzantine node.
        Leaves honest client parameters strictly untouched.
        """
        if not self.is_attacker(client_id, server_round):
            return local_parameters

        # Select attack vector
        attack_type = self.active_attack
        if attack_type == "mixed":
            # Deterministically cycle or pick an attack for mixed threat mode
            choices = ["sign_flipping", "model_poisoning", "random_update"]
            attack_type = choices[(server_round + hash(client_id)) % len(choices)]

        attack_engine = self.attack_catalog.get(attack_type)
        if attack_engine is None:
            logger.warning(f"Unknown attack '{attack_type}'. Falling back to sign_flipping.")
            attack_engine = self.attack_catalog["sign_flipping"]

        # Perturb parameters
        poisoned_params = attack_engine.perturb_parameters(
            global_parameters=global_parameters,
            local_parameters=local_parameters,
            client_id=client_id,
            server_round=server_round,
        )

        # Calculate deviation metric for audit logging
        honest_norm = float(np.sqrt(sum(np.sum(p**2) for p in local_parameters)))
        poisoned_norm = float(np.sqrt(sum(np.sum(p**2) for p in poisoned_params)))
        param_shift = float(np.sqrt(sum(np.sum((p - h)**2) for p, h in zip(poisoned_params, local_parameters))))

        # Record in strictly isolated ground-truth registry
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server_round": server_round,
            "client_id": client_id,
            "attack_type": attack_type,
            "honest_l2_norm": round(honest_norm, 4),
            "poisoned_l2_norm": round(poisoned_norm, 4),
            "l2_perturbation_shift": round(param_shift, 4),
        }
        self._ground_truth_audit.append(record)

        logger.warning(
            f"[BYZANTINE ATTACK INJECTED] Node '{client_id}' executed '{attack_type}' in Round {server_round} "
            f"| Shift: {param_shift:.4f} | L2 Norm: {honest_norm:.2f} -> {poisoned_norm:.2f}"
        )

        return poisoned_params

    def poison_training_dataset(
        self,
        client_id: str,
        server_round: int,
        dataset: MeterSequenceDataset,
    ) -> MeterSequenceDataset:
        """
        Invert labels if client is an active Byzantine attacker executing label flipping.
        """
        if self.active_attack == "label_flipping" and self.is_attacker(client_id, server_round):
            label_engine: LabelFlippingAttack = self.attack_catalog["label_flipping"]
            poisoned_dataset = label_engine.flip_dataset(dataset)
            logger.warning(
                f"[BYZANTINE LABEL FLIP] Node '{client_id}' inverted {len(dataset)} training labels in Round {server_round}."
            )
            return poisoned_dataset
        return dataset

    # --------------------------------------------------------------------------
    # Evaluation-Only Ground-Truth Accessors
    # --------------------------------------------------------------------------

    def get_ground_truth_labels(self, client_ids: List[str], server_round: int) -> Dict[str, bool]:
        """
        Return ground-truth malicious labels {client_id: is_attacker}.
        
        CRITICAL RESTRICTION:
        This method is provided EXCLUSIVELY for post-hoc experiment evaluation metrics.
        Servers and aggregators MUST NEVER query this method.
        """
        return {cid: self.is_attacker(cid, server_round) for cid in client_ids}

    def export_ground_truth_log(self, output_path: Union[str, Path]) -> Path:
        """Export full attack audit trail to JSON for evaluation benchmarks."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "active_attack": self.active_attack,
            "attacker_clients": self.attacker_clients,
            "total_attacks_injected": len(self._ground_truth_audit),
            "audit_trail": self._ground_truth_audit,
            "epistemic_notice": ANOMALY_EPISTEMIC_DISCLAIMER,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return path
