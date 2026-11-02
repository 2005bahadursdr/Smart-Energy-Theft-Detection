"""
Client-Update Validation Evaluator for Trust Scoring.
Evaluates candidate client updates against a server-side benchmark DataLoader.

Guiding Principles:
1. "Similarity alone is insufficient": Colluding attacks or subtle perturbations can match
   the empirical reference direction while still degrading classification performance.
2. "Validation can be noisy": Server benchmark sets may have limited batches or slight distribution
   shifts; a configurable loss tolerance margin ensures honest non-IID updates are not penalized.
3. Numerical robustness: Explicit handling of NaN/Inf losses, empty loaders, and divergence.
"""

import copy
import logging
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl_system.models.model_utils import set_parameters_from_numpy

logger = logging.getLogger("trust_validation")


class ClientUpdateValidationEvaluator:
    """
    Evaluates individual client updates on a server-side validation dataset.
    """

    def __init__(
        self,
        reference_model: nn.Module,
        benchmark_loader: Optional[DataLoader] = None,
        loss_tolerance: float = 0.05,
        temperature: float = 0.10,
        nan_loss_penalty: float = 0.0,
        missing_benchmark_score: float = 0.70,
        pos_weight: float = 5.0,
        device: str = "cpu",
    ):
        self.reference_model = reference_model
        self.benchmark_loader = benchmark_loader
        self.loss_tolerance = loss_tolerance
        self.temperature = max(temperature, 1e-4)
        self.nan_loss_penalty = nan_loss_penalty
        self.missing_benchmark_score = missing_benchmark_score
        self.pos_weight = pos_weight
        self.device = torch.device(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=self.device))

    def evaluate_model_loss(self, model: nn.Module) -> float:
        """
        Compute average BCE loss of a model on the benchmark dataset.
        Returns float, or np.nan if loss diverges or computation fails.
        """
        if self.benchmark_loader is None:
            return 0.0

        model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for X_b, y_b in self.benchmark_loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device).float()
                try:
                    logits = model(X_b)
                    if logits.dim() == 1 and y_b.dim() > 1:
                        y_loss = y_b.squeeze(-1)
                    elif logits.dim() > 1 and y_b.dim() == 1:
                        y_loss = y_b.unsqueeze(-1)
                    else:
                        y_loss = y_b

                    loss = self.criterion(logits, y_loss)
                    loss_val = float(loss.item())

                    if np.isnan(loss_val) or np.isinf(loss_val):
                        return float(np.nan)

                    total_loss += loss_val * len(y_b)
                    total_samples += len(y_b)
                except Exception as e:
                    logger.warning(f"Validation loss calculation error: {e}")
                    return float(np.nan)

        if total_samples == 0:
            return 0.0
        return total_loss / total_samples

    def evaluate_client_updates(
        self,
        current_global_parameters: List[np.ndarray],
        client_updates: Dict[str, List[np.ndarray]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate each client's candidate model on the server validation dataset.

        Args:
            current_global_parameters: List of numpy arrays representing current global weights.
            client_updates: Dict mapping client_id to candidate parameters (w_global + Delta w).

        Returns:
            Dict mapping client_id to:
            - 'candidate_loss': float or NaN
            - 'loss_delta': L_global - L_candidate (positive is improvement)
            - 'validation_score': normalized score in [0.0, 1.0]
        """
        # If no benchmark loader is available, return default neutral score
        if self.benchmark_loader is None:
            return {
                cid: {
                    "candidate_loss": 0.0,
                    "loss_delta": 0.0,
                    "validation_score": self.missing_benchmark_score,
                }
                for cid in client_updates
            }

        # 1. Evaluate baseline global model
        eval_model = copy.deepcopy(self.reference_model).to(self.device)
        set_parameters_from_numpy(eval_model, current_global_parameters)
        base_loss = self.evaluate_model_loss(eval_model)

        results = {}
        for cid, candidate_params in client_updates.items():
            # Check for NaN / Inf directly in candidate parameters
            has_nan_param = any(np.isnan(p).any() or np.isinf(p).any() for p in candidate_params)
            if has_nan_param:
                results[cid] = {
                    "candidate_loss": float("nan"),
                    "loss_delta": -999.0,
                    "validation_score": self.nan_loss_penalty,
                }
                continue

            # Load candidate weights into evaluation model
            set_parameters_from_numpy(eval_model, candidate_params)
            cand_loss = self.evaluate_model_loss(eval_model)

            if np.isnan(cand_loss) or np.isinf(cand_loss):
                results[cid] = {
                    "candidate_loss": float("nan"),
                    "loss_delta": -999.0,
                    "validation_score": self.nan_loss_penalty,
                }
                continue

            loss_delta = base_loss - cand_loss # Positive = improved loss

            # Sigmoid mapping centered with loss_tolerance margin
            # score = 1 / (1 + exp(-(loss_delta + loss_tolerance) / temperature))
            scaled_delta = (loss_delta + self.loss_tolerance) / self.temperature
            # Prevent overflow in exp
            scaled_delta = float(np.clip(scaled_delta, -20.0, 20.0))
            score = float(1.0 / (1.0 + np.exp(-scaled_delta)))
            score = float(np.clip(score, 0.0, 1.0))

            results[cid] = {
                "candidate_loss": round(float(cand_loss), 4),
                "loss_delta": round(float(loss_delta), 4),
                "validation_score": round(score, 4),
            }

        return results
