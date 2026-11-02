"""
Model Evaluation and Metrics Module.
Calculates Precision, Recall, F1-Score, PR-AUC, ROC-AUC (when defined),
Accuracy, and Confusion Matrix for energy theft classification.
Strictly distinguishes simulated ground truth and documents real-world validation boundaries.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("model_evaluate")

ETHICAL_VALIDATION_DISCLAIMER = (
    "RESEARCH VALIDATION NOTICE: This performance evaluation is conducted using synthetic smart meter "
    "simulation telemetry and/or the State Grid Corporation of China (SGCC) benchmark dataset. "
    "It does NOT constitute verified real-world Indian utility theft validation without access to "
    "verified, legally documented ground-truth data from Indian distribution companies (DISCOMs)."
)


@dataclass
class EvaluationMetrics:
    """Standard evaluation metrics for binary energy theft classification."""
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    pr_auc: float
    roc_auc: Optional[float]
    confusion_matrix: Dict[str, int]
    total_samples: int
    positive_samples: int
    negative_samples: int
    decision_threshold: float
    is_synthetic_evaluation: bool
    dataset_origin: str
    disclaimer: str = ETHICAL_VALIDATION_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    loss: float = 0.0,
    threshold: float = 0.50,
    is_synthetic: bool = True,
    dataset_origin: str = "Synthetic Smart Meter Simulator"
) -> EvaluationMetrics:
    """
    Computes statistical evaluation metrics with robust edge-case handling.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    total_samples = len(y_true)
    pos_samples = int(np.sum(y_true))
    neg_samples = total_samples - pos_samples

    # Basic classification metrics
    acc = float(accuracy_score(y_true, y_pred)) if total_samples > 0 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0.0))
    rec = float(recall_score(y_true, y_pred, zero_division=0.0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0.0))

    # Confusion matrix
    if total_samples > 0:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    cm_dict = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    # PR-AUC (Precision-Recall Area Under Curve: average precision)
    try:
        pr_auc = float(average_precision_score(y_true, y_prob)) if pos_samples > 0 else 0.0
    except Exception:
        pr_auc = 0.0

    # ROC-AUC (Defined ONLY when both classes 0 and 1 are present)
    roc_auc = None
    if pos_samples > 0 and neg_samples > 0:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception as e:
            logger.debug(f"ROC-AUC calculation exception: {e}")
            roc_auc = None

    return EvaluationMetrics(
        loss=round(loss, 4),
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1_score=round(f1, 4),
        pr_auc=round(pr_auc, 4),
        roc_auc=round(roc_auc, 4) if roc_auc is not None else None,
        confusion_matrix=cm_dict,
        total_samples=total_samples,
        positive_samples=pos_samples,
        negative_samples=neg_samples,
        decision_threshold=threshold,
        is_synthetic_evaluation=is_synthetic,
        dataset_origin=dataset_origin
    )


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: str = "cpu",
    threshold: float = 0.50,
    is_synthetic: bool = True,
    dataset_origin: str = "Synthetic Smart Meter Simulator"
) -> Tuple[EvaluationMetrics, np.ndarray, np.ndarray]:
    """
    Evaluates a PyTorch model on a DataLoader.

    Returns
    -------
    Tuple[EvaluationMetrics, np.ndarray, np.ndarray]
        (metrics_report, true_labels, predicted_probabilities)
    """
    model.eval()
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_y_true: List[np.ndarray] = []
    all_y_prob: List[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.item()) * len(batch_y)

            probs = torch.sigmoid(logits)
            all_y_true.append(batch_y.cpu().numpy())
            all_y_prob.append(probs.cpu().numpy())

    if not all_y_true:
        empty_metrics = calculate_binary_metrics(
            y_true=np.zeros(0), y_prob=np.zeros(0), is_synthetic=is_synthetic, dataset_origin=dataset_origin
        )
        return empty_metrics, np.zeros(0), np.zeros(0)

    y_true_arr = np.concatenate(all_y_true)
    y_prob_arr = np.concatenate(all_y_prob)
    mean_loss = total_loss / max(1, len(y_true_arr))

    metrics = calculate_binary_metrics(
        y_true=y_true_arr,
        y_prob=y_prob_arr,
        loss=mean_loss,
        threshold=threshold,
        is_synthetic=is_synthetic,
        dataset_origin=dataset_origin
    )

    return metrics, y_true_arr, y_prob_arr
