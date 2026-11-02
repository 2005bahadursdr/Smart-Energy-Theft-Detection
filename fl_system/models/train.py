"""
PyTorch Training Orchestrator for Energy Theft Classification.
Implements local epoch training, loss optimization with class-imbalance pos_weighting,
validation tracking, checkpoint persistence, and reproducible training histories.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl_system.models.evaluate import EvaluationMetrics, evaluate_model
from fl_system.models.model_utils import save_checkpoint, set_reproducible_seed

logger = logging.getLogger("model_train")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu"
) -> float:
    """
    Execute a single training epoch across mini-batches.

    Returns
    -------
    float
        Average loss across all samples in the epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        # Gradient clipping for LSTM stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        batch_size = len(batch_y)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 5,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    pos_weight: float = 5.0, # Addresses ~8.5% theft imbalance
    device: str = "cpu",
    checkpoint_dir: Optional[Union[str, Path]] = None,
    model_name: str = "energy_theft_model",
    seed: int = 42
) -> Tuple[nn.Module, Dict[str, List[float]], Optional[EvaluationMetrics]]:
    """
    Train an Energy Theft neural network with validation tracking and checkpointing.

    Returns
    -------
    Tuple[nn.Module, Dict[str, List[float]], Optional[EvaluationMetrics]]
        (best_model, training_history, best_val_metrics)
    """
    set_reproducible_seed(seed)
    model = model.to(device)

    # Imbalance-weighted binary cross entropy loss
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_pr_auc": []
    }

    best_metric = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    best_metrics_report: Optional[EvaluationMetrics] = None

    ckpt_path = None
    if checkpoint_dir:
        ckpt_path = Path(checkpoint_dir) / f"{model_name}_best.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        history["train_loss"].append(round(train_loss, 4))

        if val_loader is not None and len(val_loader.dataset) > 0:
            val_metrics, _, _ = evaluate_model(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device
            )
            history["val_loss"].append(val_metrics.loss)
            history["val_f1"].append(val_metrics.f1_score)
            history["val_pr_auc"].append(val_metrics.pr_auc)

            # Monitor PR-AUC (ideal for imbalanced fraud/theft evaluation) or F1
            eval_score = val_metrics.pr_auc if val_metrics.pr_auc > 0 else val_metrics.f1_score
            if eval_score > best_metric:
                best_metric = eval_score
                best_weights = copy.deepcopy(model.state_dict())
                best_metrics_report = val_metrics

                if ckpt_path:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics=val_metrics.to_dict(),
                        checkpoint_path=ckpt_path,
                        metadata={"best_eval_score": eval_score, "architecture": model.__class__.__name__}
                    )

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics.loss:.4f} | Val F1: {val_metrics.f1_score:.4f} | "
                f"Val PR-AUC: {val_metrics.pr_auc:.4f}"
            )
        else:
            logger.info(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f}")

    # Load best weights
    if val_loader is not None and best_metrics_report is not None:
        model.load_state_dict(best_weights)

    return model, history, best_metrics_report
