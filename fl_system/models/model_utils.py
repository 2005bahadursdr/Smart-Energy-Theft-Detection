"""
Federated Learning and Deep Learning Model Utilities.
Provides model factory, weight extraction/injection (for Flower FedAvg aggregation),
gradient flattening (for Byzantine trust & similarity detection), checkpoint management,
and reproducibility controls.
"""

from datetime import datetime
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from fl_system.models.cnn_lstm import EnergyTheftCNNLSTM
from fl_system.models.lstm import EnergyTheftLSTM

logger = logging.getLogger("model_utils")


def set_reproducible_seed(seed: int = 42) -> None:
    """
    Configure global deterministic seeds across PyTorch, NumPy, and Python runtime.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model(
    model_name: str = "cnn_lstm",
    input_size: int = 11,
    **kwargs
) -> nn.Module:
    """
    Factory creating deep learning time-series architectures.

    Parameters
    ----------
    model_name : str
        Model architecture identifier ('lstm' or 'cnn_lstm').
    input_size : int
        Feature dimensionality (1 for univariate, 11 for multivariate).

    Returns
    -------
    nn.Module
        Configured PyTorch neural network.
    """
    name_clean = model_name.lower().replace("-", "_")
    if name_clean in ("lstm", "energytheftlstm"):
        return EnergyTheftLSTM(
            input_size=input_size,
            hidden_size=kwargs.get("hidden_size", 64),
            num_layers=kwargs.get("num_layers", 2),
            bidirectional=kwargs.get("bidirectional", False),
            dropout=kwargs.get("dropout", 0.20),
            fc_dims=kwargs.get("fc_dims", [32]),
            output_size=kwargs.get("output_size", 1)
        )
    elif name_clean in ("cnn_lstm", "energytheftcnnlstm", "hybrid"):
        return EnergyTheftCNNLSTM(
            input_size=input_size,
            conv_filters=kwargs.get("conv_filters", [32, 64]),
            kernel_size=kwargs.get("kernel_size", 3),
            lstm_hidden_size=kwargs.get("lstm_hidden_size", 64),
            lstm_layers=kwargs.get("lstm_layers", 1),
            dense_units=kwargs.get("dense_units", [32]),
            dropout=kwargs.get("dropout", 0.25),
            output_size=kwargs.get("output_size", 1)
        )
    else:
        raise ValueError(f"Unknown model architecture '{model_name}'. Choose 'lstm' or 'cnn_lstm'.")


def get_parameters_as_numpy(model: nn.Module) -> List[np.ndarray]:
    """
    Extract trainable model parameters as a list of NumPy arrays.
    Compatible with Flower (flwr) NumPy client parameter representations.
    """
    return [val.cpu().numpy().copy() for _, val in model.state_dict().items()]


def set_parameters_from_numpy(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Update model weights from a list of NumPy arrays (used after federated aggregation).
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def flatten_gradients(model: nn.Module) -> np.ndarray:
    """
    Extract and concatenate all parameter gradients into a flat 1D vector.
    Used by Byzantine defense mechanisms for cosine direction and Euclidean norm scoring.
    """
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1).cpu().numpy())
        else:
            grads.append(np.zeros(param.numel(), dtype=np.float32))

    if not grads:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(grads)


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute global Euclidean (L2) norm of model parameter gradients."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_sq += float(param_norm.item() ** 2)
    return float(np.sqrt(total_norm_sq))


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, Any],
    checkpoint_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save PyTorch checkpoint bundle with metadata, training metrics, and weights.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_class": model.__class__.__name__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
        "metadata": metadata or {},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to: {path}")
    return path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    """
    Restore model and optimizer state from checkpoint.

    Returns
    -------
    Tuple[int, Dict[str, Any], Dict[str, Any]]
        (epoch, metrics, metadata)
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    metadata = checkpoint.get("metadata", {})
    return epoch, metrics, metadata
