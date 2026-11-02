"""
fl_system.models package.
Blockchain-Based Byzantine-Resilient Federated Learning for Smart Grid Energy Theft Detection.

Exposes PyTorch time-series architectures (LSTM and CNN-LSTM), leakage-free
dataset windowing and preprocessing pipelines, imbalance-aware training routines,
and comprehensive fraud evaluation metrics.
"""

from fl_system.models.cnn_lstm import EnergyTheftCNNLSTM
from fl_system.models.dataset import (
    EVENT_MAPPING,
    STATUS_MAPPING,
    MeterSequenceDataset,
    PreprocessingState,
    TimeSeriesPreprocessor,
    create_dataloaders,
    create_sequence_windows,
    extract_multivariate_features,
    prepare_leakage_free_datasets,
    split_data_leakage_free,
)
from fl_system.models.evaluate import (
    ETHICAL_VALIDATION_DISCLAIMER,
    EvaluationMetrics,
    calculate_binary_metrics,
    evaluate_model,
)
from fl_system.models.lstm import EnergyTheftLSTM
from fl_system.models.model_utils import (
    compute_gradient_norm,
    count_parameters,
    flatten_gradients,
    get_model,
    get_parameters_as_numpy,
    load_checkpoint,
    save_checkpoint,
    set_parameters_from_numpy,
    set_reproducible_seed,
)
from fl_system.models.train import (
    train_model,
    train_one_epoch,
)

__all__ = [
    # Models
    "EnergyTheftLSTM",
    "EnergyTheftCNNLSTM",
    # Dataset & Preprocessing
    "MeterSequenceDataset",
    "TimeSeriesPreprocessor",
    "PreprocessingState",
    "create_sequence_windows",
    "split_data_leakage_free",
    "prepare_leakage_free_datasets",
    "create_dataloaders",
    "extract_multivariate_features",
    "STATUS_MAPPING",
    "EVENT_MAPPING",
    # Evaluation
    "EvaluationMetrics",
    "calculate_binary_metrics",
    "evaluate_model",
    "ETHICAL_VALIDATION_DISCLAIMER",
    # Utilities & Checkpoints
    "get_model",
    "get_parameters_as_numpy",
    "set_parameters_from_numpy",
    "flatten_gradients",
    "compute_gradient_norm",
    "count_parameters",
    "save_checkpoint",
    "load_checkpoint",
    "set_reproducible_seed",
    # Training
    "train_model",
    "train_one_epoch",
]
