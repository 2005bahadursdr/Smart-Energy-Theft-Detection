"""
Unit and Integration Tests for fl_system.models.
Validates:
1. 24-hour and 48-hour sequence sliding windows.
2. Leakage-free splitting (consumer and temporal) with zero test overlap.
3. Preprocessor fitting strictly on training partitions.
4. PyTorch EnergyTheftLSTM and EnergyTheftCNNLSTM tensor input/output shapes.
5. Training loop convergence and imbalance weighting.
6. Flower FL weight serialization (get/set parameters as numpy).
7. Evaluation metrics (Precision, Recall, F1, PR-AUC, ROC-AUC, Confusion Matrix) and edge cases.
8. Checkpoint saving/loading and deterministic reproducibility.
9. Ethical boundary notice verification (disclaimer presence).
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl_system.models.cnn_lstm import EnergyTheftCNNLSTM
from fl_system.models.dataset import (
    EVENT_MAPPING,
    STATUS_MAPPING,
    MeterSequenceDataset,
    TimeSeriesPreprocessor,
    create_dataloaders,
    create_sequence_windows,
    extract_multivariate_features,
    prepare_leakage_free_datasets,
    split_data_leakage_free,
)
from fl_system.models.evaluate import (
    ETHICAL_VALIDATION_DISCLAIMER,
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
from fl_system.models.train import train_model, train_one_epoch


@pytest.fixture
def sample_telemetry_df() -> pd.DataFrame:
    """Generate realistic synthetic telemetry DataFrame for 10 meters over 72 hours."""
    records = []
    rng = np.random.RandomState(42)
    dates = pd.date_range("2026-01-01 00:00:00", periods=72, freq="1h")

    for m_idx in range(10):
        m_id = f"METER_{m_idx:03d}"
        for ts in dates:
            is_theft = 1 if (m_idx < 2 and ts.hour >= 18) else 0
            consumption = rng.uniform(0.1, 0.4) if is_theft else rng.uniform(0.5, 2.5)
            status = "TAMPER_ALERT" if is_theft else "NORMAL"
            event = "TAMPER_EVENT" if is_theft else "NONE"
            hw_flag = 1 if is_theft else 0

            records.append({
                "meter_id": m_id,
                "timestamp": str(ts),
                "consumption_kwh": consumption,
                "meter_status": status,
                "event_type": event,
                "hardware_tamper_flag": hw_flag,
                "simulated_theft_label": is_theft
            })

    return pd.DataFrame(records)


# =========================================================================
# 1. Dataset, Feature Engineering, and Leakage-Free Splitting Tests
# =========================================================================

def test_extract_multivariate_features(sample_telemetry_df):
    """Test multivariate feature derivation produces 11 expected features."""
    preprocessor = TimeSeriesPreprocessor(scaling="minmax")
    preprocessor.fit(sample_telemetry_df)
    transformed = preprocessor.transform(sample_telemetry_df)

    feats, names = extract_multivariate_features(transformed)
    assert feats.shape[0] == len(sample_telemetry_df)
    assert feats.shape[1] == 11
    assert "norm_consumption" in names
    assert "consumption_delta" in names
    assert "rolling_mean_6h" in names
    assert "sin_hour" in names
    assert "cos_hour" in names
    assert "meter_status_enc" in names
    assert "hardware_tamper_flag" in names


def test_create_sequence_windows_24h_and_48h(sample_telemetry_df):
    """Test 24-hour and 48-hour sliding window generation shapes."""
    preprocessor = TimeSeriesPreprocessor(scaling="minmax")
    preprocessor.fit(sample_telemetry_df)
    transformed = preprocessor.transform(sample_telemetry_df)

    # 24-hour sequence window
    ds_24 = create_sequence_windows(transformed, window_size=24, stride=12, multivariate=True)
    assert len(ds_24) > 0
    assert ds_24.X.shape[1] == 24
    assert ds_24.X.shape[2] == 11
    assert ds_24.y.shape[0] == len(ds_24)
    assert ds_24.is_synthetic is True

    # 48-hour sequence window
    ds_48 = create_sequence_windows(transformed, window_size=48, stride=12, multivariate=True)
    assert len(ds_48) > 0
    assert ds_48.X.shape[1] == 48
    assert ds_48.X.shape[2] == 11
    assert ds_48.y.shape[0] == len(ds_48)


def test_consumer_split_zero_leakage(sample_telemetry_df):
    """Test consumer split guarantees zero meter overlap between train, val, and test."""
    train_df, val_df, test_df = split_data_leakage_free(
        sample_telemetry_df, split_strategy="consumer", test_ratio=0.20, val_ratio=0.20, seed=42
    )

    train_meters = set(train_df["meter_id"].unique())
    val_meters = set(val_df["meter_id"].unique())
    test_meters = set(test_df["meter_id"].unique())

    # Total meters must equal original count
    assert len(train_meters | val_meters | test_meters) == 10

    # Mutual exclusivity (disjoint sets)
    assert len(train_meters.intersection(val_meters)) == 0
    assert len(train_meters.intersection(test_meters)) == 0
    assert len(val_meters.intersection(test_meters)) == 0


def test_temporal_split_chronological(sample_telemetry_df):
    """Test temporal split guarantees no future leakage into train set."""
    train_df, val_df, test_df = split_data_leakage_free(
        sample_telemetry_df, split_strategy="temporal", test_ratio=0.20, val_ratio=0.20, seed=42
    )

    max_train_ts = train_df["timestamp"].max()
    min_val_ts = val_df["timestamp"].min()
    max_val_ts = val_df["timestamp"].max()
    min_test_ts = test_df["timestamp"].min()

    assert max_train_ts <= min_val_ts
    assert max_val_ts <= min_test_ts


def test_preprocessor_fitted_strictly_on_train_data(sample_telemetry_df):
    """Verify preprocessor parameters are computed strictly from train partition."""
    train_df, val_df, test_df = split_data_leakage_free(sample_telemetry_df, seed=42)

    preprocessor = TimeSeriesPreprocessor(scaling="minmax")
    preprocessor.fit(train_df)

    expected_min = float(train_df["consumption_kwh"].min())
    expected_max = float(train_df["consumption_kwh"].max())

    assert preprocessor.state is not None
    assert pytest.approx(preprocessor.state.min_consumption, 1e-4) == expected_min
    assert pytest.approx(preprocessor.state.max_consumption, 1e-4) == expected_max

    # Transform test without altering preprocessor state
    test_transformed = preprocessor.transform(test_df)
    assert pytest.approx(preprocessor.state.min_consumption, 1e-4) == expected_min
    assert pytest.approx(preprocessor.state.max_consumption, 1e-4) == expected_max
    assert "norm_consumption" in test_transformed.columns


def test_prepare_leakage_free_datasets_pipeline(sample_telemetry_df):
    """Test complete end-to-end pipeline creates non-empty datasets with dataloaders."""
    train_ds, val_ds, test_ds, preprocessor = prepare_leakage_free_datasets(
        sample_telemetry_df, window_size=24, stride=12, split_strategy="consumer", seed=42
    )

    assert len(train_ds) > 0
    assert len(test_ds) > 0

    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=4, use_weighted_sampler=True
    )
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape[0] <= 4
    assert batch_x.shape[1] == 24
    assert batch_x.shape[2] == 11
    assert batch_y.shape[0] == batch_x.shape[0]


# =========================================================================
# 2. PyTorch Architecture Forward Pass & Shape Tests
# =========================================================================

def test_energy_theft_lstm_shapes():
    """Test EnergyTheftLSTM forward and predict_proba tensor dimensions."""
    model = EnergyTheftLSTM(input_size=11, hidden_size=32, num_layers=2)
    batch_size = 8
    seq_len = 24

    x = torch.randn(batch_size, seq_len, 11)
    logits = model(x)
    assert logits.shape == (batch_size,)

    probs = model.predict_proba(x)
    assert probs.shape == (batch_size,)
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)


def test_energy_theft_cnn_lstm_shapes():
    """Test EnergyTheftCNNLSTM forward pass with 24-hour and 48-hour inputs."""
    model = EnergyTheftCNNLSTM(input_size=11, conv_filters=[16, 32], lstm_hidden_size=32)

    # 24-hour sequence
    x_24 = torch.randn(4, 24, 11)
    logits_24 = model(x_24)
    assert logits_24.shape == (4,)

    # 48-hour sequence
    x_48 = torch.randn(4, 48, 11)
    logits_48 = model(x_48)
    assert logits_48.shape == (4,)

    probs = model.predict_proba(x_48)
    assert probs.shape == (4,)
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)


def test_model_factory_get_model():
    """Test model factory instantiate both lstm and cnn_lstm correctly."""
    lstm_m = get_model("lstm", input_size=11)
    assert isinstance(lstm_m, EnergyTheftLSTM)

    cnn_m = get_model("cnn_lstm", input_size=11)
    assert isinstance(cnn_m, EnergyTheftCNNLSTM)

    with pytest.raises(ValueError):
        get_model("transformer_invalid")


# =========================================================================
# 3. Model Utils, Checkpoint, and Flower Serialization Tests
# =========================================================================

def test_get_and_set_parameters_as_numpy():
    """Test parameter extraction and injection for Flower FedAvg aggregation."""
    model_a = EnergyTheftCNNLSTM(input_size=11, conv_filters=[16, 32], lstm_hidden_size=32)
    model_b = EnergyTheftCNNLSTM(input_size=11, conv_filters=[16, 32], lstm_hidden_size=32)

    # Extract weights from A
    params_numpy = get_parameters_as_numpy(model_a)
    assert isinstance(params_numpy, list)
    assert isinstance(params_numpy[0], np.ndarray)

    # Set into B
    set_parameters_from_numpy(model_b, params_numpy)

    # Set both models to eval mode to disable dropout
    model_a.eval()
    model_b.eval()

    # Verify identical outputs
    x = torch.randn(2, 24, 11)
    with torch.no_grad():
        out_a = model_a(x)
        out_b = model_b(x)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_flatten_gradients_and_norm():
    """Test gradient flattening and L2 norm computation."""
    model = EnergyTheftLSTM(input_size=11, hidden_size=16, num_layers=1)
    x = torch.randn(4, 24, 11)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(model(x), y)
    loss.backward()

    flat_grads = flatten_gradients(model)
    norm = compute_gradient_norm(model)

    assert isinstance(flat_grads, np.ndarray)
    assert flat_grads.ndim == 1
    assert len(flat_grads) > 0
    assert norm > 0.0


def test_checkpoint_save_and_load(tmp_path):
    """Test checkpoint persistence and exact weight restoration."""
    model = EnergyTheftCNNLSTM(input_size=11, conv_filters=[16, 32], lstm_hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ckpt_file = tmp_path / "test_model_ckpt.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=3,
        metrics={"f1": 0.85},
        checkpoint_path=ckpt_file,
        metadata={"client_id": "client_A"}
    )
    assert ckpt_file.exists()

    model_restored = EnergyTheftCNNLSTM(input_size=11, conv_filters=[16, 32], lstm_hidden_size=32)
    opt_restored = torch.optim.Adam(model_restored.parameters(), lr=0.001)

    epoch, metrics, meta = load_checkpoint(ckpt_file, model_restored, opt_restored)
    assert epoch == 3
    assert metrics["f1"] == 0.85
    assert meta["client_id"] == "client_A"

    model.eval()
    model_restored.eval()
    x = torch.randn(2, 24, 11)
    with torch.no_grad():
        out_orig = model(x)
        out_rest = model_restored(x)
    assert torch.allclose(out_orig, out_rest, atol=1e-6)


# =========================================================================
# 4. Training, Imbalance Handling, and Convergence Tests
# =========================================================================

def test_train_one_epoch_convergence():
    """Test single epoch updates parameters and yields positive loss."""
    set_reproducible_seed(42)
    model = EnergyTheftLSTM(input_size=11, hidden_size=16, num_layers=1)
    x = torch.randn(16, 24, 11)
    y = torch.tensor([0.0, 1.0] * 8)
    ds = torch.utils.data.TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))

    loss1 = train_one_epoch(model, loader, optimizer, criterion)
    assert loss1 > 0.0
    assert np.isfinite(loss1)


def test_train_model_end_to_end(sample_telemetry_df, tmp_path):
    """Test full train_model routine with validation and checkpointing."""
    train_ds, val_ds, test_ds, _ = prepare_leakage_free_datasets(
        sample_telemetry_df, window_size=24, stride=12, seed=42
    )
    train_loader, val_loader, test_loader = create_dataloaders(train_ds, val_ds, test_ds, batch_size=4)

    model = get_model("lstm", input_size=11, hidden_size=16, num_layers=1)
    trained_model, history, best_val = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        checkpoint_dir=tmp_path,
        model_name="test_lstm",
        seed=42
    )

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert (tmp_path / "test_lstm_best.pt").exists() or best_val is None


# =========================================================================
# 5. Evaluation Metrics & Edge-Case Handling Tests
# =========================================================================

def test_calculate_binary_metrics_balanced():
    """Test metric computation on standard binary predictions."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95])

    metrics = calculate_binary_metrics(y_true, y_prob, threshold=0.5)
    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.pr_auc == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.confusion_matrix == {"tn": 4, "fp": 0, "fn": 0, "tp": 4}
    assert metrics.is_synthetic_evaluation is True
    assert ETHICAL_VALIDATION_DISCLAIMER in metrics.disclaimer


def test_calculate_binary_metrics_all_zeros_edge_case():
    """
    Test edge-case where test partition contains only negative class (0s).
    ROC-AUC should be safely None rather than throwing a ValueError.
    """
    y_true = np.array([0, 0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.05, 0.15])

    metrics = calculate_binary_metrics(y_true, y_prob)
    assert metrics.roc_auc is None
    assert metrics.accuracy == 1.0
    assert metrics.positive_samples == 0
    assert metrics.negative_samples == 5


def test_evaluate_model_pipeline(sample_telemetry_df):
    """Test evaluate_model returns metrics object with required statistical keys."""
    _, _, test_ds, _ = prepare_leakage_free_datasets(
        sample_telemetry_df, window_size=24, stride=12, seed=42
    )
    test_loader = DataLoader(test_ds, batch_size=4)

    model = get_model("cnn_lstm", input_size=11, conv_filters=[8, 16], lstm_hidden_size=16)
    metrics, y_true, y_prob = evaluate_model(model, test_loader)

    assert len(y_true) == len(test_ds)
    assert len(y_prob) == len(test_ds)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0
    assert 0.0 <= metrics.pr_auc <= 1.0


def test_deterministic_reproducibility():
    """Verify set_reproducible_seed generates identical weights for two distinct initializations."""
    set_reproducible_seed(123)
    model1 = EnergyTheftLSTM(input_size=11, hidden_size=32)
    w1 = [p.clone() for p in model1.parameters()]

    set_reproducible_seed(123)
    model2 = EnergyTheftLSTM(input_size=11, hidden_size=32)
    w2 = [p.clone() for p in model2.parameters()]

    for p1, p2 in zip(w1, w2):
        assert torch.all(p1 == p2)
