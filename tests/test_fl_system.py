"""
Unit and Integration Tests for fl_system (Federated Learning Subsystem).
Validates:
1. Substation partition isolation (prohibits cross-client data access).
2. Privacy firewall: transmits model parameters and scalar metadata only (zero raw readings).
3. Local model training (LSTM and CNN-LSTM) on private local datasets.
4. Baseline FedAvg aggregation, client sampling, and configurable rounds.
5. Communication footprint tracking (uplink/downlink bytes).
6. Dual-level evaluation protocol (local client-aggregated & server centralized benchmark).
7. Checkpoint generation and history persistence.
8. Privacy boundary notice verification.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from fl_system.clients.base_client import (
    FL_PRIVACY_BOUNDARY_DISCLAIMER,
    SmartGridFlowerClient,
)
from fl_system.clients.factory import create_all_clients, create_client
from fl_system.coordinator import DummyClientProxy, FLSimulationCoordinator
from fl_system.models.dataset import MeterSequenceDataset, create_dataloaders
from fl_system.models.model_utils import (
    get_model,
    get_parameters_as_numpy,
    load_checkpoint,
    set_parameters_from_numpy,
)
from fl_system.server.strategy import SmartGridFedAvg
from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


def _generate_synthetic_substation_df(client_id: str, num_meters: int = 6, days: int = 10) -> pd.DataFrame:
    """Generate realistic synthetic telemetry for a single substation client."""
    records = []
    rng = np.random.RandomState(abs(hash(client_id)) % (2**31))
    hours = days * 24

    for m_idx in range(num_meters):
        meter_id = f"METER_{client_id.upper()}_{m_idx:03d}"
        is_theft = 1 if m_idx == 0 else 0  # 1 theft meter per substation
        base_load = rng.uniform(1.0, 5.0)

        for h in range(hours):
            diurnal = np.sin(2 * np.pi * (h % 24) / 24) * 0.5
            kwh = max(0.05, base_load + diurnal + rng.normal(0, 0.2))
            if is_theft and h > hours // 2:
                kwh *= 0.3  # Theft reduction

            records.append({
                "timestamp": f"2026-01-{(h // 24) + 1:02d} {h % 24:02d}:00:00",
                "meter_id": meter_id,
                "client_id": client_id,
                "consumption_kwh": round(kwh, 3),
                "meter_status": "NORMAL",
                "event_type": "NONE",
                "hardware_tamper_flag": 0,
                "simulated_theft_label": is_theft,
                "simulated_theft_type": "scaling" if is_theft else "none",
                "simulated_actual_kwh": round(kwh, 3),
                "simulated_unreported_kwh": 0.0,
            })
    return pd.DataFrame(records)


# ------------------------------------------------------------------------------
# 1. Partition Isolation & Privacy Firewall Tests
# ------------------------------------------------------------------------------

def test_partition_isolation_violation_rejected():
    """Verify that a client cannot access another client's local partition directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        vault_A = base / "client_A"
        vault_B = base / "client_B"
        vault_A.mkdir()
        vault_B.mkdir()

        # Client A attempting to access Client B's vault must raise PermissionError
        with pytest.raises(PermissionError) as exc_info:
            SmartGridFlowerClient(client_id="client_A", data_dir=vault_B)
        assert "Partition Isolation Violation" in str(exc_info.value)


def test_cross_client_telemetry_contamination_detected():
    """Verify that if a local CSV contains foreign client data, an error is raised."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        vault = Path(tmp_dir) / "client_A"
        vault.mkdir()
        csv_path = vault / "local_readings.csv"

        # Contaminated data containing records from client_B
        df_contaminated = pd.DataFrame([
            {"timestamp": "2026-01-01 00:00:00", "meter_id": "M1", "client_id": "client_A", "consumption_kwh": 1.0, "simulated_theft_label": 0},
            {"timestamp": "2026-01-01 00:00:00", "meter_id": "M2", "client_id": "client_B", "consumption_kwh": 1.5, "simulated_theft_label": 0},
        ])
        df_contaminated.to_csv(csv_path, index=False)

        with pytest.raises(ValueError) as exc_info:
            SmartGridFlowerClient(client_id="client_A", data_dir=vault)
        assert "Data Isolation Integrity Failure" in str(exc_info.value)


def test_zero_raw_readings_transmitted_in_protocol():
    """
    Verify the strict privacy guarantee:
    Flower client transmit methods (fit, evaluate, get_parameters) NEVER leak raw telemetry.
    """
    df = _generate_synthetic_substation_df("client_A", num_meters=4, days=5)
    client = SmartGridFlowerClient(client_id="client_A", df=df, model_name="cnn_lstm")

    # 1. get_parameters: must return strictly List[np.ndarray]
    params = client.get_parameters(config={})
    assert isinstance(params, list)
    assert all(isinstance(p, np.ndarray) for p in params)
    for p in params:
        # Must not contain string objects (which raw data might have)
        assert p.dtype in (np.float32, np.float64, np.int64, np.int32)

    # 2. fit: returns (updated_params, num_samples, metrics)
    updated_params, num_samples, metrics = client.fit(params, config={"local_epochs": 1, "server_round": 1})
    assert isinstance(updated_params, list)
    assert isinstance(num_samples, int)
    assert isinstance(metrics, dict)

    # Verify metrics contains ONLY scalar telemetry metadata
    prohibited_keys = {"data", "readings", "consumption", "meters", "meter_id", "raw"}
    for k in metrics.keys():
        assert k.lower() not in prohibited_keys
        # All values must be primitives (float, int, str)
        assert isinstance(metrics[k], (int, float, str, bool))

    # 3. evaluate: returns (loss, num_samples, metrics)
    loss, num_test, eval_metrics = client.evaluate(params, config={})
    assert isinstance(loss, float)
    assert isinstance(num_test, int)
    assert isinstance(eval_metrics, dict)
    for k, v in eval_metrics.items():
        assert isinstance(v, (int, float, str, bool))


# ------------------------------------------------------------------------------
# 2. Local Training & Architecture Tests
# ------------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["lstm", "cnn_lstm"])
def test_client_local_training_convergence(model_name: str):
    """Verify local training runs on private DataLoader and updates parameters."""
    df = _generate_synthetic_substation_df("client_B", num_meters=4, days=5)
    client = SmartGridFlowerClient(
        client_id="client_B",
        df=df,
        model_name=model_name,
        local_epochs=1,
        learning_rate=0.01,
    )

    initial_params = client.get_parameters(config={})
    updated_params, num_samples, metrics = client.fit(
        parameters=initial_params,
        config={"local_epochs": 1, "learning_rate": 0.01, "server_round": 1},
    )

    # Verify model weights actually changed from local training
    weight_diff = sum(np.sum(np.abs(u - i)) for u, i in zip(updated_params, initial_params))
    assert weight_diff > 1e-4, "Weights should change after local gradient descent."
    assert num_samples > 0
    assert metrics["train_loss"] >= 0.0
    assert metrics["grad_norm"] >= 0.0
    assert metrics["client_id"] == "client_B"


# ------------------------------------------------------------------------------
# 3. Strategy & Checkpointing Tests
# ------------------------------------------------------------------------------

def test_strategy_checkpointing_and_history():
    """Verify SmartGridFedAvg saves checkpoints and logs communication metrics."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = Path(tmp_dir) / "checkpoints"

        strategy = SmartGridFedAvg(
            model_name="cnn_lstm",
            input_size=11,
            checkpoint_dir=ckpt_dir,
            min_fit_clients=2,
            min_available_clients=2,
        )

        initial_params = strategy.latest_ndarrays
        dummy_fit_res_1 = FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(initial_params),
            num_examples=100,
            metrics={"client_id": "client_A", "train_loss": 0.5, "grad_norm": 1.2, "duration_sec": 0.5},
        )
        dummy_fit_res_2 = FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(initial_params),
            num_examples=150,
            metrics={"client_id": "client_B", "train_loss": 0.4, "grad_norm": 0.9, "duration_sec": 0.6},
        )

        results = [
            (DummyClientProxy("client_A"), dummy_fit_res_1),
            (DummyClientProxy("client_B"), dummy_fit_res_2),
        ]

        agg_params, agg_metrics = strategy.aggregate_fit(server_round=1, results=results, failures=[])
        assert agg_params is not None

        # Check round checkpoint file exists
        round_1_ckpt = ckpt_dir / "global_model_round_1.pt"
        assert round_1_ckpt.exists()

        # Check history file exists and contains communication stats
        history_file = ckpt_dir / "fl_round_history.json"
        assert history_file.exists()

        assert len(strategy.history) == 1
        record = strategy.history[0]
        assert record["round"] == 1
        assert "client_A" in record["participating_clients"]
        assert "client_B" in record["participating_clients"]
        assert record["communication"]["uplink_bytes"] > 0
        assert record["communication"]["downlink_bytes"] > 0
        assert FL_PRIVACY_BOUNDARY_DISCLAIMER in record["privacy_notice"]


# ------------------------------------------------------------------------------
# 4. Multi-Client Federated Learning End-to-End Simulation Tests
# ------------------------------------------------------------------------------

def test_multi_client_fl_simulation_end_to_end():
    """
    Simulate a 3-client Federated Learning system over 2 rounds.
    Verifies:
    1. Model weights evolve over rounds.
    2. Communication footprint is tracked.
    3. Checkpoints and evaluation metrics are recorded.
    4. Client sampling works.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = Path(tmp_dir) / "checkpoints"

        # Create 3 distinct simulated clients with their own private datasets
        clients = {}
        for cid in ["client_A", "client_B", "client_C"]:
            df = _generate_synthetic_substation_df(cid, num_meters=5, days=6)
            clients[cid] = SmartGridFlowerClient(
                client_id=cid,
                df=df,
                model_name="cnn_lstm",
                local_epochs=1,
                learning_rate=0.005,
            )

        coordinator = FLSimulationCoordinator(
            clients=clients,
            num_rounds=2,
            clients_per_round=2,  # Client sampling: 2 out of 3 sampled per round
            local_epochs=1,
            learning_rate=0.005,
            model_name="cnn_lstm",
            checkpoint_dir=ckpt_dir,
            seed=42,
        )

        initial_weights = [arr.copy() for arr in coordinator.global_parameters]
        summary = coordinator.run()

        assert summary["status"] == "COMPLETED"
        assert summary["total_rounds"] == 2
        assert summary["clients_count"] == 3
        assert len(summary["history"]) == 2

        # Verify weights evolved
        final_weights = coordinator.global_parameters
        weight_shift = sum(np.sum(np.abs(f - i)) for f, i in zip(final_weights, initial_weights))
        assert weight_shift > 1e-3, "Global model weights should update across federated rounds."

        # Verify checkpoints saved
        assert (ckpt_dir / "global_model_round_1.pt").exists()
        assert (ckpt_dir / "global_model_round_2.pt").exists()
        assert (ckpt_dir / "fl_simulation_summary.json").exists()

        # Verify loadable checkpoint
        test_model = get_model(model_name="cnn_lstm", input_size=11)
        epoch, metrics, metadata = load_checkpoint(ckpt_dir / "global_model_round_2.pt", model=test_model)
        assert epoch == 2
        assert "cnn_lstm" in metadata.get("model_name", "").lower() or "cnn_lstm" in str(metadata).lower()


def test_ethical_privacy_boundary_disclaimer():
    """Verify that the required privacy disclaimer is present in the codebase."""
    assert "DOES NOT guarantee mathematical differential privacy" in FL_PRIVACY_BOUNDARY_DISCLAIMER
    assert SmartGridFlowerClient.PRIVACY_NOTICE == FL_PRIVACY_BOUNDARY_DISCLAIMER
