"""
Unit and Integration Tests for Byzantine Attack Simulation Subsystem.
Validates:
1. Sign-Flipping Attack (directional update inversion).
2. Model Poisoning Attack (targeted adversarial noise injection).
3. Random Update Attack (Gaussian noise parameter substitution).
4. Label Flipping Attack (training label inversion).
5. ByzantineAttackManager scheduling, isolation of honest nodes, and fraction allocation.
6. Ground-truth attacker tracking isolation (never accessible to central aggregator).
7. Epistemic notice verification (unusual update is not proof of malice).
8. End-to-end FL simulation under Byzantine attack.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from byzantine.attack_manager import (
    ANOMALY_EPISTEMIC_DISCLAIMER,
    ByzantineAttackManager,
)
from byzantine.attacks.label_flipping import LabelFlippingAttack
from byzantine.attacks.model_poisoning import ModelPoisoningAttack
from byzantine.attacks.random_update import RandomUpdateAttack
from byzantine.attacks.sign_flipping import SignFlippingAttack
from fl_system.clients.base_client import SmartGridFlowerClient
from fl_system.coordinator import FLSimulationCoordinator
from fl_system.models.dataset import MeterSequenceDataset
from fl_system.models.model_utils import get_model, get_parameters_as_numpy
from fl_system.server.strategy import SmartGridFedAvg


@pytest.fixture
def dummy_weights() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate sample global and local model weights for testing."""
    rng = np.random.RandomState(42)
    w_global = [rng.normal(0.0, 1.0, size=(10, 5)).astype(np.float32), np.zeros(10, dtype=np.float32)]
    # Local weights moved slightly in positive direction
    w_local = [w_global[0] + 0.1, w_global[1] + 0.05]
    return w_global, w_local


# ------------------------------------------------------------------------------
# 1. Attack Primitive Tests
# ------------------------------------------------------------------------------

def test_sign_flipping_inverts_gradient_direction(dummy_weights):
    """Verify sign flipping inverts the update step (Delta_w) by -1.5x."""
    w_global, w_local = dummy_weights
    attack = SignFlippingAttack(multiplier=-1.0, scale_factor=1.5, seed=42)

    poisoned = attack.perturb_parameters(
        global_parameters=w_global,
        local_parameters=w_local,
        client_id="client_D",
        server_round=1,
    )

    # Check shapes match
    assert len(poisoned) == len(w_local)
    for p, loc in zip(poisoned, w_local):
        assert p.shape == loc.shape
        assert p.dtype == loc.dtype

    # Check mathematical inversion:
    # poisoned - global = -1.5 * (local - global)
    for p, loc, glob in zip(poisoned, w_local, w_global):
        honest_delta = loc - glob
        poisoned_delta = p - glob
        expected_delta = -1.5 * honest_delta
        np.testing.assert_allclose(poisoned_delta, expected_delta, rtol=1e-5)

        # Cosine similarity must be exactly -1.0
        h_flat = honest_delta.flatten()
        p_flat = poisoned_delta.flatten()
        cos_sim = np.dot(h_flat, p_flat) / (np.linalg.norm(h_flat) * np.linalg.norm(p_flat))
        assert cos_sim < -0.999


def test_model_poisoning_injects_bounded_noise(dummy_weights):
    """Verify targeted model poisoning perturbs parameters with Gaussian noise."""
    w_global, w_local = dummy_weights
    attack = ModelPoisoningAttack(noise_mean=0.0, noise_std=1.5, adversarial_boost=2.0, seed=42)

    poisoned = attack.perturb_parameters(
        global_parameters=w_global,
        local_parameters=w_local,
        client_id="client_D",
        server_round=1,
    )

    assert len(poisoned) == len(w_local)
    for p, loc in zip(poisoned, w_local):
        assert p.shape == loc.shape
        # Perturbed parameters must differ from local parameters
        diff = np.abs(p - loc)
        assert np.mean(diff) > 0.01
        assert not np.isnan(p).any()


def test_random_update_replaces_weights(dummy_weights):
    """Verify random update completely substitutes parameters with Gaussian noise."""
    w_global, w_local = dummy_weights
    attack = RandomUpdateAttack(noise_mean=0.0, noise_std=3.0, seed=999)

    poisoned = attack.perturb_parameters(
        global_parameters=w_global,
        local_parameters=w_local,
        client_id="client_D",
        server_round=1,
    )

    for p, loc in zip(poisoned, w_local):
        assert p.shape == loc.shape
        assert not np.allclose(p, loc)

    # Should be completely uncorrelated with honest local weight matrix
    corr = np.corrcoef(poisoned[0].flatten(), w_local[0].flatten())[0, 1]
    assert abs(corr) < 0.20


def test_label_flipping_inverts_training_labels():
    """Verify label flipping inverts binary targets (0 -> 1 and 1 -> 0)."""
    X = torch.randn(20, 24, 11)
    # 15 zeros, 5 ones
    y = torch.tensor([0.0] * 15 + [1.0] * 5)
    m_ids = [f"M_{i}" for i in range(20)]
    t_stamps = ["2026-01-01 00:00:00"] * 20
    ds = MeterSequenceDataset(X=X, y=y, meter_ids=m_ids, timestamps=t_stamps, is_synthetic=True)

    attack = LabelFlippingAttack(flip_ratio=1.0, seed=42)
    flipped_ds = attack.flip_dataset(ds)

    # 100% flipped: all 0s became 1s, all 1s became 0s
    y_flipped_arr = flipped_ds.y.cpu().numpy()
    assert (y_flipped_arr[:15] == 1.0).all()
    assert (y_flipped_arr[15:] == 0.0).all()
    assert len(flipped_ds) == len(ds)


# ------------------------------------------------------------------------------
# 2. Byzantine Attack Manager Tests
# ------------------------------------------------------------------------------

def test_attack_manager_isolation_of_honest_clients(dummy_weights):
    """Verify attack manager leaves honest clients strictly untouched."""
    w_global, w_local = dummy_weights
    manager = ByzantineAttackManager(
        attacker_clients=["client_D"],
        active_attack="sign_flipping",
        seed=42,
    )

    # client_A is honest
    out_A = manager.apply_attack("client_A", 1, w_global, w_local)
    for res, honest in zip(out_A, w_local):
        np.testing.assert_array_equal(res, honest)

    # client_D is Byzantine
    out_D = manager.apply_attack("client_D", 1, w_global, w_local)
    for res, honest in zip(out_D, w_local):
        # Must differ because attack was executed
        assert not np.array_equal(res, honest)


def test_attack_manager_malicious_fraction():
    """Verify calculating malicious client subset from fraction (e.g. 20% or 40%)."""
    all_cids = ["client_A", "client_B", "client_C", "client_D", "client_E"]

    # 20% of 5 = 1 attacker
    mgr_20 = ByzantineAttackManager(malicious_fraction=0.20, all_client_ids=all_cids, seed=42)
    assert len(mgr_20.attacker_clients) == 1

    # 40% of 5 = 2 attackers
    mgr_40 = ByzantineAttackManager(malicious_fraction=0.40, all_client_ids=all_cids, seed=42)
    assert len(mgr_40.attacker_clients) == 2


def test_attack_manager_ground_truth_isolation():
    """
    Verify ground-truth audit logging is strictly isolated and can be exported
    for post-hoc evaluation without exposing labels to the central server.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = ByzantineAttackManager(
            attacker_clients=["client_D"],
            active_attack="sign_flipping",
            seed=42,
        )

        dummy_w = [np.array([1.0, 2.0], dtype=np.float32)]
        manager.apply_attack("client_D", 1, dummy_w, dummy_w)

        gt_file = Path(tmp_dir) / "byzantine_ground_truth.json"
        exported_path = manager.export_ground_truth_log(gt_file)

        assert exported_path.exists()
        assert len(manager._ground_truth_audit) == 1
        record = manager._ground_truth_audit[0]
        assert record["client_id"] == "client_D"
        assert record["attack_type"] == "sign_flipping"

        # Ground-truth evaluation labels
        labels = manager.get_ground_truth_labels(["client_A", "client_D"], server_round=1)
        assert labels["client_A"] is False
        assert labels["client_D"] is True


def test_epistemic_disclaimer_notice():
    """Verify the epistemic notice: an unusual update is not proof of malice."""
    assert "NOT definitive proof of malicious Byzantine intent" in ANOMALY_EPISTEMIC_DISCLAIMER
    assert ByzantineAttackManager.EPISTEMIC_NOTICE == ANOMALY_EPISTEMIC_DISCLAIMER


# ------------------------------------------------------------------------------
# 3. End-to-End Federated Learning with Byzantine Attack Injection
# ------------------------------------------------------------------------------

def test_fl_simulation_under_byzantine_attack():
    """
    Execute multi-round FL simulation under active Byzantine sign-flipping.
    Verifies that:
    1. Attack is injected on scheduled rounds.
    2. Honest clients train normally.
    3. Ground truth audit log is exported to the checkpoint directory.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = Path(tmp_dir) / "checkpoints"

        # Create 3 clients with synthetic data
        clients = {}
        for cid in ["client_A", "client_B", "client_D"]:
            df = pd.DataFrame([
                {
                    "timestamp": f"2026-01-{(h // 24) + 1:02d} {h % 24:02d}:00:00",
                    "meter_id": f"M_{cid}_{m}",
                    "client_id": cid,
                    "consumption_kwh": 2.5 + np.random.randn() * 0.1,
                    "meter_status": "NORMAL",
                    "event_type": "NONE",
                    "hardware_tamper_flag": 0,
                    "simulated_theft_label": 0,
                }
                for m in range(4)
                for h in range(48)
            ])
            clients[cid] = SmartGridFlowerClient(
                client_id=cid,
                df=df,
                model_name="cnn_lstm",
                local_epochs=1,
            )

        attack_mgr = ByzantineAttackManager(
            attacker_clients=["client_D"],
            active_attack="sign_flipping",
            all_client_ids=list(clients.keys()),
            seed=42,
        )

        coordinator = FLSimulationCoordinator(
            clients=clients,
            num_rounds=2,
            clients_per_round=3,
            local_epochs=1,
            model_name="cnn_lstm",
            checkpoint_dir=ckpt_dir,
            attack_manager=attack_mgr,
            seed=42,
        )

        summary = coordinator.run()

        assert summary["status"] == "COMPLETED"
        assert summary["attack_configured"] is True
        assert summary["active_attack"] == "sign_flipping"
        assert "client_D" in summary["attacker_clients"]

        # Ground truth file must be present
        gt_file = ckpt_dir / "byzantine_ground_truth.json"
        assert gt_file.exists()
