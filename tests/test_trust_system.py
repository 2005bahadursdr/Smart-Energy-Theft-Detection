"""
Comprehensive Unit and Integration Tests for Trust Engine and Byzantine-Resilient Aggregators.

Validates:
1. Honest vs Malicious client discrimination (sign-flipping, model poisoning).
2. Missing history handling (new/unseen clients, unselected rounds).
3. Identical updates (d = 0, cos = 1.0, no zero division).
4. Zero weights / All-quarantined fallback handling.
5. Non-IID variance tolerance and reputation forgiveness recovery.
6. Numerical edge cases (zero gradient norm, NaN/Inf sanitization, non-negative weights).
7. Coordinate-wise median aggregation mathematics.
8. Detection diagnostics (Precision, Recall, F1, FPR, Malicious Influence).
9. End-to-end TrustAwareFedAvg simulation integration.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from aggregation.coordinate_median import CoordinateMedianStrategy, aggregate_coordinate_median
from aggregation.trust_aware_fedavg import TrustAwareFedAvg
from byzantine.attacks.model_poisoning import ModelPoisoningAttack
from byzantine.attacks.sign_flipping import SignFlippingAttack
from fl_system.clients.base_client import SmartGridFlowerClient
from fl_system.coordinator import FLSimulationCoordinator
from fl_system.models.model_utils import get_model, get_parameters_as_numpy
from trust.reputation.engine import HistoricalReputationEngine
from trust.similarity.cosine_distance import (
    compute_coordinate_median,
    compute_cosine_similarity,
    compute_euclidean_distance,
    compute_parameter_delta,
    evaluate_similarity_and_distance,
    flatten_parameters,
)
from trust.trust_engine import ClientTrustEvaluation, TrustEngine
from trust.validation.evaluator import ClientUpdateValidationEvaluator


@pytest.fixture
def synthetic_global_weights() -> list[np.ndarray]:
    """Sample global model weights."""
    rng = np.random.RandomState(42)
    return [
        rng.normal(0.0, 1.0, size=(10, 5)).astype(np.float32),
        rng.normal(0.0, 0.5, size=(10,)).astype(np.float32),
    ]


# ------------------------------------------------------------------------------
# 1. Similarity & Distance Edge Cases
# ------------------------------------------------------------------------------

def test_identical_updates(synthetic_global_weights):
    """Verify that identical updates yield zero distance and cosine similarity of 1.0."""
    w_global = synthetic_global_weights
    w_local = [w + 0.1 for w in w_global]

    deltas = {
        "client_A": compute_parameter_delta(w_local, w_global),
        "client_B": compute_parameter_delta(w_local, w_global),
    }

    metrics = evaluate_similarity_and_distance(deltas)
    assert metrics["client_A"]["raw_cosine"] == 1.0
    assert metrics["client_B"]["raw_cosine"] == 1.0
    assert metrics["client_A"]["euclidean_distance"] == 0.0
    assert metrics["client_B"]["euclidean_distance"] == 0.0
    assert metrics["client_A"]["similarity_score"] == 1.0
    assert metrics["client_B"]["similarity_score"] == 1.0


def test_zero_gradient_norm(synthetic_global_weights):
    """Verify zero gradient norm (no weight change) does not trigger division by zero or NaN."""
    w_global = synthetic_global_weights
    deltas = {
        "client_A": [np.zeros_like(w) for w in w_global],
        "client_B": [np.zeros_like(w) for w in w_global],
    }

    metrics = evaluate_similarity_and_distance(deltas)
    assert not np.isnan(metrics["client_A"]["raw_cosine"])
    assert not np.isnan(metrics["client_A"]["similarity_score"])
    assert metrics["client_A"]["norm"] == 0.0


def test_nan_inf_sanitization():
    """Verify that NaN or Inf vectors are safely converted without crashing."""
    vec_a = np.array([1.0, np.nan, np.inf, 2.0], dtype=np.float32)
    vec_b = np.array([1.0, 0.0, 0.0, 2.0], dtype=np.float32)

    cos = compute_cosine_similarity(vec_a, vec_b)
    dist = compute_euclidean_distance(vec_a, vec_b)

    assert not np.isnan(cos)
    assert not np.isinf(cos)
    assert not np.isnan(dist)
    assert -1.0 <= cos <= 1.0


# ------------------------------------------------------------------------------
# 2. Historical Reputation Engine
# ------------------------------------------------------------------------------

def test_reputation_missing_history_and_cold_start():
    """Verify new clients get prior reputation and unselected clients retain reputation."""
    rep_engine = HistoricalReputationEngine(initial_reputation=0.70, unseen_retention=0.90)

    # 1. Unseen client has initial reputation prior
    assert rep_engine.get_reputation("client_NEW") == 0.70

    # 2. Round 1: only client_A participates
    updated = rep_engine.update_reputations(
        round_trust_scores={"client_A": 0.95},
        all_known_client_ids=["client_A", "client_B"],
        server_round=1,
    )

    # client_A reputation increases via EMA
    assert updated["client_A"] > 0.70
    # client_B not participating: retains with slight passive decay (0.70 * 0.90 = 0.63)
    assert 0.60 <= updated["client_B"] <= 0.65


def test_reputation_non_iid_fairness_and_forgiveness():
    """Verify reputation floor and recovery boost for non-IID clients."""
    rep_engine = HistoricalReputationEngine(
        initial_reputation=0.70,
        decay_alpha=0.70,
        min_reputation_floor=0.15,
        forgiveness_boost=0.08,
    )

    # Simulate client_E having two tough non-IID rounds
    rep_engine.update_reputations({"client_E": 0.0}, server_round=1)
    rep_engine.update_reputations({"client_E": 0.0}, server_round=2)
    low_rep = rep_engine.get_reputation("client_E")
    # Must never drop below configured minimum floor
    assert low_rep >= 0.15

    # Simulate client_E subsequent recovery with good updates
    rep_engine.update_reputations({"client_E": 0.90}, server_round=3)
    rep_engine.update_reputations({"client_E": 0.95}, server_round=4)
    recovered_rep = rep_engine.get_reputation("client_E")
    # Forgiveness boost should accelerate recovery
    assert recovered_rep > low_rep + 0.20


# ------------------------------------------------------------------------------
# 3. Composite Trust Engine & Edge Cases
# ------------------------------------------------------------------------------

def test_trust_engine_honest_vs_malicious(synthetic_global_weights):
    """
    Verify TrustEngine flags sign-flipping and model-poisoning attackers
    while keeping honest nodes trusted.
    """
    w_global = synthetic_global_weights
    # 4 honest clients with slight gradient descent updates
    client_updates = {}
    for cid in ["client_A", "client_B", "client_C", "client_E"]:
        client_updates[cid] = [w - 0.05 + np.random.randn(*w.shape) * 0.005 for w in w_global]

    # 1 malicious client with sign flipping
    sf_attack = SignFlippingAttack(multiplier=-1.0, scale_factor=1.5, seed=42)
    client_updates["client_D"] = sf_attack.perturb_parameters(
        global_parameters=w_global,
        local_parameters=client_updates["client_A"],
        client_id="client_D",
        server_round=1,
    )

    sample_counts = {cid: 100 for cid in client_updates}

    engine = TrustEngine(
        config_dict={
            "trust_system": {
                "weights": {"similarity": 0.40, "update_distance": 0.30, "validation": 0.15, "reputation": 0.15},
                "aggregation": {"suspicious_threshold": 0.40, "policy": "quarantine"},
            }
        }
    )

    weights, evals, fallback = engine.evaluate_and_weight_updates(
        current_global_parameters=w_global,
        client_updates=client_updates,
        client_sample_counts=sample_counts,
        server_round=1,
    )

    assert not fallback
    # Honest clients must be trusted
    for cid in ["client_A", "client_B", "client_C", "client_E"]:
        assert evals[cid].is_suspicious is False
        assert evals[cid].composite_trust_score >= 0.40
        assert weights[cid] > 0.0

    # Attacker client_D must be flagged and quarantined
    assert evals["client_D"].is_suspicious is True
    assert evals["client_D"].quarantined is True
    assert weights["client_D"] == 0.0

    # Weight partition of unity check
    assert pytest.approx(sum(weights.values()), abs=1e-5) == 1.0
    for w in weights.values():
        assert w >= 0.0


def test_trust_engine_all_quarantined_fallback(synthetic_global_weights):
    """
    Verify safe fallback when all clients are adversarial or quarantined.
    Must not crash, divide by zero, or produce NaN weights.
    """
    w_global = synthetic_global_weights
    # All clients are malicious sign-flippers
    sf_attack = SignFlippingAttack(multiplier=-1.0, scale_factor=1.5, seed=42)
    client_updates = {}
    for cid in ["client_A", "client_B"]:
        honest = [w - 0.05 for w in w_global]
        client_updates[cid] = sf_attack.perturb_parameters(w_global, honest, cid, 1)

    # Artificially strict threshold to force quarantine on all
    engine = TrustEngine(
        config_dict={
            "trust_system": {
                "aggregation": {
                    "suspicious_threshold": 0.99, # Impossible threshold
                    "policy": "quarantine",
                    "zero_weight_fallback": "coordinate_median",
                }
            }
        }
    )

    weights, evals, fallback = engine.evaluate_and_weight_updates(
        current_global_parameters=w_global,
        client_updates=client_updates,
        client_sample_counts={"client_A": 50, "client_B": 50},
        server_round=1,
    )

    assert fallback is True
    assert pytest.approx(sum(weights.values()), abs=1e-5) == 1.0
    for w in weights.values():
        assert w > 0.0


# ------------------------------------------------------------------------------
# 4. Coordinate-Wise Median Aggregator
# ------------------------------------------------------------------------------

def test_coordinate_wise_median_aggregation():
    """Verify coordinate-wise median aggregator computes exact coordinate median."""
    class DummyProxy:
        def __init__(self, cid): self.cid = cid

    class DummyFitRes:
        def __init__(self, arrs):
            from flwr.common import ndarrays_to_parameters
            self.parameters = ndarrays_to_parameters(arrs)
            self.metrics = {"client_id": "test"}
            self.num_examples = 10

    # 3 clients with 1D array of length 3
    # c1: [1.0, 10.0, 100.0]
    # c2: [2.0, 20.0, 200.0]
    # c3: [999.0, 30.0, 300.0] (c3 is an extreme outlier on coord 0)
    results = [
        (DummyProxy("c1"), DummyFitRes([np.array([1.0, 10.0, 100.0], dtype=np.float32)])),
        (DummyProxy("c2"), DummyFitRes([np.array([2.0, 20.0, 200.0], dtype=np.float32)])),
        (DummyProxy("c3"), DummyFitRes([np.array([999.0, 30.0, 300.0], dtype=np.float32)])),
    ]

    agg = aggregate_coordinate_median(results)
    assert len(agg) == 1
    # Median along each coordinate:
    # coord 0: median([1.0, 2.0, 999.0]) = 2.0
    # coord 1: median([10.0, 20.0, 30.0]) = 20.0
    # coord 2: median([100.0, 200.0, 300.0]) = 200.0
    expected = np.array([2.0, 20.0, 200.0], dtype=np.float32)
    np.testing.assert_allclose(agg[0], expected, rtol=1e-5)


# ------------------------------------------------------------------------------
# 5. Detection Diagnostics Metrics
# ------------------------------------------------------------------------------

def test_detection_diagnostics():
    """Verify Precision, Recall, F1, FPR, and Malicious Influence calculations."""
    evals = {
        "client_A": ClientTrustEvaluation(
            client_id="client_A", num_samples=100, similarity_score=0.9, distance_score=0.9,
            validation_score=0.9, reputation_score=0.8, composite_trust_score=0.88,
            is_suspicious=False, raw_cosine=0.95, euclidean_distance=1.0, update_norm=5.0,
            aggregation_weight=0.5, quarantined=False,
        ),
        "client_B": ClientTrustEvaluation(
            client_id="client_B", num_samples=100, similarity_score=0.8, distance_score=0.8,
            validation_score=0.8, reputation_score=0.8, composite_trust_score=0.80,
            is_suspicious=False, raw_cosine=0.90, euclidean_distance=1.2, update_norm=5.1,
            aggregation_weight=0.5, quarantined=False,
        ),
        "client_D": ClientTrustEvaluation(
            client_id="client_D", num_samples=100, similarity_score=0.1, distance_score=0.1,
            validation_score=0.1, reputation_score=0.2, composite_trust_score=0.12,
            is_suspicious=True, raw_cosine=-0.95, euclidean_distance=15.0, update_norm=12.0,
            aggregation_weight=0.0, quarantined=True,
        ),
    }

    # Ground truth: client_D is Byzantine
    metrics = TrustEngine.calculate_detection_metrics(evals, ground_truth_attackers=["client_D"])

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["true_negatives"] == 2
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["malicious_influence"] == 0.0 # 0% weight to client_D


# ------------------------------------------------------------------------------
# 6. End-to-End Simulation Integration
# ------------------------------------------------------------------------------

def test_trust_aware_fedavg_simulation():
    """
    Execute 2-round simulation using TrustAwareFedAvg with active Byzantine attack.
    Verifies that the strategy executes, filters the attacker, updates reputation,
    and logs the round history without errors.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = Path(tmp_dir) / "checkpoints"

        # 3 simulated clients
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
                for m in range(3)
                for h in range(48)
            ])
            clients[cid] = SmartGridFlowerClient(
                client_id=cid,
                df=df,
                model_name="cnn_lstm",
                local_epochs=1,
            )

        strategy = TrustAwareFedAvg(
            model_name="cnn_lstm",
            checkpoint_dir=ckpt_dir,
            min_fit_clients=3,
            min_available_clients=3,
        )
        strategy.set_ground_truth_for_evaluation(["client_D"])

        from byzantine.attack_manager import ByzantineAttackManager
        attack_mgr = ByzantineAttackManager(
            attacker_clients=["client_D"],
            active_attack="sign_flipping",
            all_client_ids=list(clients.keys()),
            seed=42,
        )

        coordinator = FLSimulationCoordinator(
            clients=clients,
            strategy=strategy,
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
        assert len(strategy.history) == 2
        final_round = strategy.history[-1]
        assert "trust_evaluations" in final_round
        assert "client_D" in final_round["trust_evaluations"]
        assert "detection_diagnostics" in final_round
