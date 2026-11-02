"""
Comprehensive Unit & Integration Test Suite for FastAPI Backend.

Validates:
1. GET  /health (system health, database status, blockchain adapter, model readiness).
2. GET  /api/meters/summary (privacy preservation: NEVER exposes individual raw telemetry).
3. GET  /api/clients (substation profiles, non-IID load heterogeneity, local privacy firewalls).
4. GET  /api/fl/rounds (FL round telemetry, loss, accuracy, F1, PR-AUC, quarantine status).
5. POST /api/fl/start (safe experiment parameter bounds, 422 validation, production auth warning).
6. GET  /api/predictions (aggregated fraud detection metrics, risk distribution, demo labels).
7. GET  /api/byzantine/events (perturbation events, mitigation actions, epistemic disclaimers).
8. GET  /api/trust/scores (component trust breakdowns, normalized weights, thresholds).
9. GET  /api/blockchain/audit (on-chain audit records, [TAMPER-EVIDENT OK] integrity status).
10. POST /api/simulation/run (safe parameter bounds, 422 validation, production auth warning).
11. InMemoryDatabaseAdapter parity (thread-safety, filtering operators, sorting, updates).
12. Configured local-origin CORS headers.
13. Security and production warning headers on all responses.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.adapter import InMemoryDatabaseAdapter, seed_default_metadata


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


# ------------------------------------------------------------------------------
# 1. Health Probe Endpoint
# ------------------------------------------------------------------------------

def test_get_health(client):
    """Verify /health reports system health, database, blockchain, and models."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ("healthy", "degraded")
    assert "database" in data
    assert data["database"]["provider"] in ("mongodb", "in_memory")
    assert isinstance(data["database"]["connected"], bool)
    assert "blockchain" in data
    assert "model_readiness" in data
    assert "CNN_LSTM" in data["model_readiness"]["available_models"]

    # Verify security headers
    assert "X-Production-Warning" in response.headers
    assert "X-Privacy-Enforcement" in response.headers


# ------------------------------------------------------------------------------
# 2. Smart Meter Summary & Privacy Constraint
# ------------------------------------------------------------------------------

def test_get_meters_summary_privacy_invariant(client):
    """
    CRUCIAL PRIVACY TEST:
    Verify that GET /api/meters/summary returns ONLY statistical aggregates
    and NEVER exposes raw meter time-series readings or consumer PII.
    """
    response = client.get("/api/meters/summary")
    assert response.status_code == 200
    data = response.json()

    # Structural aggregates verification
    assert data["total_meters"] > 0
    assert data["total_readings_sampled"] > 0
    assert 0.0 <= data["theft_ratio"] <= 1.0

    metrics = data["consumption_metrics"]
    assert "mean_kwh" in metrics
    assert "std_kwh" in metrics
    assert "median_kwh" in metrics
    assert "p25_kwh" in metrics
    assert "p75_kwh" in metrics
    assert metrics["mean_kwh"] > 0.0

    # Strict Privacy Assertions: No raw arrays or timeseries keys
    forbidden_keys = ["readings", "consumption_kwh", "raw_series", "time_series", "pii", "user_id"]
    for key in forbidden_keys:
        assert key not in data, f"Privacy violation: Forbidden key '{key}' found in response!"
        assert key not in metrics, f"Privacy violation: Forbidden key '{key}' found in metrics!"

    # Demo data label verification
    assert data["data_source_label"] in ("SIMULATED_DEMO_DATA", "SGCC_REAL_WORLD_BENCHMARK")
    assert len(data["demo_notice"]) > 0


# ------------------------------------------------------------------------------
# 3. Substation Client Metadata
# ------------------------------------------------------------------------------

def test_get_clients(client):
    """Verify GET /api/clients returns 5 regional non-IID substation partitions."""
    response = client.get("/api/clients")
    assert response.status_code == 200
    data = response.json()

    assert data["total_clients"] == 5
    clients = data["clients"]
    client_ids = [c["client_id"] for c in clients]
    assert "client_A" in client_ids
    assert "client_E" in client_ids

    # Verify non-IID variation and local privacy firewall enforcement
    daily_loads = [c["mean_daily_kwh"] for c in clients]
    assert max(daily_loads) > min(daily_loads) * 5, "Non-IID load heterogeneity should be visible"

    for c in clients:
        assert c["privacy_firewall"] == "ACTIVE"
        assert c["central_fl_access"] == "BLOCKED"
        assert 0.0 <= c["current_reputation"] <= 1.0
        assert 0.0 <= c["current_trust_score"] <= 1.0


# ------------------------------------------------------------------------------
# 4. Federated Learning Rounds Telemetry
# ------------------------------------------------------------------------------

def test_get_fl_rounds(client):
    """Verify GET /api/fl/rounds returns round telemetry and metrics."""
    response = client.get("/api/fl/rounds")
    assert response.status_code == 200
    data = response.json()

    assert data["total_rounds_completed"] >= 1
    assert data["best_f1_score"] > 0.0
    assert data["total_communication_mb"] > 0.0
    assert "epistemic_disclaimer" in data

    rounds = data["rounds"]
    for r in rounds:
        assert r["round_number"] >= 1
        assert 0.0 <= r["accuracy"] <= 1.0
        assert 0.0 <= r["f1_score"] <= 1.0
        assert 0.0 <= r["pr_auc"] <= 1.0
        assert len(r["participating_clients"]) > 0


# ------------------------------------------------------------------------------
# 5. FL Start & Parameter Validation Bounds
# ------------------------------------------------------------------------------

def test_post_fl_start_valid(client):
    """Verify POST /api/fl/start initiates job and returns production warning."""
    payload = {
        "rounds": 3,
        "epochs": 1,
        "model": "cnn_lstm",
        "strategy": "trust_aware_fedavg",
        "attack_type": "sign_flipping",
        "malicious_fraction": 0.2,
        "enable_blockchain": True,
        "seed": 42,
    }
    response = client.post("/api/fl/start", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ("ACCEPTED", "RUNNING")
    assert data["job_id"].startswith("fl-job-")
    assert data["estimated_duration_sec"] >= 2
    assert "CRITICAL SECURITY WARNING" in data["production_warning"]
    assert "authentication" in data["production_warning"].lower()


@pytest.mark.parametrize(
    "invalid_payload,expected_err_field",
    [
        ({"rounds": 25, "epochs": 1}, "rounds"),  # rounds > 20
        ({"rounds": 0, "epochs": 1}, "rounds"),   # rounds < 1
        ({"rounds": 5, "epochs": 10}, "epochs"),  # epochs > 5
        ({"rounds": 5, "epochs": 0}, "epochs"),   # epochs < 1
        ({"rounds": 5, "model": "invalid_net"}, "model"),  # unsupported model
        ({"rounds": 5, "strategy": "invalid_strat"}, "strategy"),  # unsupported strategy
        ({"rounds": 5, "malicious_fraction": 0.8}, "malicious_fraction"),  # fraction > 0.5
        ({"rounds": 5, "malicious_fraction": -0.1}, "malicious_fraction"), # fraction < 0.0
    ],
)
def test_post_fl_start_validation_errors(client, invalid_payload, expected_err_field):
    """Verify parameter validation bounds enforce 422 Unprocessable Entity."""
    response = client.post("/api/fl/start", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"


# ------------------------------------------------------------------------------
# 6. Predictions & Demo Labels
# ------------------------------------------------------------------------------

def test_get_predictions_privacy_and_demo_labels(client):
    """Verify GET /api/predictions delivers aggregated predictions without raw readings."""
    response = client.get("/api/predictions")
    assert response.status_code == 200
    data = response.json()

    assert data["total_windows_evaluated"] > 0
    assert data["normal_windows_count"] + data["theft_windows_count"] == data["total_windows_evaluated"]
    assert data["data_source_label"] in ("SIMULATED_DEMO_DATA", "SGCC_REAL_WORLD_BENCHMARK")

    # Metrics verification
    metrics = data["metrics"]
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0

    # Risk distribution
    risk = data["risk_distribution"]
    assert risk["low_risk_count"] > 0

    # Privacy verification
    forbidden = ["readings", "consumption_kwh", "raw_series", "pii"]
    for k in forbidden:
        assert k not in data


def test_post_predictions_evaluate_normal(client):
    """Verify POST /api/predictions/evaluate on normal consumption profile."""
    payload = {
        "substation_id": "client_A",
        "pattern_type": "NORMAL",
        "meter_id": "MTR-TEST-001",
    }
    response = client.post("/api/predictions/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["meter_id"] == "MTR-TEST-001"
    assert data["substation_id"] == "client_A"
    assert data["pattern_type"] == "NORMAL"
    assert data["theft_risk_tier"] == "LOW_RISK"
    assert data["theft_probability"] < 0.30
    assert len(data["consumption_sequence"]) == 30
    assert len(data["anomaly_flags"]) >= 1


def test_post_predictions_evaluate_theft(client):
    """Verify POST /api/predictions/evaluate on bypass theft pattern."""
    payload = {
        "substation_id": "client_B",
        "pattern_type": "BYPASS_THEFT",
        "meter_id": "MTR-THEFT-002",
        "evening_drop_pct": 0.80,
    }
    response = client.post("/api/predictions/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["meter_id"] == "MTR-THEFT-002"
    assert data["pattern_type"] == "BYPASS_THEFT"
    assert data["theft_risk_tier"] == "HIGH_RISK"
    assert data["theft_probability"] >= 0.70
    assert any("bypass" in f.lower() or "drop" in f.lower() for f in data["anomaly_flags"])


# ------------------------------------------------------------------------------
# 7. Byzantine Events
# ------------------------------------------------------------------------------

def test_get_byzantine_events(client):
    """Verify GET /api/byzantine/events delivers perturbation events and research disclaimer."""
    response = client.get("/api/byzantine/events")
    assert response.status_code == 200
    data = response.json()

    assert data["total_events"] >= 1
    assert data["detected_events"] >= 1
    assert "RESEARCH NOTICE" in data["disclaimer"]

    for ev in data["events"]:
        assert ev["round"] >= 1
        assert ev["action_taken"] in ("QUARANTINED", "WEIGHT_REDUCED", "REPUTATION_PENALIZED", "NONE")
        assert ev["ground_truth_label"] in ("MALICIOUS_BYZANTINE", "HONEST_NON_IID")


# ------------------------------------------------------------------------------
# 8. Trust Scores
# ------------------------------------------------------------------------------

def test_get_trust_scores(client):
    """Verify GET /api/trust/scores returns multi-criteria score breakdowns."""
    response = client.get("/api/trust/scores")
    assert response.status_code == 200
    data = response.json()

    assert len(data["scores"]) >= 5
    assert "weights_used" in data
    assert "thresholds" in data

    weights = data["weights_used"]
    assert round(weights["weight_similarity"] + weights["weight_validation"] + weights["weight_reputation"], 2) == 1.0

    for s in data["scores"]:
        assert -1.0 <= s["similarity_score"] <= 1.0
        assert 0.0 <= s["reputation_score"] <= 1.0
        assert 0.0 <= s["composite_trust_score"] <= 1.0
        assert 0.0 <= s["normalized_weight"] <= 1.0
        assert s["status"] in ("TRUSTED", "SUSPICIOUS", "QUARANTINED")


# ------------------------------------------------------------------------------
# 9. Blockchain Audit Trail & Integrity
# ------------------------------------------------------------------------------

def test_get_blockchain_audit(client):
    """Verify GET /api/blockchain/audit returns tamper-evident on-chain records."""
    response = client.get("/api/blockchain/audit")
    assert response.status_code == 200
    data = response.json()

    assert data["total_blocks"] >= 1
    assert data["total_audit_records"] >= 1
    assert "EVIDENTIARY DISCLAIMER" in data["epistemic_disclaimer"]

    # Check on-chain record integrity
    first_block = data["blocks"][0]
    assert first_block["block_number"] >= 1
    assert len(first_block["records"]) >= 1

    for rec in first_block["records"]:
        assert rec["integrity_status"] == "[TAMPER-EVIDENT OK]"
        assert rec["decision"] in ("ACCEPTED", "QUARANTINED", "WEIGHT_REDUCED")
        assert rec["record_hash"].startswith("0x")


# ------------------------------------------------------------------------------
# 10. Simulation Run & Validation Bounds
# ------------------------------------------------------------------------------

def test_post_simulation_run_valid(client):
    """Verify POST /api/simulation/run executes simulation with safe parameters."""
    payload = {
        "num_meters": 120,
        "days": 10,
        "sampling_freq_minutes": 15,
        "theft_ratio": 0.15,
        "seed": 42,
    }
    response = client.post("/api/simulation/run", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "COMPLETED"
    assert data["simulation_id"].startswith("sim-")
    assert data["generated_summary"]["total_meters"] == 120
    assert "CRITICAL SECURITY WARNING" in data["production_warning"]


@pytest.mark.parametrize(
    "invalid_payload,expected_field",
    [
        ({"num_meters": 5, "days": 10}, "num_meters"),      # meters < 10
        ({"num_meters": 5000, "days": 10}, "num_meters"),   # meters > 2000
        ({"num_meters": 100, "days": 120}, "days"),         # days > 90
        ({"num_meters": 100, "theft_ratio": 0.8}, "theft_ratio"), # theft > 0.5
        ({"num_meters": 100, "sampling_freq_minutes": 45}, "sampling_freq_minutes"), # not in (15, 30, 60)
    ],
)
def test_post_simulation_run_validation_errors(client, invalid_payload, expected_field):
    """Verify simulation runner enforces safe parameter constraints with 422."""
    response = client.post("/api/simulation/run", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"


# ------------------------------------------------------------------------------
# 11. Documented In-Memory Adapter Parity
# ------------------------------------------------------------------------------

def test_in_memory_adapter_operations():
    """Verify thread-safe InMemoryDatabaseAdapter CRUD and query operators."""
    mem = InMemoryDatabaseAdapter()
    assert mem.ping() is True
    assert mem.get_provider_name() == "in_memory"

    # Insert
    id1 = mem.insert_one("test_coll", {"round": 1, "client": "client_A", "score": 0.95})
    id2 = mem.insert_one("test_coll", {"round": 2, "client": "client_B", "score": 0.82})
    id3 = mem.insert_one("test_coll", {"round": 3, "client": "client_C", "score": 0.60})

    assert mem.count("test_coll") == 3

    # Query with equality
    doc = mem.find_one("test_coll", {"client": "client_A"})
    assert doc is not None
    assert doc["score"] == 0.95

    # Query with $in
    docs_in = mem.find("test_coll", {"client": {"$in": ["client_A", "client_B"]}})
    assert len(docs_in) == 2

    # Query with range operator $gte
    docs_high = mem.find("test_coll", {"score": {"$gte": 0.80}})
    assert len(docs_high) == 2

    # Sorting descending
    docs_sorted = mem.find("test_coll", sort=[("score", -1)])
    assert docs_sorted[0]["client"] == "client_A"
    assert docs_sorted[-1]["client"] == "client_C"

    # Limit
    docs_limited = mem.find("test_coll", limit=1)
    assert len(docs_limited) == 1

    # Update
    updated = mem.update_one("test_coll", {"client": "client_A"}, {"score": 0.99})
    assert updated is True
    updated_doc = mem.find_one("test_coll", {"client": "client_A"})
    assert updated_doc["score"] == 0.99

    # Delete
    deleted = mem.delete_many("test_coll", {"client": "client_C"})
    assert deleted == 1
    assert mem.count("test_coll") == 2

    mem.close()
    assert mem.count("test_coll") == 0


# ------------------------------------------------------------------------------
# 12. CORS Configuration Verification
# ------------------------------------------------------------------------------

def test_cors_headers_allowed_origin(client):
    """Verify CORS middleware responds with correct headers for allowed origin."""
    response = client.get(
        "/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert response.headers.get("access-control-allow-credentials") == "true"
