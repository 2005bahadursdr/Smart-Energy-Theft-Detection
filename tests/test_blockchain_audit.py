"""
Unit and Integration Tests for Blockchain Audit Subsystem.

Validates:
1. Data safety and privacy boundary enforcement (no raw readings, weights, or PII).
2. Canonical hashing determinism and tamper-evident integrity verification.
3. ReputationRegistry contract deployment, authorization, and access control.
4. Single and batch transaction submission, receipt handling, and gas tracking.
5. On-chain query functions (getRecord, getRecordsByRound, getRecordsByClient, getLatestReputation).
6. Smart contract event emission (AuditEventRecorded) and log parsing.
7. MockBlockchainAdapter parity and failure simulation.
8. Full FL round audit commitment pipeline (commit_fl_round_audit).
9. Epistemic notice verification (blockchain records evidence, does not detect poisoning).
"""

import copy
from pathlib import Path
import numpy as np
import pytest
from web3 import Web3

from blockchain.canonical_hasher import (
    compute_canonical_record_hash,
    hash_client_identifier,
    hash_parameter_arrays,
    scale_reputation_for_contract,
    unscale_reputation_from_contract,
    verify_record_integrity,
)
from blockchain.mock_adapter import MockBlockchainAdapter
from blockchain.service import commit_fl_round_audit, get_blockchain_adapter
from blockchain.web3_adapter import Web3BlockchainAdapter


# ------------------------------------------------------------------------------
# 1. Privacy Boundary & Canonical Hashing Tests
# ------------------------------------------------------------------------------

def test_privacy_boundary_rejects_raw_telemetry():
    """Verify that attempting to hash raw readings or model weights raises ValueError."""
    forbidden_payload = {
        "round": 1,
        "client_id": "client_A",
        "consumption_kwh": [2.5, 3.1, 4.0], # FORBIDDEN
        "decision": "ACCEPTED",
    }
    with pytest.raises(ValueError, match="PRIVACY VIOLATION DETECTED"):
        compute_canonical_record_hash(forbidden_payload)


def test_canonical_hashing_determinism_and_tamper_detection():
    """
    Verify canonical hashing is deterministic regardless of key order,
    and detect any field modification.
    """
    payload_1 = {
        "round": 1,
        "client_id": "client_A",
        "decision": "ACCEPTED",
        "new_reputation": 0.85,
        "model_update_hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
    }
    # Same payload with different dictionary insertion order
    payload_2 = {
        "model_update_hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        "decision": "ACCEPTED",
        "client_id": "client_A",
        "round": 1,
        "new_reputation": 0.85,
    }

    hash_1 = compute_canonical_record_hash(payload_1)
    hash_2 = compute_canonical_record_hash(payload_2)
    assert hash_1 == hash_2
    assert hash_1.startswith("0x")
    assert len(hash_1) == 66 # 32 bytes hex + '0x'

    # Tampering test: verify changing even a single character fails verification
    tampered_payload = copy.deepcopy(payload_1)
    tampered_payload["new_reputation"] = 0.86 # Tampered
    assert not verify_record_integrity(tampered_payload, hash_1)


def test_hash_parameter_arrays():
    """Verify deterministic SHA-256 hashing of parameter arrays."""
    arr1 = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    arr2 = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    arr3 = [np.array([1.0, 2.0, 3.1], dtype=np.float32)]

    h1 = hash_parameter_arrays(arr1)
    h2 = hash_parameter_arrays(arr2)
    h3 = hash_parameter_arrays(arr3)

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 66


# ------------------------------------------------------------------------------
# 2. Web3BlockchainAdapter & ReputationRegistry Smart Contract Tests
# ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def web3_adapter():
    """Fixture providing initialized Web3 adapter on local EthereumTesterProvider."""
    adapter = Web3BlockchainAdapter(provider_type="eth_tester")
    adapter.deploy_contract()
    return adapter


def test_contract_deployment_and_owner(web3_adapter):
    """Verify ReputationRegistry contract is deployed and owner is configured."""
    assert web3_adapter.is_connected()
    assert web3_adapter.contract_address is not None
    assert Web3.is_address(web3_adapter.contract_address)

    # Owner should be account address
    owner = web3_adapter.contract.functions.owner().call()
    assert owner == web3_adapter.account_address


def test_coordinator_authorization(web3_adapter):
    """Verify contract owner can authorize a new coordinator address."""
    # Secondary test account from eth_tester
    secondary_account = web3_adapter.w3.eth.accounts[1]
    res = web3_adapter.authorize_coordinator(secondary_account)
    assert res["status"] == 1
    assert web3_adapter.contract.functions.authorizedCoordinators(secondary_account).call() is True


def test_record_single_audit_event(web3_adapter):
    """Verify recording a single audit event on the smart contract."""
    model_hash = Web3.keccak(text="grad_update_round_1").hex()
    record_hash = Web3.keccak(text="canonical_payload_1").hex()

    receipt = web3_adapter.record_audit_event(
        round_num=1,
        client_id="client_A",
        model_update_hash=model_hash,
        decision="ACCEPTED",
        reputation_change=0.05,
        new_reputation=0.75,
        record_hash=record_hash,
    )

    assert receipt["status"] == 1
    assert receipt["recordId"] is not None
    assert receipt["blockNumber"] > 0
    assert receipt["gasUsed"] > 0


def test_batch_record_audit_events(web3_adapter):
    """Verify batch committing multiple audit events in a single transaction."""
    entries = [
        {
            "round": 2,
            "client_id": "client_A",
            "model_update_hash": Web3.keccak(text="client_A_r2").hex(),
            "decision": "ACCEPTED",
            "reputation_change": 0.05,
            "new_reputation": 0.80,
            "record_hash": Web3.keccak(text="rec_A_r2").hex(),
        },
        {
            "round": 2,
            "client_id": "client_D",
            "model_update_hash": Web3.keccak(text="client_D_r2_poison").hex(),
            "decision": "QUARANTINED",
            "reputation_change": -0.30,
            "new_reputation": 0.40,
            "record_hash": Web3.keccak(text="rec_D_r2").hex(),
        },
    ]

    receipt = web3_adapter.batch_record_audit_events(entries)
    assert receipt["status"] == 1
    assert receipt["recordCount"] == 2
    assert len(receipt["recordIds"]) == 2


def test_contract_queries_and_integrity(web3_adapter):
    """Verify getRecord, getRecordsByRound, getRecordsByClient, and integrity verification."""
    count = web3_adapter.get_record_count()
    assert count >= 3

    # Query last record
    last_id = count - 1
    record = web3_adapter.get_record(last_id)
    assert record["recordId"] == last_id
    assert record["round"] == 2
    assert record["decisionName"] == "QUARANTINED"
    assert record["newReputation"] == 0.40

    # Query by round
    r2_ids = web3_adapter.get_records_by_round(2)
    assert last_id in r2_ids

    # Query by client
    d_ids = web3_adapter.get_records_by_client("client_D")
    assert last_id in d_ids

    # Latest reputation
    rep_d = web3_adapter.get_latest_reputation("client_D")
    assert rep_d == 0.40

    # Verify integrity
    is_valid = web3_adapter.verify_record_integrity(last_id, record["recordHash"])
    assert is_valid is True

    # Tampered expected hash fails verification
    fake_hash = Web3.keccak(text="tampered").hex()
    is_valid_fake = web3_adapter.verify_record_integrity(last_id, fake_hash)
    assert is_valid_fake is False


def test_get_audit_events_log_retrieval(web3_adapter):
    """Verify parsing emitted AuditEventRecorded event logs from the blockchain."""
    events = web3_adapter.get_audit_events(from_block=0)
    assert len(events) >= 3
    first_ev = events[0]
    assert "recordId" in first_ev
    assert "round" in first_ev
    assert "modelUpdateHash" in first_ev
    assert "decisionName" in first_ev
    assert "newReputation" in first_ev
    assert "transactionHash" in first_ev


# ------------------------------------------------------------------------------
# 3. Mock Adapter Parity & Failure Tests
# ------------------------------------------------------------------------------

def test_mock_adapter_parity_and_failure():
    """Verify MockBlockchainAdapter behaves identically in offline mode and simulates failures."""
    mock = MockBlockchainAdapter()
    assert mock.is_connected()

    # Deploy
    addr = mock.deploy_contract()
    assert Web3.is_address(addr)

    # Record event
    r_hash = Web3.keccak(text="mock_rec").hex()
    m_hash = Web3.keccak(text="mock_update").hex()
    receipt = mock.record_audit_event(
        round_num=1,
        client_id="client_B",
        model_update_hash=m_hash,
        decision="ACCEPTED",
        reputation_change=0.05,
        new_reputation=0.75,
        record_hash=r_hash,
    )
    assert receipt["status"] == 1
    assert mock.get_record_count() == 1

    # Verify integrity
    assert mock.verify_record_integrity(0, r_hash) is True
    assert mock.verify_record_integrity(0, "0x0000000000000000000000000000000000000000000000000000000000000000") is False

    # Event retrieval
    events = mock.get_audit_events()
    assert len(events) == 1
    assert events[0]["round"] == 1

    # Failure simulation
    failing_mock = MockBlockchainAdapter(simulate_failure=True)
    assert not failing_mock.is_connected()
    with pytest.raises(ConnectionError, match="Transaction failed: simulated node failure"):
        failing_mock.record_audit_event(1, "c1", m_hash, "ACCEPTED", 0.0, 0.7, r_hash)


# ------------------------------------------------------------------------------
# 4. End-to-End FL Round Commit Pipeline
# ------------------------------------------------------------------------------

def test_commit_fl_round_audit_pipeline():
    """Verify commit_fl_round_audit commits full round records and outputs audit report."""
    mock = MockBlockchainAdapter()

    evaluations = {
        "client_A": {"is_suspicious": False, "quarantined": False, "reputation_score": 0.70},
        "client_D": {"is_suspicious": True, "quarantined": True, "reputation_score": 0.70},
    }
    deltas = {
        "client_A": [np.array([0.1, 0.2], dtype=np.float32)],
        "client_D": [np.array([-0.5, -1.0], dtype=np.float32)],
    }
    rep_updates = {
        "client_A": 0.75,
        "client_D": 0.40,
    }

    result = commit_fl_round_audit(
        round_num=1,
        client_evaluations=evaluations,
        client_deltas=deltas,
        reputation_updates=rep_updates,
        adapter=mock,
    )

    assert result["status"] == 1
    assert result["round"] == 1
    assert len(result["records"]) == 2
    assert result["records"][0]["decision"] == "ACCEPTED"
    assert result["records"][1]["decision"] == "QUARANTINED"
    assert mock.get_record_count() == 2
