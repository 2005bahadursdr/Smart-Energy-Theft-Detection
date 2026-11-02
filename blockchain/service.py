"""
Blockchain Audit Subsystem Service & Orchestrator.

Provides unified interface for FL Coordinator and API services to:
1. Initialize appropriate blockchain adapter (EthereumTesterProvider, HTTP, or Mock).
2. Commit round-by-round federated learning audit trails to the blockchain.
3. Query tamper-evident historical audit logs and reputations.

CRITICAL DISCLOSURES & EPISTEMIC PRINCIPLES:
- "Blockchain records evidence; it does not itself detect or prevent poisoning":
  The smart contract maintains an immutable audit trail of aggregator dispositions,
  cryptographic parameter hashes, and reputation changes. It does not replace the
  statistical or trust-based detection algorithms.
- "Hashes do not prove decisions were correct":
  The hash proves non-repudiation and temporal ordering of the coordinator's claim.
- ZERO RAW DATA & ZERO WEIGHTS ON-CHAIN:
  Never submit raw smart meter telemetry, consumer PII, or neural network weights.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from blockchain.canonical_hasher import (
    compute_canonical_record_hash,
    hash_parameter_arrays,
    verify_record_integrity,
)
from blockchain.mock_adapter import MockBlockchainAdapter
from blockchain.web3_adapter import Web3BlockchainAdapter

logger = logging.getLogger("blockchain_service")


def get_blockchain_adapter(
    provider_type: Optional[str] = None,
    rpc_url: Optional[str] = None,
    force_mock: bool = False,
) -> Union[Web3BlockchainAdapter, MockBlockchainAdapter]:
    """
    Factory creating appropriate blockchain adapter based on environment or configuration.
    Falls back gracefully to MockBlockchainAdapter if web3 or node is unavailable.
    """
    if force_mock:
        logger.info("Instantiating MockBlockchainAdapter (force_mock=True).")
        return MockBlockchainAdapter()

    prov = (provider_type or os.getenv("BLOCKCHAIN_PROVIDER_TYPE", "eth_tester")).lower()

    if prov == "mock":
        return MockBlockchainAdapter()

    try:
        adapter = Web3BlockchainAdapter(provider_type=prov, rpc_url=rpc_url)
        if adapter.is_connected():
            return adapter
        logger.warning(f"Web3 provider '{prov}' is not connected. Falling back to MockBlockchainAdapter.")
        return MockBlockchainAdapter()
    except Exception as e:
        logger.warning(f"Failed to initialize Web3BlockchainAdapter: {e}. Falling back to MockBlockchainAdapter.")
        return MockBlockchainAdapter()


def commit_fl_round_audit(
    round_num: int,
    client_evaluations: Dict[str, Any],
    client_deltas: Dict[str, List[np.ndarray]],
    reputation_updates: Dict[str, float],
    adapter: Optional[Union[Web3BlockchainAdapter, MockBlockchainAdapter]] = None,
    persist_to_disk: bool = True,
) -> Dict[str, Any]:
    """
    Commit full federated learning round audit records to the blockchain.
    Only authentic on-chain transactions with verified cryptographic transaction hashes
    are persisted to the on-disk audit trail to prevent dummy/mock record pollution.

    Args:
        round_num: Current FL round number.
        client_evaluations: Dict mapping client_id to ClientTrustEvaluation or evaluation dict.
        client_deltas: Dict mapping client_id to gradient delta parameter arrays.
        reputation_updates: Dict mapping client_id to updated float reputation in [0, 1].
        adapter: Blockchain adapter instance. If None, initialized from environment.
        persist_to_disk: Whether to persist authentic on-chain receipts to disk (default True).

    Returns:
        Dict containing commit status, transaction hash, block number, and record details.
    """
    blockchain = adapter or get_blockchain_adapter()
    blockchain._ensure_contract() if hasattr(blockchain, "_ensure_contract") else None

    timestamp_iso = datetime.now(timezone.utc).isoformat()
    audit_inputs = []
    local_audit_trail = []

    for cid, ev in client_evaluations.items():
        # 1. Compute deterministic hash of model parameter delta
        deltas = client_deltas.get(cid, [])
        model_hash = hash_parameter_arrays(deltas)

        # 2. Determine discrete decision
        is_suspicious = getattr(ev, "is_suspicious", False) if hasattr(ev, "is_suspicious") else ev.get("is_suspicious", False)
        is_quarantined = getattr(ev, "quarantined", False) if hasattr(ev, "quarantined") else ev.get("quarantined", False)

        if is_quarantined:
            decision_str = "QUARANTINED"
        elif is_suspicious:
            decision_str = "ATTENUATED"
        else:
            decision_str = "ACCEPTED"

        new_rep = float(reputation_updates.get(cid, 0.70))
        prev_rep = getattr(ev, "reputation_score", 0.70) if hasattr(ev, "reputation_score") else ev.get("reputation_score", 0.70)
        rep_change = round(new_rep - float(prev_rep), 4)

        # 3. Canonical audit payload for hash commitment
        canonical_payload = {
            "round": int(round_num),
            "client_id": str(cid),
            "model_update_hash": model_hash,
            "decision": decision_str,
            "reputation_change": rep_change,
            "new_reputation": round(new_rep, 4),
            "timestamp": timestamp_iso,
        }

        # 4. Generate canonical Keccak-256 record hash
        record_hash = compute_canonical_record_hash(canonical_payload)

        audit_inputs.append({
            "round": int(round_num),
            "client_id": str(cid),
            "model_update_hash": model_hash,
            "decision": decision_str,
            "reputation_change": rep_change,
            "new_reputation": new_rep,
            "record_hash": record_hash,
        })

        local_audit_trail.append({
            **canonical_payload,
            "record_hash": record_hash,
        })

    # 5. Submit batch transaction to blockchain
    receipt = blockchain.batch_record_audit_events(audit_inputs)

    result_payload = {
        "status": receipt.get("status", 1),
        "round": round_num,
        "blockNumber": receipt.get("blockNumber"),
        "transactionHash": receipt.get("transactionHash"),
        "recordIds": receipt.get("recordIds", []),
        "contractAddress": getattr(blockchain, "contract_address", None),
        "records": local_audit_trail,
        "timestamp": timestamp_iso,
    }

    # Persist round audit block to disk ONLY for authentic on-chain transactions
    tx_hash = receipt.get("transactionHash")
    is_real_tx = (
        tx_hash is not None
        and str(tx_hash).strip() not in ("", "None", "null")
        and not str(tx_hash).startswith("0xmock")
    )
    is_mock = isinstance(blockchain, MockBlockchainAdapter) or blockchain.__class__.__name__ == "MockBlockchainAdapter"

    if persist_to_disk and is_real_tx and not is_mock:
        audit_file = Path("experiments/results/blockchain_audit_trail.json")
        try:
            audit_file.parent.mkdir(parents=True, exist_ok=True)
            existing_trail = []
            if audit_file.exists() and audit_file.stat().st_size > 0:
                try:
                    with open(audit_file, "r", encoding="utf-8") as f:
                        existing_trail = json.load(f) or []
                except Exception:
                    existing_trail = []
            # Prevent duplicate round entries with same transaction hash
            existing_trail = [
                b for b in existing_trail
                if not (b.get("round") == round_num and b.get("transactionHash") == tx_hash)
            ]
            existing_trail.append(result_payload)
            with open(audit_file, "w", encoding="utf-8") as f:
                json.dump(existing_trail, f, indent=2)
            logger.info(f"[BLOCKCHAIN AUDIT] Persisted authentic round {round_num} block (tx: {tx_hash}) to disk.")
        except Exception as e:
            logger.warning(f"Failed to persist blockchain audit trail to disk: {e}")
    else:
        logger.info(
            f"[BLOCKCHAIN AUDIT] Round {round_num} evaluated via in-memory/mock adapter. "
            f"Dummy record excluded from persistent blockchain audit trail."
        )

    logger.info(
        f"[BLOCKCHAIN AUDIT] Round {round_num} processed. "
        f"Block: {receipt.get('blockNumber')} | Records: {receipt.get('recordCount')} "
        f"| Tx: {receipt.get('transactionHash')}"
    )

    return result_payload
