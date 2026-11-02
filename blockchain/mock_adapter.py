"""
Mock Blockchain Adapter for Offline Testing and Degraded Network Simulation.

Provides an in-memory Ethereum-compatible ledger without requiring an active Web3 RPC node.
Maintains identical API signatures, event structures, and receipt semantics as Web3BlockchainAdapter.
"""

from datetime import datetime, timezone
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union
from web3 import Web3

from blockchain.canonical_hasher import (
    hash_client_identifier,
    scale_reputation_for_contract,
    unscale_reputation_from_contract,
)

logger = logging.getLogger("mock_blockchain_adapter")

# Integer representation of ReputationRegistry.Decision enum
DECISION_MAP = {
    "ACCEPTED": 0,
    "ATTENUATED": 1,
    "QUARANTINED": 2,
    "REJECTED": 3,
    0: 0,
    1: 1,
    2: 2,
    3: 3,
}
DECISION_NAMES = {0: "ACCEPTED", 1: "ATTENUATED", 2: "QUARANTINED", 3: "REJECTED"}


class MockBlockchainAdapter:
    """
    In-memory mock adapter mimicking the ReputationRegistry smart contract.
    """

    def __init__(
        self,
        contract_address: Optional[str] = None,
        owner_address: str = "0x90F8bf6A479f320ead074411a4B0e7944Ea8c9C1",
        simulate_failure: bool = False,
    ):
        self.owner_address = Web3.to_checksum_address(owner_address)
        self.caller_address = self.owner_address
        self.contract_address = contract_address or "0x5FbDB2315678afecb367f032d93F642f64180aa3"
        self.simulate_failure = simulate_failure

        # In-memory storage
        self.authorized_coordinators = {self.owner_address: True}
        self.audit_records: List[Dict[str, Any]] = []
        self.latest_reputations: Dict[str, int] = {}
        self.emitted_events: List[Dict[str, Any]] = []
        self.current_block_number = 1
        self.tx_count = 0

        logger.info(
            f"Initialized MockBlockchainAdapter | Contract: {self.contract_address} | "
            f"Owner: {self.owner_address}"
        )

    def is_connected(self) -> bool:
        """Returns True since mock adapter is always operational in memory."""
        return not self.simulate_failure

    def set_caller(self, address: str) -> None:
        """Set caller address to test authorized / unauthorized coordinator transactions."""
        self.caller_address = Web3.to_checksum_address(address)

    def authorize_coordinator(self, coordinator_address: str) -> Dict[str, Any]:
        """Authorize a coordinator address."""
        if self.caller_address != self.owner_address:
            raise PermissionError("ReputationRegistry: caller is not owner")
        c_addr = Web3.to_checksum_address(coordinator_address)
        self.authorized_coordinators[c_addr] = True
        return {"status": 1, "authorized": c_addr}

    def deploy_contract(self) -> str:
        """Simulate contract deployment, increment block number, and return address."""
        if self.simulate_failure:
            raise ConnectionError("Mock deployment failure simulated.")
        self.current_block_number += 1
        return self.contract_address

    def record_audit_event(
        self,
        round_num: int,
        client_id: str,
        model_update_hash: str,
        decision: Union[str, int],
        reputation_change: float,
        new_reputation: float,
        record_hash: str,
    ) -> Dict[str, Any]:
        """
        Record a single audit event to the in-memory ledger.
        """
        if self.simulate_failure:
            raise ConnectionError("Transaction failed: simulated node failure.")

        if not self.authorized_coordinators.get(self.caller_address, False):
            raise PermissionError("ReputationRegistry: caller is not authorized coordinator")

        decision_code = DECISION_MAP.get(decision, 0)
        c_id_hash = hash_client_identifier(client_id)
        rep_change_scaled = int(round(reputation_change * 10000))
        new_rep_scaled = scale_reputation_for_contract(new_reputation)

        record_id = len(self.audit_records)
        self.current_block_number += 1
        self.tx_count += 1
        tx_hash = "0x" + hashlib.sha256(f"mock_tx_{self.tx_count}".encode()).hexdigest()
        timestamp = int(datetime.now(timezone.utc).timestamp())

        record = {
            "recordId": record_id,
            "round": round_num,
            "clientIdentifier": c_id_hash,
            "rawClientId": client_id,
            "modelUpdateHash": model_update_hash,
            "decision": decision_code,
            "decisionName": DECISION_NAMES[decision_code],
            "reputationChangeScaled": rep_change_scaled,
            "reputationChange": reputation_change,
            "newReputationScaled": new_rep_scaled,
            "newReputation": new_reputation,
            "timestamp": timestamp,
            "recordHash": record_hash,
            "blockNumber": self.current_block_number,
            "txHash": tx_hash,
        }

        self.audit_records.append(record)
        prev_rep_scaled = self.latest_reputations.get(c_id_hash, 7000)
        self.latest_reputations[c_id_hash] = new_rep_scaled

        # Emit simulated event
        event_entry = {
            "event": "AuditEventRecorded",
            "recordId": record_id,
            "round": round_num,
            "clientIdentifier": c_id_hash,
            "modelUpdateHash": model_update_hash,
            "decision": decision_code,
            "newReputation": new_rep_scaled,
            "timestamp": timestamp,
            "recordHash": record_hash,
            "blockNumber": self.current_block_number,
            "txHash": tx_hash,
        }
        self.emitted_events.append(event_entry)

        receipt = {
            "status": 1,
            "transactionHash": tx_hash,
            "blockNumber": self.current_block_number,
            "gasUsed": 85000,
            "recordId": record_id,
            "events": [event_entry],
        }
        return receipt

    def batch_record_audit_events(
        self,
        audit_entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Record multiple audit records in a single batch transaction.
        """
        record_ids = []
        for entry in audit_entries:
            rec = self.record_audit_event(
                round_num=entry["round"],
                client_id=entry["client_id"],
                model_update_hash=entry["model_update_hash"],
                decision=entry["decision"],
                reputation_change=entry["reputation_change"],
                new_reputation=entry["new_reputation"],
                record_hash=entry["record_hash"],
            )
            record_ids.append(rec["recordId"])

        return {
            "status": 1,
            "blockNumber": self.current_block_number,
            "recordCount": len(record_ids),
            "recordIds": record_ids,
        }

    def get_record_count(self) -> int:
        """Return total number of recorded audit records."""
        return len(self.audit_records)

    def get_record(self, record_id: int) -> Dict[str, Any]:
        """Retrieve record by ID."""
        if record_id < 0 or record_id >= len(self.audit_records):
            raise IndexError(f"Record ID {record_id} does not exist.")
        return dict(self.audit_records[record_id])

    def get_records_by_round(self, round_num: int) -> List[int]:
        """Return record IDs committed in a specific FL round."""
        return [r["recordId"] for r in self.audit_records if r["round"] == round_num]

    def get_records_by_client(self, client_id: str) -> List[int]:
        """Return record IDs for a specific client."""
        c_hash = hash_client_identifier(client_id)
        return [
            r["recordId"]
            for r in self.audit_records
            if r["clientIdentifier"] == c_hash or r.get("rawClientId") == client_id
        ]

    def get_latest_reputation(self, client_id: str) -> float:
        """Retrieve current on-chain reputation for a client (default 0.70 prior)."""
        c_hash = hash_client_identifier(client_id)
        scaled = self.latest_reputations.get(c_hash, 7000)
        return unscale_reputation_from_contract(scaled)

    def verify_record_integrity(self, record_id: int, expected_hash: str) -> bool:
        """Verify on-chain record hash matches expected canonical hash."""
        rec = self.get_record(record_id)
        return rec["recordHash"].lower() == expected_hash.lower()

    def get_audit_events(
        self,
        from_block: int = 0,
        to_block: Union[str, int] = "latest",
        round_filter: Optional[int] = None,
        client_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query and filter emitted audit events."""
        events = list(self.emitted_events)
        if round_filter is not None:
            events = [e for e in events if e.get("round") == round_filter]
        if client_filter is not None:
            c_hash = hash_client_identifier(client_filter)
            events = [e for e in events if e.get("clientIdentifier") == c_hash]
        return events
