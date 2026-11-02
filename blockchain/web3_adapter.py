"""
Web3 Blockchain Adapter for ReputationRegistry Smart Contract.

Integrates with local Ethereum-compatible chains (e.g. EthereumTesterProvider, Anvil, Hardhat, Ganache)
via Web3.py.

CRITICAL INVARIANTS:
1. "Blockchain records evidence; it does not itself detect or prevent poisoning":
   On-chain commits provide immutable, tamper-evident auditability. Hashes do not prove
   that an aggregation decision was mathematically correct.
2. ZERO RAW DATA & ZERO WEIGHTS ON-CHAIN:
   Only pseudonymous client hashes, model update hashes, discrete decisions, and scalar reputations
   are submitted in transactions.
3. Credentials stored in environment variables, NEVER committed to Git.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from eth_account import Account
import numpy as np
from web3 import Web3
from web3.exceptions import Web3Exception

from blockchain.canonical_hasher import (
    hash_client_identifier,
    scale_reputation_for_contract,
    unscale_reputation_from_contract,
)

from web3.logs import DISCARD

logger = logging.getLogger("web3_blockchain_adapter")

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


def to_bytes32(val: Union[str, bytes]) -> bytes:
    """Helper to convert hex string or bytes to exact 32-byte representation."""
    if isinstance(val, bytes):
        return val.rjust(32, b"\x00") if len(val) <= 32 else val[:32]
    val_str = str(val).strip()
    hex_str = val_str[2:] if val_str.startswith("0x") else val_str
    b = bytes.fromhex(hex_str)
    return b.rjust(32, b"\x00") if len(b) <= 32 else b[:32]


class Web3BlockchainAdapter:
    """
    Ethereum Web3 Adapter connecting to ReputationRegistry smart contract.
    """

    def __init__(
        self,
        provider_type: Optional[str] = None,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        account_address: Optional[str] = None,
        contract_address: Optional[str] = None,
        artifact_path: Optional[Union[str, Path]] = None,
    ):
        # 1. Resolve Provider Configuration from arguments or environment
        self.provider_type = (provider_type or os.getenv("BLOCKCHAIN_PROVIDER_TYPE", "eth_tester")).lower()
        self.rpc_url = rpc_url or os.getenv("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545")

        if self.provider_type == "eth_tester":
            from web3 import EthereumTesterProvider
            self.w3 = Web3(EthereumTesterProvider())
            logger.info("Initialized Web3 with in-memory EthereumTesterProvider.")
        else:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            logger.info(f"Initialized Web3 with HTTPProvider at {self.rpc_url}.")

        # 2. Resolve Credentials (from environment, never hardcoded in source)
        self.private_key = private_key or os.getenv("BLOCKCHAIN_PRIVATE_KEY")
        if self.private_key:
            self.account = Account.from_key(self.private_key)
            self.account_address = self.account.address
        elif self.provider_type == "eth_tester" and self.w3.eth.accounts:
            self.account = None
            self.account_address = self.w3.eth.accounts[0]
        else:
            raw_addr = account_address or os.getenv("BLOCKCHAIN_ACCOUNT_ADDRESS")
            self.account_address = Web3.to_checksum_address(raw_addr) if raw_addr else None
            self.account = None

        # If on local eth_tester, ensure configured account is funded
        if self.provider_type == "eth_tester" and self.w3.eth.accounts and self.account_address:
            try:
                if self.w3.eth.get_balance(self.account_address) == 0:
                    self.w3.eth.send_transaction({
                        "from": self.w3.eth.accounts[0],
                        "to": self.account_address,
                        "value": self.w3.to_wei(100, "ether"),
                    })
            except Exception as e:
                logger.debug(f"eth_tester account pre-funding notice: {e}")

        # 3. Load Smart Contract Artifact (ABI and Bytecode)
        artifact_p = Path(artifact_path) if artifact_path else Path("blockchain/contracts/ReputationRegistry.json")
        if not artifact_p.exists():
            raise FileNotFoundError(f"Contract artifact not found at {artifact_p}. Run contract compilation first.")

        with open(artifact_p, "r", encoding="utf-8") as f:
            artifact_data = json.load(f)

        self.abi = artifact_data.get("abi", [])
        self.bytecode = artifact_data.get("bytecode", "")

        # 4. Bind Contract Instance if address provided
        raw_contract_addr = contract_address or os.getenv("REPUTATION_REGISTRY_CONTRACT_ADDRESS")
        self.contract_address: Optional[str] = None
        self.contract = None

        if raw_contract_addr and Web3.is_address(raw_contract_addr):
            self.contract_address = Web3.to_checksum_address(raw_contract_addr)
            self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.abi)
            logger.info(f"Bound ReputationRegistry contract at address: {self.contract_address}")

    def is_connected(self) -> bool:
        """Verify node connectivity."""
        try:
            return bool(self.w3.is_connected())
        except Exception:
            return False

    def deploy_contract(self) -> str:
        """
        Deploy ReputationRegistry contract to the active chain.
        Returns newly deployed checksum contract address.
        """
        if not self.bytecode:
            raise ValueError("Cannot deploy contract: bytecode is empty in artifact.")

        contract_factory = self.w3.eth.contract(abi=self.abi, bytecode=self.bytecode)

        if self.account:
            nonce = self.w3.eth.get_transaction_count(self.account_address)
            tx = contract_factory.constructor().build_transaction({
                "from": self.account_address,
                "nonce": nonce,
                "gas": 3000000,
                "gasPrice": self.w3.eth.gas_price,
            })
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        else:
            tx_hash = contract_factory.constructor().transact({"from": self.account_address})

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status != 1:
            raise RuntimeError(f"Contract deployment failed! Transaction status: {receipt.status}")

        self.contract_address = receipt.contractAddress
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.abi)
        logger.info(f"Successfully deployed ReputationRegistry at: {self.contract_address} (Block {receipt.blockNumber})")
        return self.contract_address

    def _ensure_contract(self) -> None:
        """Helper to ensure contract is deployed/bound before transactions."""
        if self.contract is None:
            logger.info("Contract instance not found. Auto-deploying ReputationRegistry...")
            self.deploy_contract()

    def _send_transaction(self, contract_function) -> Dict[str, Any]:
        """
        Helper to send, sign (if private key available), and wait for a transaction receipt.
        """
        self._ensure_contract()
        if self.account:
            nonce = self.w3.eth.get_transaction_count(self.account_address)
            tx = contract_function.build_transaction({
                "from": self.account_address,
                "nonce": nonce,
                "gas": 1500000,
                "gasPrice": self.w3.eth.gas_price,
            })
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        else:
            tx_hash = contract_function.transact({"from": self.account_address})

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status != 1:
            raise RuntimeError(f"Transaction reverted! Status: {receipt.status}, Tx: {receipt.transactionHash.hex()}")
        return receipt

    def authorize_coordinator(self, coordinator_address: str) -> Dict[str, Any]:
        """Authorize an address to commit audit records."""
        self._ensure_contract()
        c_addr = Web3.to_checksum_address(coordinator_address)
        fn = self.contract.functions.authorizeCoordinator(c_addr)
        receipt = self._send_transaction(fn)
        return {
            "status": receipt.status,
            "blockNumber": receipt.blockNumber,
            "transactionHash": receipt.transactionHash.hex(),
            "authorized": c_addr,
        }

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
        Submit a single audit event transaction to ReputationRegistry.
        """
        self._ensure_contract()

        # Format inputs with exact 32-byte representation
        c_id_bytes = to_bytes32(hash_client_identifier(client_id))
        m_hash_bytes = to_bytes32(model_update_hash)
        r_hash_bytes = to_bytes32(record_hash)
        decision_code = DECISION_MAP.get(decision, 0)
        rep_change_scaled = int(round(reputation_change * 10000))
        new_rep_scaled = scale_reputation_for_contract(new_reputation)

        fn = self.contract.functions.recordAuditEvent(
            int(round_num),
            c_id_bytes,
            m_hash_bytes,
            int(decision_code),
            int(rep_change_scaled),
            int(new_rep_scaled),
            r_hash_bytes,
        )

        receipt = self._send_transaction(fn)

        # Parse emitted event to extract recordId
        logs = self.contract.events.AuditEventRecorded().process_receipt(receipt, errors=DISCARD)
        record_id = logs[0]["args"]["recordId"] if logs else None

        return {
            "status": receipt.status,
            "transactionHash": receipt.transactionHash.hex(),
            "blockNumber": receipt.blockNumber,
            "gasUsed": receipt.gasUsed,
            "recordId": record_id,
            "round": round_num,
            "client_id": client_id,
            "decision": DECISION_NAMES[decision_code],
            "new_reputation": new_reputation,
        }

    def batch_record_audit_events(
        self,
        audit_entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Submit multiple audit records in a single batch transaction.
        """
        self._ensure_contract()

        tuple_inputs = []
        for entry in audit_entries:
            c_id_bytes = to_bytes32(hash_client_identifier(entry["client_id"]))
            m_hash_bytes = to_bytes32(entry["model_update_hash"])
            r_hash_bytes = to_bytes32(entry["record_hash"])
            decision_code = DECISION_MAP.get(entry["decision"], 0)
            rep_change_scaled = int(round(entry["reputation_change"] * 10000))
            new_rep_scaled = scale_reputation_for_contract(entry["new_reputation"])

            tuple_inputs.append((
                int(entry["round"]),
                c_id_bytes,
                m_hash_bytes,
                int(decision_code),
                int(rep_change_scaled),
                int(new_rep_scaled),
                r_hash_bytes,
            ))

        fn = self.contract.functions.batchRecordAuditEvents(tuple_inputs)
        receipt = self._send_transaction(fn)

        logs = self.contract.events.AuditEventRecorded().process_receipt(receipt, errors=DISCARD)
        record_ids = [l["args"]["recordId"] for l in logs]

        return {
            "status": receipt.status,
            "transactionHash": receipt.transactionHash.hex(),
            "blockNumber": receipt.blockNumber,
            "gasUsed": receipt.gasUsed,
            "recordCount": len(tuple_inputs),
            "recordIds": record_ids,
        }

        return {
            "status": receipt.status,
            "transactionHash": receipt.transactionHash.hex(),
            "blockNumber": receipt.blockNumber,
            "gasUsed": receipt.gasUsed,
            "recordCount": len(tuple_inputs),
            "recordIds": record_ids,
        }

    def get_record_count(self) -> int:
        """Query total count of records stored in contract."""
        self._ensure_contract()
        return self.contract.functions.getRecordCount().call()

    def get_record(self, record_id: int) -> Dict[str, Any]:
        """Query a single audit record by its ID."""
        self._ensure_contract()
        raw = self.contract.functions.getRecord(int(record_id)).call()
        # raw is tuple: (round, clientIdentifier, modelUpdateHash, decision, repChange, newRep, timestamp, recordHash)
        return {
            "recordId": record_id,
            "round": raw[0],
            "clientIdentifier": "0x" + raw[1].hex(),
            "modelUpdateHash": "0x" + raw[2].hex(),
            "decision": raw[3],
            "decisionName": DECISION_NAMES.get(raw[3], "UNKNOWN"),
            "reputationChangeScaled": raw[4],
            "reputationChange": round(float(raw[4]) / 10000.0, 4),
            "newReputationScaled": raw[5],
            "newReputation": unscale_reputation_from_contract(raw[5]),
            "timestamp": raw[6],
            "recordHash": "0x" + raw[7].hex(),
        }

    def get_records_by_round(self, round_num: int) -> List[int]:
        """Query record IDs recorded for a given FL round."""
        self._ensure_contract()
        return list(self.contract.functions.getRecordsByRound(int(round_num)).call())

    def get_records_by_client(self, client_id: str) -> List[int]:
        """Query record IDs recorded for a given client."""
        self._ensure_contract()
        c_id_bytes = to_bytes32(hash_client_identifier(client_id))
        return list(self.contract.functions.getRecordsByClient(c_id_bytes).call())

    def get_latest_reputation(self, client_id: str) -> float:
        """Query current on-chain reputation for client."""
        self._ensure_contract()
        c_id_bytes = to_bytes32(hash_client_identifier(client_id))
        scaled = self.contract.functions.getLatestReputation(c_id_bytes).call()
        return unscale_reputation_from_contract(scaled)

    def verify_record_integrity(self, record_id: int, expected_hash: str) -> bool:
        """Verify on-chain record hash matches expected hash."""
        self._ensure_contract()
        exp_bytes = to_bytes32(expected_hash)
        return bool(self.contract.functions.verifyRecordIntegrity(int(record_id), exp_bytes).call())

    def get_audit_events(
        self,
        from_block: int = 0,
        to_block: Union[str, int] = "latest",
        round_filter: Optional[int] = None,
        client_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve and parse emitted AuditEventRecorded event logs."""
        self._ensure_contract()
        event_filter = self.contract.events.AuditEventRecorded()

        # Construct argument filters if requested
        args_filter = {}
        if round_filter is not None:
            args_filter["round"] = int(round_filter)
        if client_filter is not None:
            args_filter["clientIdentifier"] = to_bytes32(hash_client_identifier(client_filter))

        logs = event_filter.get_logs(from_block=from_block, to_block=to_block, argument_filters=args_filter if args_filter else None)

        parsed_events = []
        for l in logs:
            args = l["args"]
            parsed_events.append({
                "recordId": args["recordId"],
                "round": args["round"],
                "clientIdentifier": "0x" + args["clientIdentifier"].hex(),
                "modelUpdateHash": "0x" + args["modelUpdateHash"].hex(),
                "decision": args["decision"],
                "decisionName": DECISION_NAMES.get(args["decision"], "UNKNOWN"),
                "newReputation": unscale_reputation_from_contract(args["newReputation"]),
                "timestamp": args["timestamp"],
                "recordHash": "0x" + args["recordHash"].hex(),
                "blockNumber": l["blockNumber"],
                "transactionHash": l["transactionHash"].hex(),
            })

        return parsed_events
