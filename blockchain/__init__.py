"""
Blockchain Audit Subsystem for Blockchain-Based Byzantine-Resilient FL.

Guiding Principles:
1. Blockchain records evidence; it does NOT itself detect or prevent poisoning.
2. Hashes do not prove that decisions were correct—they only provide immutable,
   tamper-evident proof of coordinator claims.
3. NEVER PUT RAW READINGS, PERSONAL INFORMATION OR MODEL WEIGHTS ON-CHAIN.
"""

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

__all__ = [
    "MockBlockchainAdapter",
    "Web3BlockchainAdapter",
    "commit_fl_round_audit",
    "compute_canonical_record_hash",
    "get_blockchain_adapter",
    "hash_client_identifier",
    "hash_parameter_arrays",
    "scale_reputation_for_contract",
    "unscale_reputation_from_contract",
    "verify_record_integrity",
]
