"""
Canonical Hashing & Data Safety Utilities for Blockchain Audit Subsystem.

CRITICAL PRIVACY & SECURITY GUARDRAILS:
1. ZERO RAW DATA & ZERO WEIGHTS ON-CHAIN:
   Never put raw meter telemetry, personal consumer information (PII),
   or full deep neural network weights on the blockchain.
   Only cryptographic hashes (SHA-256 / Keccak-256), pseudonymous client identifiers,
   discrete decisions, and scalar reputation metrics are committed.

2. EPISTEMIC NOTICE:
   Hashes do not prove that an aggregation decision was mathematically correct;
   they only provide immutable, tamper-evident proof of what was observed,
   decided, and committed at round t.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Union
import numpy as np
from web3 import Web3

logger = logging.getLogger("blockchain_hasher")

# Forbidden raw keys that must never appear in canonical on-chain audit records
FORBIDDEN_RAW_KEYS = {
    "consumption_kwh",
    "raw_readings",
    "meter_readings",
    "consumer_id",
    "customer_name",
    "address",
    "model_weights",
    "full_parameters",
    "raw_gradients",
}


def hash_parameter_arrays(parameter_arrays: List[np.ndarray]) -> str:
    """
    Compute deterministic SHA-256 hash of a list of numpy parameter arrays (weights/gradients).
    Returns 32-byte hex string (with '0x' prefix).
    """
    if not parameter_arrays:
        return "0x" + "0" * 64

    hasher = hashlib.sha256()
    for arr in parameter_arrays:
        # Enforce contiguous C-order float32 representation for determinism
        contiguous = np.ascontiguousarray(arr, dtype=np.float32)
        hasher.update(contiguous.tobytes())

    return "0x" + hasher.hexdigest()


def hash_client_identifier(client_id: str) -> str:
    """
    Compute deterministic Keccak-256 pseudonymous identifier for a client substation ID.
    Returns 32-byte hex string (with '0x' prefix).
    """
    clean_id = client_id.strip()
    raw_hex = Web3.keccak(text=clean_id).hex()
    return raw_hex if raw_hex.startswith("0x") else f"0x{raw_hex}"


def compute_canonical_record_hash(audit_data: Dict[str, Any]) -> str:
    """
    Compute canonical Keccak-256 hash of an audit event payload.
    Enforces key sorting, JSON normalization, and data privacy isolation.
    Returns 32-byte hex string (with '0x' prefix).
    """
    # 1. Enforce privacy boundary check
    lower_keys = {k.lower() for k in audit_data.keys()}
    violating = lower_keys.intersection(FORBIDDEN_RAW_KEYS)
    if violating:
        raise ValueError(
            f"PRIVACY VIOLATION DETECTED: Audit record contains forbidden raw data keys: {violating}. "
            f"Never submit raw consumption telemetry, consumer PII, or raw weights to the blockchain."
        )

    # 2. Canonical JSON serialization (sorted keys, no spaces)
    canonical_json = json.dumps(audit_data, sort_keys=True, separators=(",", ":"))
    raw_hex = Web3.keccak(text=canonical_json).hex()
    return raw_hex if raw_hex.startswith("0x") else f"0x{raw_hex}"


def verify_record_integrity(audit_data: Dict[str, Any], expected_hash: str) -> bool:
    """
    Verify whether an audit record's canonical hash matches its on-chain committed hash.
    Detects any post-hoc tampering or discrepancy.
    """
    computed = compute_canonical_record_hash(audit_data).lower()
    exp = expected_hash.lower()
    computed_clean = computed[2:] if computed.startswith("0x") else computed
    exp_clean = exp[2:] if exp.startswith("0x") else exp
    return computed_clean == exp_clean


def scale_reputation_for_contract(rep_float: float) -> int:
    """
    Scale floating-point reputation in [0.0, 1.0] to integer scaled by 10,000 (0..10,000).
    """
    clamped = max(0.0, min(1.0, float(rep_float)))
    return int(round(clamped * 10000))


def unscale_reputation_from_contract(rep_int: int) -> float:
    """
    Convert scaled integer reputation (0..10,000) back to float in [0.0, 1.0].
    """
    return round(float(rep_int) / 10000.0, 4)
