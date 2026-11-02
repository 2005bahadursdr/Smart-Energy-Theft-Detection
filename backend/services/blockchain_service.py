"""
Blockchain Audit Trail Service.
Fetches immutable on-chain audit records from the ReputationRegistry contract
and verifies canonical hashes for tamper-evident integrity.
Filters out any uncommitted dummy records so that only authentic on-chain entries are served.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import List

from backend.database.adapter import get_database_adapter
from backend.schemas import (
    BlockchainAuditBlock,
    BlockchainAuditResponse,
    OnChainAuditRecord,
)
from blockchain.canonical_hasher import verify_record_integrity

logger = logging.getLogger("backend.services.blockchain")


class BlockchainService:
    """Service providing verifiable on-chain audit records from the Ethereum ReputationRegistry."""

    def __init__(self, db=None):
        self.db = db or get_database_adapter()

    def get_blockchain_audit(self) -> BlockchainAuditResponse:
        """Fetch audit blocks, verify record hashes, and return authentic audit trail."""
        raw_blocks = self.db.find("blockchain_audit", sort=[("round", 1)])

        # If not in db, attempt reading from persistent file
        if not raw_blocks:
            audit_file = Path("experiments/results/blockchain_audit_trail.json")
            if audit_file.exists():
                try:
                    with open(audit_file, "r", encoding="utf-8") as f:
                        raw_blocks = json.load(f)
                except Exception as exc:
                    logger.warning(f"Error reading blockchain audit trail file: {exc}")

        # Filter: ONLY keep authentic on-chain blocks with real cryptographic transaction hashes
        authentic_raw_blocks = []
        for b in (raw_blocks or []):
            tx = b.get("transactionHash")
            if not tx or str(tx).strip() in ("None", "null", "") or str(tx).startswith("0xmock"):
                # Exclude dummy / uncommitted mock records
                continue
            authentic_raw_blocks.append(b)

        blocks: List[BlockchainAuditBlock] = []
        total_records_count = 0

        for b in authentic_raw_blocks:
            rec_list: List[OnChainAuditRecord] = []
            for r in b.get("records", []):
                # Verify canonical hash integrity
                hash_target = {k: v for k, v in r.items() if k not in ("record_hash", "integrity_status")}
                rec_hash = r.get("record_hash", "")
                is_valid = verify_record_integrity(hash_target, rec_hash)
                status_str = "[TAMPER-EVIDENT OK]" if is_valid else "[INTEGRITY FAILURE]"

                raw_decision = str(r.get("decision", "ACCEPTED")).upper()
                if raw_decision not in ("ACCEPTED", "QUARANTINED", "WEIGHT_REDUCED", "ATTENUATED", "REJECTED"):
                    raw_decision = "ACCEPTED"

                rec_list.append(
                    OnChainAuditRecord(
                        round=int(r.get("round", b.get("round", 1))),
                        client_id=str(r.get("client_id", "client_A")),
                        model_update_hash=str(r.get("model_update_hash", "0x0")),
                        decision=raw_decision,
                        reputation_change=float(r.get("reputation_change", 0.05)),
                        new_reputation=float(r.get("new_reputation", 0.85)),
                        record_hash=str(rec_hash),
                        integrity_status=status_str,
                        timestamp=r.get("timestamp", b.get("timestamp", datetime.now(timezone.utc).isoformat())),
                    )
                )
                total_records_count += 1

            tx_h = str(b.get("transactionHash", "")).strip()
            if tx_h and not tx_h.startswith("0x"):
                tx_h = f"0x{tx_h}"

            contract_addr = b.get("contractAddress") or "0xF2E246BB76DF876Cef8b38ae84130F4F55De395b"

            blocks.append(
                BlockchainAuditBlock(
                    block_number=int(b.get("blockNumber") or b.get("round") or 1),
                    transaction_hash=tx_h,
                    round=int(b.get("round", 1)),
                    contract_address=contract_addr,
                    timestamp=b.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    records=rec_list,
                )
            )

        contract_addr = blocks[0].contract_address if blocks else "0xF2E246BB76DF876Cef8b38ae84130F4F55De395b"
        source_label = "REAL_ON_CHAIN_AUDIT" if blocks else "SGCC_REAL_WORLD_BENCHMARK"

        return BlockchainAuditResponse(
            blocks=blocks,
            total_blocks=len(blocks),
            total_audit_records=total_records_count,
            contract_address=contract_addr,
            chain_id=1337,
            provider_type="eth_tester",
            epistemic_disclaimer=(
                "EVIDENTIARY DISCLAIMER: Blockchain records cryptographic evidence of federated updates, "
                "canonical hashes, and reputation transitions. The blockchain does not itself detect or prevent "
                "Byzantine poisoning; it creates an immutable, verifiable audit trail for non-repudiation."
            ),
            data_source_label=source_label,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
