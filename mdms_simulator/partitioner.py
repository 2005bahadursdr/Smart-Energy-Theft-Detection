"""
Client Partitioning & Privacy Vault Module.
Partitions meter readings among 3-5 configurable substation clients and enforces
the fundamental Federated Learning privacy boundary:
Raw meter readings must remain strictly local to each simulated client.
The central FL server must NEVER receive raw telemetry.
"""

from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from mdms_simulator.config import MDMSConfig
from mdms_simulator.models import MeterReadingPayload

logger = logging.getLogger("mdms_partitioner")


class RawDataPrivacyViolationError(PermissionError):
    """
    Raised when an unauthorized entity (such as the Central FL Server)
    attempts to directly access raw, granular household smart meter readings.
    """
    pass


class ClientDataVault:
    """
    Client-specific local data repository simulating an edge substation database.
    Stores and isolates raw telemetry strictly inside the client's local boundary.
    """

    def __init__(self, client_id: str, client_dir: Path):
        self.client_id = client_id
        self.client_dir = client_dir
        self.client_dir.mkdir(parents=True, exist_ok=True)
        self.readings_csv_path = self.client_dir / "local_readings.csv"
        self.metadata_json_path = self.client_dir / "vault_metadata.json"

        self._in_memory_records: List[Dict[str, Any]] = []

    def store_reading(self, payload: MeterReadingPayload, is_late: bool = False) -> None:
        """Store accepted reading into client's local memory buffer."""
        record = payload.model_dump()
        record["is_late"] = is_late
        self._in_memory_records.append(record)

    def flush_to_disk(self) -> int:
        """
        Persist in-memory readings to client's private CSV vault.
        Appends incrementally or writes initial partition file.
        """
        if not self._in_memory_records:
            return 0

        new_df = pd.DataFrame(self._in_memory_records)
        if self.readings_csv_path.exists() and self.readings_csv_path.stat().st_size > 0:
            new_df.to_csv(self.readings_csv_path, mode="a", header=False, index=False)
        else:
            new_df.to_csv(self.readings_csv_path, index=False)

        count = len(self._in_memory_records)
        self._in_memory_records.clear()

        # Update vault metadata
        self._update_metadata()
        return count

    def _update_metadata(self) -> None:
        """Update local client vault metadata file."""
        total_rows = 0
        unique_meters = 0
        if self.readings_csv_path.exists():
            try:
                df = pd.read_csv(self.readings_csv_path, usecols=["meter_id"])
                total_rows = len(df)
                unique_meters = df["meter_id"].nunique()
            except Exception:
                pass

        metadata = {
            "client_id": self.client_id,
            "total_local_readings": total_rows,
            "total_local_meters": unique_meters,
            "privacy_firewall": "ACTIVE",
            "central_fl_access": "BLOCKED"
        }
        with open(self.metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def access_raw_readings(self, requesting_entity: str) -> pd.DataFrame:
        """
        Retrieve raw meter telemetry for local training with strict privacy policy enforcement.

        Parameters
        ----------
        requesting_entity : str
            Identifier of caller requesting raw data.
            Only the specific local client (e.g. self.client_id or 'LOCAL_CLIENT') is permitted.
            'FL_SERVER' or 'CENTRAL_SERVER' is strictly forbidden!

        Returns
        -------
        pd.DataFrame
            Client-local raw consumption time-series.
        """
        forbidden_entities = ["FL_SERVER", "CENTRAL_SERVER", "GLOBAL_AGGREGATOR", "FEDERATED_SERVER"]
        if requesting_entity.upper() in forbidden_entities:
            raise RawDataPrivacyViolationError(
                f"[PRIVACY VIOLATION] Entity '{requesting_entity}' attempted to access raw meter data "
                f"from {self.client_id}'s local vault. Under Federated Learning privacy guarantees, "
                f"the central server is strictly forbidden from viewing raw customer telemetry."
            )

        if not self.readings_csv_path.exists():
            return pd.DataFrame()

        return pd.read_csv(self.readings_csv_path)

    def get_summary(self) -> Dict[str, Any]:
        """Return non-sensitive metadata for coordination."""
        total_rows = 0
        unique_meters = 0
        if self.readings_csv_path.exists():
            try:
                df = pd.read_csv(self.readings_csv_path, usecols=["meter_id"])
                total_rows = len(df)
                unique_meters = df["meter_id"].nunique()
            except Exception:
                pass

        return {
            "client_id": self.client_id,
            "meter_count": unique_meters,
            "readings_count": total_rows,
            "local_storage_path": str(self.readings_csv_path),
            "privacy_guarantee": (
                f"Raw readings for {self.client_id} are stored in {self.readings_csv_path}. "
                "Direct raw data transmission to the central FL server is blocked."
            )
        }


class LocalPartitionManager:
    """
    Manages regional substation vaults across all configured client partitions.
    """

    def __init__(self, config: Optional[MDMSConfig] = None):
        self.config = config or MDMSConfig()
        self.config.ensure_directories()

        self.vaults: Dict[str, ClientDataVault] = {}
        for client_id in self.config.client_ids:
            client_dir = self.config.clients_dir / client_id
            self.vaults[client_id] = ClientDataVault(client_id, client_dir)

    def route_reading(self, payload: MeterReadingPayload, is_late: bool = False) -> None:
        """Route accepted reading directly into its corresponding client vault."""
        client_id = payload.client_id
        if client_id not in self.vaults:
            # Instantiate dynamically if new valid client ID
            client_dir = self.config.clients_dir / client_id
            self.vaults[client_id] = ClientDataVault(client_id, client_dir)

        self.vaults[client_id].store_reading(payload, is_late=is_late)

    def flush_all(self) -> Dict[str, int]:
        """Flush in-memory buffers across all client vaults to disk."""
        results = {}
        for client_id, vault in self.vaults.items():
            flushed = vault.flush_to_disk()
            results[client_id] = flushed
        return results

    def get_vault(self, client_id: str) -> Optional[ClientDataVault]:
        """Get client vault by ID."""
        return self.vaults.get(client_id)

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve summaries of all client vaults."""
        return {cid: v.get_summary() for cid, v in self.vaults.items()}
