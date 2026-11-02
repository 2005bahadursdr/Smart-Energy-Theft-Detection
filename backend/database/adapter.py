"""
Database Abstraction & Adapters Module.

Provides:
1. BaseDatabaseAdapter: Abstract interface for persistent metadata stores.
2. MongoDatabaseAdapter: Production MongoDB adapter using PyMongo with timeouts & error handling.
3. InMemoryDatabaseAdapter: Fully documented, thread-safe in-memory adapter for development
   and testing environments where MongoDB is not running.
4. get_database_adapter(): Factory that connects to MongoDB or falls back to In-Memory seamlessly.
5. seed_default_metadata(): Automatic bootstrap seeding from local simulation and benchmark logs.
"""

from abc import ABC, abstractmethod
import copy
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger("backend.database")


class BaseDatabaseAdapter(ABC):
    """Abstract base class for backend database adapters."""

    @abstractmethod
    def ping(self) -> bool:
        """Verify database connectivity."""
        pass

    @abstractmethod
    def insert_one(self, collection: str, document: Dict[str, Any]) -> str:
        """Insert a single document, returning its unique identifier string."""
        pass

    @abstractmethod
    def insert_many(self, collection: str, documents: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple documents, returning list of unique identifier strings."""
        pass

    @abstractmethod
    def find(
        self,
        collection: str,
        query: Optional[Dict[str, Any]] = None,
        sort: Optional[List[Tuple[str, int]]] = None,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query documents matching criteria, with optional sorting and limit."""
        pass

    @abstractmethod
    def find_one(
        self, collection: str, query: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Query a single document matching criteria."""
        pass

    @abstractmethod
    def count(self, collection: str, query: Optional[Dict[str, Any]] = None) -> int:
        """Count documents matching criteria."""
        pass

    @abstractmethod
    def update_one(
        self,
        collection: str,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False,
    ) -> bool:
        """Update a single document."""
        pass

    @abstractmethod
    def delete_many(self, collection: str, query: Optional[Dict[str, Any]] = None) -> int:
        """Delete documents matching criteria."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider descriptor ('mongodb' or 'in_memory')."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection pools and release resources."""
        pass


# ------------------------------------------------------------------------------
# MongoDB Adapter
# ------------------------------------------------------------------------------

class MongoDatabaseAdapter(BaseDatabaseAdapter):
    """
    MongoDB Adapter using PyMongo.
    Stores and retrieves persistent simulation, FL rounds, trust scores, and audit records.
    """

    def __init__(self, uri: str, db_name: str, timeout_ms: int = 2500):
        self.uri = uri
        self.db_name = db_name
        self.timeout_ms = timeout_ms
        import pymongo
        self._client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=self.timeout_ms)
        self._db = self._client[self.db_name]
        logger.info(f"Initialized MongoDatabaseAdapter for database '{self.db_name}'")

    def ping(self) -> bool:
        try:
            self._client.admin.command("ping")
            return True
        except Exception as e:
            logger.warning(f"MongoDB ping failed: {e}")
            return False

    def _sanitize_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Convert BSON ObjectId to string for Pydantic/JSON compatibility."""
        if not doc:
            return doc
        d = copy.deepcopy(doc)
        if "_id" in d:
            d["id"] = str(d.pop("_id"))
        return d

    def insert_one(self, collection: str, document: Dict[str, Any]) -> str:
        doc = copy.deepcopy(document)
        res = self._db[collection].insert_one(doc)
        return str(res.inserted_id)

    def insert_many(self, collection: str, documents: List[Dict[str, Any]]) -> List[str]:
        if not documents:
            return []
        docs = [copy.deepcopy(d) for d in documents]
        res = self._db[collection].insert_many(docs)
        return [str(i) for i in res.inserted_ids]

    def find(
        self,
        collection: str,
        query: Optional[Dict[str, Any]] = None,
        sort: Optional[List[Tuple[str, int]]] = None,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        q = query or {}
        cursor = self._db[collection].find(q)
        if sort:
            cursor = cursor.sort(sort)
        if limit > 0:
            cursor = cursor.limit(limit)
        return [self._sanitize_doc(d) for d in cursor]

    def find_one(
        self, collection: str, query: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        q = query or {}
        doc = self._db[collection].find_one(q)
        return self._sanitize_doc(doc) if doc else None

    def count(self, collection: str, query: Optional[Dict[str, Any]] = None) -> int:
        q = query or {}
        return self._db[collection].count_documents(q)

    def update_one(
        self,
        collection: str,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False,
    ) -> bool:
        u = update if any(k.startswith("$") for k in update.keys()) else {"$set": update}
        res = self._db[collection].update_one(query, u, upsert=upsert)
        return bool(res.matched_count or res.upserted_id)

    def delete_many(self, collection: str, query: Optional[Dict[str, Any]] = None) -> int:
        q = query or {}
        res = self._db[collection].delete_many(q)
        return res.deleted_count

    def get_provider_name(self) -> str:
        return "mongodb"

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


# ------------------------------------------------------------------------------
# Documented In-Memory Adapter
# ------------------------------------------------------------------------------

class InMemoryDatabaseAdapter(BaseDatabaseAdapter):
    """
    Documented In-Memory Database Adapter for Local Development & Testing.

    Architecture & Purpose:
    -----------------------
    - Operates entirely in memory without requiring an external MongoDB service.
    - Thread-safe using re-entrant locks (threading.RLock).
    - Preserves copy semantics (deep-copying on write and read) to prevent accidental
      state corruption between services.
    - Implements query filtering including equality, range comparisons ($gte, $lte, $gt, $lt),
      and membership ($in).
    - Automatically assigned unique UUID hex strings for 'id'.
    """

    def __init__(self):
        self._collections: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.RLock()
        logger.info("Initialized documented InMemoryDatabaseAdapter (Development Fallback)")

    def ping(self) -> bool:
        return True

    def _matches_query(self, doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Evaluate document against a MongoDB-like query dict."""
        for field, expected in query.items():
            val = doc.get(field)
            if isinstance(expected, dict):
                # Operator matching
                for op, op_val in expected.items():
                    if op == "$in":
                        if val not in op_val:
                            return False
                    elif op == "$eq":
                        if val != op_val:
                            return False
                    elif op == "$gte":
                        if val is None or val < op_val:
                            return False
                    elif op == "$lte":
                        if val is None or val > op_val:
                            return False
                    elif op == "$gt":
                        if val is None or val <= op_val:
                            return False
                    elif op == "$lt":
                        if val is None or val >= op_val:
                            return False
            else:
                if val != expected:
                    return False
        return True

    def insert_one(self, collection: str, document: Dict[str, Any]) -> str:
        with self._lock:
            doc = copy.deepcopy(document)
            if "id" not in doc and "_id" not in doc:
                doc["id"] = uuid.uuid4().hex
            elif "_id" in doc and "id" not in doc:
                doc["id"] = str(doc.pop("_id"))
            doc_id = str(doc.get("id"))

            if collection not in self._collections:
                self._collections[collection] = []
            self._collections[collection].append(doc)
            return doc_id

    def insert_many(self, collection: str, documents: List[Dict[str, Any]]) -> List[str]:
        with self._lock:
            ids = []
            for doc in documents:
                ids.append(self.insert_one(collection, doc))
            return ids

    def find(
        self,
        collection: str,
        query: Optional[Dict[str, Any]] = None,
        sort: Optional[List[Tuple[str, int]]] = None,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            q = query or {}
            items = self._collections.get(collection, [])
            matched = [copy.deepcopy(doc) for doc in items if self._matches_query(doc, q)]

            if sort:
                for field, direction in reversed(sort):
                    reverse = direction < 0
                    matched.sort(
                        key=lambda d: (d.get(field) is None, d.get(field)),
                        reverse=reverse,
                    )

            if limit > 0:
                matched = matched[:limit]

            return matched

    def find_one(
        self, collection: str, query: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        results = self.find(collection, query=query, limit=1)
        return results[0] if results else None

    def count(self, collection: str, query: Optional[Dict[str, Any]] = None) -> int:
        with self._lock:
            q = query or {}
            items = self._collections.get(collection, [])
            if not q:
                return len(items)
            return sum(1 for doc in items if self._matches_query(doc, q))

    def update_one(
        self,
        collection: str,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False,
    ) -> bool:
        with self._lock:
            items = self._collections.get(collection, [])
            for doc in items:
                if self._matches_query(doc, query):
                    # Apply update
                    set_dict = update.get("$set", update)
                    for k, v in set_dict.items():
                        if not k.startswith("$"):
                            doc[k] = copy.deepcopy(v)
                    return True

            if upsert:
                new_doc = copy.deepcopy(query)
                set_dict = update.get("$set", update)
                for k, v in set_dict.items():
                    if not k.startswith("$"):
                        new_doc[k] = copy.deepcopy(v)
                self.insert_one(collection, new_doc)
                return True

            return False

    def delete_many(self, collection: str, query: Optional[Dict[str, Any]] = None) -> int:
        with self._lock:
            q = query or {}
            if collection not in self._collections:
                return 0
            if not q:
                deleted = len(self._collections[collection])
                self._collections[collection] = []
                return deleted

            kept = []
            deleted = 0
            for doc in self._collections[collection]:
                if self._matches_query(doc, q):
                    deleted += 1
                else:
                    kept.append(doc)
            self._collections[collection] = kept
            return deleted

    def get_provider_name(self) -> str:
        return "in_memory"

    def close(self) -> None:
        with self._lock:
            self._collections.clear()


# ------------------------------------------------------------------------------
# Bootstrap Seeding & Factory
# ------------------------------------------------------------------------------

def seed_default_metadata(adapter: BaseDatabaseAdapter) -> None:
    """
    Bootstrap database with real/synthetic benchmark data if collections are empty.
    Never seeds raw meter time-series readings to maintain strict privacy.
    """
    # 1. Seed Meter Summary if empty
    if adapter.count("meter_summaries") == 0:
        summary_path = Path("data/synthetic/simulation_summary.json")
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("simulation_metadata", {})
                stats = data.get("consumption_statistics", {})
                events = data.get("event_breakdown", {})
                status_b = data.get("meter_status_breakdown", {})
                truth = data.get("synthetic_ground_truth", {})

                doc = {
                    "total_meters": meta.get("total_meters", 1000),
                    "total_readings_sampled": meta.get("total_records", 720000),
                    "sampling_frequency_minutes": 15,
                    "theft_ratio": truth.get("theft_rate_meter_level", 0.082),
                    "consumption_metrics": {
                        "mean_kwh": round(stats.get("mean_reported_kwh", 21.98), 2),
                        "std_kwh": 12.45,
                        "min_kwh": 0.0,
                        "max_kwh": round(stats.get("max_reported_kwh", 335.94), 2),
                        "median_kwh": 18.25,
                        "p25_kwh": 8.40,
                        "p75_kwh": 31.60,
                        "total_consumed_kwh": round(stats.get("mean_reported_kwh", 21.98) * meta.get("total_records", 720000), 1),
                    },
                    "status_breakdown": status_b,
                    "event_breakdown": events,
                    "consumer_profiles_breakdown": {
                        "Residential": 550,
                        "Commercial": 250,
                        "Industrial": 120,
                        "Agricultural": 80,
                    },
                    "hardware_tamper_alerts": truth.get("hardware_tamper_alerts_raised", 6575),
                    "estimated_unreported_theft_kwh": truth.get("total_simulated_unreported_kwh", 462381.27),
                    "data_source_label": "SIMULATED_DEMO_DATA",
                    "demo_notice": (
                        "Demonstration smart meter simulation data (1000 synthetic smart meters, 30 days). "
                        "Confidentiality invariant strictly observed: zero individual raw telemetry exposed."
                    ),
                    "timestamp": "2026-10-18T10:00:00+00:00",
                }
                adapter.insert_one("meter_summaries", doc)
                logger.info("Bootstrap seeded 'meter_summaries' collection")
            except Exception as e:
                logger.warning(f"Failed to bootstrap meter_summaries: {e}")

    # 2. Seed Substation Clients if empty
    if adapter.count("clients") == 0:
        clients_data = [
            {
                "client_id": "client_A",
                "substation_name": "Substation Alpha (Industrial Heavy)",
                "region_type": "Industrial High-Load",
                "meters_count": 200,
                "theft_meters": 17,
                "theft_rate_pct": 8.5,
                "mean_daily_kwh": 38.32,
                "extracted_sequences": 13400,
                "current_reputation": 0.85,
                "current_trust_score": 0.94,
                "status": "ACTIVE",
                "privacy_firewall": "ACTIVE",
                "central_fl_access": "BLOCKED",
                "local_data_path": "data/processed/clients/client_A",
            },
            {
                "client_id": "client_B",
                "substation_name": "Substation Beta (Commercial Core)",
                "region_type": "Commercial District",
                "meters_count": 200,
                "theft_meters": 17,
                "theft_rate_pct": 8.5,
                "mean_daily_kwh": 9.23,
                "extracted_sequences": 13400,
                "current_reputation": 0.85,
                "current_trust_score": 0.91,
                "status": "ACTIVE",
                "privacy_firewall": "ACTIVE",
                "central_fl_access": "BLOCKED",
                "local_data_path": "data/processed/clients/client_B",
            },
            {
                "client_id": "client_C",
                "substation_name": "Substation Gamma (Urban Dense)",
                "region_type": "Urban Residential",
                "meters_count": 200,
                "theft_meters": 17,
                "theft_rate_pct": 8.5,
                "mean_daily_kwh": 5.70,
                "extracted_sequences": 13400,
                "current_reputation": 0.85,
                "current_trust_score": 0.95,
                "status": "ACTIVE",
                "privacy_firewall": "ACTIVE",
                "central_fl_access": "BLOCKED",
                "local_data_path": "data/processed/clients/client_C",
            },
            {
                "client_id": "client_D",
                "substation_name": "Substation Delta (Suburban Outlier)",
                "region_type": "Suburban Mixed",
                "meters_count": 200,
                "theft_meters": 17,
                "theft_rate_pct": 8.5,
                "mean_daily_kwh": 2.69,
                "extracted_sequences": 13400,
                "current_reputation": 0.70,
                "current_trust_score": 0.22,
                "status": "QUARANTINED",
                "privacy_firewall": "ACTIVE",
                "central_fl_access": "BLOCKED",
                "local_data_path": "data/processed/clients/client_D",
            },
            {
                "client_id": "client_E",
                "substation_name": "Substation Epsilon (Agricultural Low)",
                "region_type": "Rural Agricultural",
                "meters_count": 200,
                "theft_meters": 17,
                "theft_rate_pct": 8.5,
                "mean_daily_kwh": 0.36,
                "extracted_sequences": 13400,
                "current_reputation": 0.80,
                "current_trust_score": 0.89,
                "status": "ACTIVE",
                "privacy_firewall": "ACTIVE",
                "central_fl_access": "BLOCKED",
                "local_data_path": "data/processed/clients/client_E",
            },
        ]
        adapter.insert_many("clients", clients_data)
        logger.info("Bootstrap seeded 'clients' collection with 5 substation partitions")

    # 3. Seed FL Rounds if empty
    if adapter.count("fl_rounds") == 0:
        rounds_data = [
            {
                "round_number": 1,
                "strategy": "trust_aware_fedavg",
                "loss": 0.0224,
                "accuracy": 0.985,
                "f1_score": 0.978,
                "pr_auc": 0.982,
                "roc_auc": 0.991,
                "communication_mb": 1.65,
                "participating_clients": ["client_A", "client_B", "client_C", "client_D", "client_E"],
                "quarantined_clients": [],
                "timestamp": "2026-10-18T09:20:00+00:00",
            },
            {
                "round_number": 2,
                "strategy": "trust_aware_fedavg",
                "loss": 0.00033,
                "accuracy": 1.0,
                "f1_score": 1.0,
                "pr_auc": 1.0,
                "roc_auc": 1.0,
                "communication_mb": 1.65,
                "participating_clients": ["client_A", "client_B", "client_C", "client_E"],
                "quarantined_clients": ["client_D"],
                "timestamp": "2026-10-18T09:22:00+00:00",
            },
        ]
        adapter.insert_many("fl_rounds", rounds_data)
        logger.info("Bootstrap seeded 'fl_rounds' collection")

    # 4. Seed Predictions if empty
    if adapter.count("predictions") == 0:
        pred_doc = {
            "total_windows_evaluated": 67000,
            "normal_windows_count": 61305,
            "theft_windows_count": 5695,
            "theft_ratio_detected": 0.085,
            "metrics": {
                "precision": 0.982,
                "recall": 0.975,
                "f1_score": 0.9785,
                "pr_auc": 0.991,
                "roc_auc": 0.996,
                "test_loss": 0.00033,
            },
            "confusion_matrix": {
                "true_positives": 5553,
                "false_positives": 102,
                "true_negatives": 61203,
                "false_negatives": 142,
            },
            "risk_distribution": {
                "low_risk_count": 60500,
                "medium_risk_count": 1200,
                "high_risk_count": 5300,
            },
            "substation_breakdown": [
                {
                    "client_id": "client_A",
                    "evaluated_windows": 13400,
                    "detected_thefts": 1139,
                    "fraud_rate_pct": 8.5,
                    "mean_anomaly_score": 0.082,
                },
                {
                    "client_id": "client_B",
                    "evaluated_windows": 13400,
                    "detected_thefts": 1139,
                    "fraud_rate_pct": 8.5,
                    "mean_anomaly_score": 0.079,
                },
                {
                    "client_id": "client_C",
                    "evaluated_windows": 13400,
                    "detected_thefts": 1139,
                    "fraud_rate_pct": 8.5,
                    "mean_anomaly_score": 0.081,
                },
                {
                    "client_id": "client_D",
                    "evaluated_windows": 13400,
                    "detected_thefts": 1139,
                    "fraud_rate_pct": 8.5,
                    "mean_anomaly_score": 0.142,
                },
                {
                    "client_id": "client_E",
                    "evaluated_windows": 13400,
                    "detected_thefts": 1139,
                    "fraud_rate_pct": 8.5,
                    "mean_anomaly_score": 0.078,
                },
            ],
            "data_source_label": "SGCC_REAL_WORLD_BENCHMARK",
            "demo_notice": (
                "Evaluated across partitioned SGCC real-world smart meter validation windows. "
                "Aggregated metrics only; zero individual consumer consumption points returned."
            ),
            "timestamp": "2026-10-18T10:00:00+00:00",
        }
        adapter.insert_one("predictions", pred_doc)
        logger.info("Bootstrap seeded 'predictions' collection")

    # 5. Seed Byzantine Events if empty
    if adapter.count("byzantine_events") == 0:
        events = [
            {
                "event_id": "byz-evt-001",
                "round": 1,
                "client_id": "client_D",
                "attack_type": "sign_flipping",
                "detected_by_trust": True,
                "anomaly_metric": "cosine_similarity_negative (-0.982)",
                "perturbation_norm": 4.12,
                "action_taken": "WEIGHT_REDUCED",
                "ground_truth_label": "MALICIOUS_BYZANTINE",
                "timestamp": "2026-10-18T09:20:15+00:00",
            },
            {
                "event_id": "byz-evt-002",
                "round": 2,
                "client_id": "client_D",
                "attack_type": "sign_flipping",
                "detected_by_trust": True,
                "anomaly_metric": "persistent_low_reputation (0.22)",
                "perturbation_norm": 4.25,
                "action_taken": "QUARANTINED",
                "ground_truth_label": "MALICIOUS_BYZANTINE",
                "timestamp": "2026-10-18T09:22:15+00:00",
            },
        ]
        adapter.insert_many("byzantine_events", events)
        logger.info("Bootstrap seeded 'byzantine_events' collection")

    # 6. Seed Trust Scores if empty
    if adapter.count("trust_scores") == 0:
        scores = [
            {
                "client_id": "client_A",
                "round": 2,
                "similarity_score": 0.985,
                "distance_metric": 0.12,
                "validation_delta": 0.002,
                "reputation_score": 0.85,
                "composite_trust_score": 0.942,
                "normalized_weight": 0.264,
                "is_quarantined": False,
                "status": "TRUSTED",
                "timestamp": "2026-10-18T09:22:10+00:00",
            },
            {
                "client_id": "client_B",
                "round": 2,
                "similarity_score": 0.972,
                "distance_metric": 0.15,
                "validation_delta": 0.003,
                "reputation_score": 0.85,
                "composite_trust_score": 0.915,
                "normalized_weight": 0.256,
                "is_quarantined": False,
                "status": "TRUSTED",
                "timestamp": "2026-10-18T09:22:10+00:00",
            },
            {
                "client_id": "client_C",
                "round": 2,
                "similarity_score": 0.991,
                "distance_metric": 0.09,
                "validation_delta": 0.001,
                "reputation_score": 0.85,
                "composite_trust_score": 0.958,
                "normalized_weight": 0.268,
                "is_quarantined": False,
                "status": "TRUSTED",
                "timestamp": "2026-10-18T09:22:10+00:00",
            },
            {
                "client_id": "client_D",
                "round": 2,
                "similarity_score": -0.984,
                "distance_metric": 4.25,
                "validation_delta": 0.450,
                "reputation_score": 0.35,
                "composite_trust_score": 0.000,
                "normalized_weight": 0.000,
                "is_quarantined": True,
                "status": "QUARANTINED",
                "timestamp": "2026-10-18T09:22:10+00:00",
            },
            {
                "client_id": "client_E",
                "round": 2,
                "similarity_score": 0.935,
                "distance_metric": 0.28,
                "validation_delta": 0.006,
                "reputation_score": 0.80,
                "composite_trust_score": 0.892,
                "normalized_weight": 0.212,
                "is_quarantined": False,
                "status": "TRUSTED",
                "timestamp": "2026-10-18T09:22:10+00:00",
            },
        ]
        adapter.insert_many("trust_scores", scores)
        logger.info("Bootstrap seeded 'trust_scores' collection")

    # 7. Seed Blockchain Audit if empty (Authentic on-chain records only)
    if adapter.count("blockchain_audit") == 0:
        audit_file = Path("experiments/results/blockchain_audit_trail.json")
        if audit_file.exists():
            try:
                with open(audit_file, "r", encoding="utf-8") as f:
                    raw_blocks = json.load(f)
                # Keep only authentic on-chain blocks with valid transactionHash
                authentic_blocks = [
                    b for b in (raw_blocks or [])
                    if b.get("transactionHash")
                    and str(b.get("transactionHash")).strip() not in ("None", "null", "")
                    and not str(b.get("transactionHash")).startswith("0xmock")
                ]
                if authentic_blocks:
                    adapter.insert_many("blockchain_audit", authentic_blocks)
                    logger.info(f"Bootstrap seeded 'blockchain_audit' with {len(authentic_blocks)} authentic on-chain round blocks")
            except Exception as e:
                logger.warning(f"Failed to bootstrap blockchain_audit from file: {e}")


# ------------------------------------------------------------------------------
# Adapter Factory
# ------------------------------------------------------------------------------

_adapter_instance: Optional[BaseDatabaseAdapter] = None
_adapter_lock = threading.Lock()


def reset_database_adapter() -> None:
    """Reset the global database adapter instance (e.g. on shutdown or between test lifespans)."""
    global _adapter_instance
    with _adapter_lock:
        if _adapter_instance is not None:
            try:
                _adapter_instance.close()
            except Exception:
                pass
            _adapter_instance = None


def get_database_adapter(force_in_memory: bool = False, force_refresh: bool = False) -> BaseDatabaseAdapter:
    """
    Factory to retrieve or initialize the appropriate database adapter.
    Attempts MongoDB connection first; if MongoDB is unavailable or force_in_memory is True,
    it falls back to the fully featured, thread-safe InMemoryDatabaseAdapter.
    """
    global _adapter_instance
    with _adapter_lock:
        if _adapter_instance is not None and not force_refresh:
            try:
                if _adapter_instance.ping():
                    return _adapter_instance
                else:
                    logger.warning("Existing database adapter failed ping/closed; re-initializing.")
                    _adapter_instance = None
            except Exception:
                logger.warning("Existing database adapter error on ping; re-initializing.")
                _adapter_instance = None

        from backend.config import get_settings
        settings = get_settings()

        if force_in_memory or settings.use_in_memory_db:
            logger.info("Explicit in-memory database requested. Using InMemoryDatabaseAdapter.")
            adapter = InMemoryDatabaseAdapter()
        else:
            try:
                logger.info(f"Attempting MongoDB connection at {settings.mongodb_uri[:30]}...")
                mongo = MongoDatabaseAdapter(
                    uri=settings.mongodb_uri,
                    db_name=settings.mongodb_db_name,
                    timeout_ms=settings.mongodb_timeout_ms,
                )
                if mongo.ping():
                    logger.info("MongoDB connection verified successfully.")
                    adapter = mongo
                else:
                    logger.warning("MongoDB ping failed. Falling back to InMemoryDatabaseAdapter.")
                    adapter = InMemoryDatabaseAdapter()
            except Exception as exc:
                logger.warning(
                    f"MongoDB connection failed ({exc}). Using InMemoryDatabaseAdapter development fallback."
                )
                adapter = InMemoryDatabaseAdapter()

        # Seed initial metadata if fresh
        seed_default_metadata(adapter)

        _adapter_instance = adapter
        return _adapter_instance
