"""
MDMS Local Storage Engine.
Provides high-performance, thread-safe SQLite persistence for smart meter telemetry,
with unique indexing on (meter_id, timestamp) for deduplication, fast historical range queries,
and operational meter status lookups.
"""

from contextlib import contextmanager
from datetime import datetime
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from mdms_simulator.config import MDMSConfig
from mdms_simulator.models import MeterReadingPayload, MeterStatusResponse

logger = logging.getLogger("mdms_storage")


class MDMSStorage:
    """
    Local SQLite persistence engine for the MDMS Simulator.
    """

    def __init__(self, db_path: Optional[Path] = None, config: Optional[MDMSConfig] = None):
        self.config = config or MDMSConfig()
        self.db_path = db_path or self.config.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_schema()

    @contextmanager
    def _get_connection(self):
        """Create a sqlite3 connection with automatic closure upon context exit."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database tables and indexes."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meter_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    meter_id TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    consumption_kwh REAL,
                    meter_status TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    hardware_tamper_flag INTEGER NOT NULL DEFAULT 0,
                    is_late INTEGER NOT NULL DEFAULT 0,
                    is_missing INTEGER NOT NULL DEFAULT 0,
                    simulated_theft_label INTEGER DEFAULT 0,
                    simulated_theft_type TEXT DEFAULT 'NONE',
                    simulated_actual_kwh REAL,
                    simulated_unreported_kwh REAL DEFAULT 0.0,
                    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Unique constraint ensures zero duplicate records for the same meter at the same timestamp
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_meter_timestamp 
                ON meter_readings(meter_id, timestamp);
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_meter_id 
                ON meter_readings(meter_id);
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_client_id 
                ON meter_readings(client_id);
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON meter_readings(timestamp);
            """)
            conn.commit()

    def insert_reading(
        self,
        payload: MeterReadingPayload,
        is_late: bool = False,
        is_missing: bool = False
    ) -> bool:
        """
        Insert single reading into SQLite.
        Returns True if inserted, False if rejected due to duplicate key.
        """
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO meter_readings (
                        timestamp, meter_id, client_id, consumption_kwh,
                        meter_status, event_type, hardware_tamper_flag,
                        is_late, is_missing, simulated_theft_label,
                        simulated_theft_type, simulated_actual_kwh, simulated_unreported_kwh
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    payload.timestamp,
                    payload.meter_id,
                    payload.client_id,
                    payload.consumption_kwh,
                    payload.meter_status,
                    payload.event_type,
                    payload.hardware_tamper_flag,
                    1 if is_late else 0,
                    1 if is_missing else 0,
                    payload.simulated_theft_label or 0,
                    payload.simulated_theft_type or "NONE",
                    payload.simulated_actual_kwh,
                    payload.simulated_unreported_kwh or 0.0,
                ))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Duplicate (meter_id, timestamp) key caught by unique index
                return False

    def insert_batch(
        self,
        batch: List[Tuple[MeterReadingPayload, bool, bool]]
    ) -> Tuple[int, int]:
        """
        Batch insert readings in a single atomic transaction.

        Returns
        -------
        Tuple[int, int]
            (inserted_count, duplicate_count)
        """
        if not batch:
            return 0, 0

        inserted = 0
        duplicates = 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for payload, is_late, is_missing in batch:
                try:
                    cursor.execute("""
                        INSERT INTO meter_readings (
                            timestamp, meter_id, client_id, consumption_kwh,
                            meter_status, event_type, hardware_tamper_flag,
                            is_late, is_missing, simulated_theft_label,
                            simulated_theft_type, simulated_actual_kwh, simulated_unreported_kwh
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (
                        payload.timestamp,
                        payload.meter_id,
                        payload.client_id,
                        payload.consumption_kwh,
                        payload.meter_status,
                        payload.event_type,
                        payload.hardware_tamper_flag,
                        1 if is_late else 0,
                        1 if is_missing else 0,
                        payload.simulated_theft_label or 0,
                        payload.simulated_theft_type or "NONE",
                        payload.simulated_actual_kwh,
                        payload.simulated_unreported_kwh or 0.0,
                    ))
                    inserted += 1
                except sqlite3.IntegrityError:
                    duplicates += 1
            conn.commit()

        return inserted, duplicates

    def get_meter_status(self, meter_id: str) -> Optional[MeterStatusResponse]:
        """
        Query current status, latest reading, and data quality health for a meter.
        """
        with self._get_connection() as conn:
            # Latest reading
            latest = conn.execute("""
                SELECT * FROM meter_readings 
                WHERE meter_id = ? 
                ORDER BY timestamp DESC LIMIT 1;
            """, (meter_id,)).fetchone()

            if not latest:
                return None

            # Aggregate telemetry health
            agg = conn.execute("""
                SELECT 
                    COUNT(*) as total_count,
                    SUM(is_missing) as missing_count,
                    MAX(hardware_tamper_flag) as tamper_flag_ever
                FROM meter_readings 
                WHERE meter_id = ?;
            """, (meter_id,)).fetchone()

            total = agg["total_count"] or 0
            missing = agg["missing_count"] or 0
            quality_score = round(max(0.0, ((total - missing) / max(1, total))) * 100, 1)

            return MeterStatusResponse(
                meter_id=meter_id,
                client_id=latest["client_id"],
                current_status=latest["meter_status"],
                latest_timestamp=latest["timestamp"],
                latest_consumption_kwh=latest["consumption_kwh"],
                latest_event_type=latest["event_type"],
                hardware_tamper_active=(latest["hardware_tamper_flag"] == 1),
                total_readings_recorded=total,
                missing_readings_count=missing,
                data_quality_score_pct=quality_score
            )

    def list_meters(
        self,
        client_id: Optional[str] = None,
        meter_status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List distinct meters with current status and client assignment."""
        query = """
            SELECT meter_id, client_id, meter_status, MAX(timestamp) as latest_timestamp,
                   COUNT(*) as reading_count
            FROM meter_readings
        """
        params: List[Any] = []
        conditions: List[str] = []

        if client_id:
            conditions.append("client_id = ?")
            params.append(client_id)
        if meter_status:
            conditions.append("meter_status = ?")
            params.append(meter_status)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY meter_id ORDER BY meter_id ASC LIMIT ?;"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_meter_history(self, meter_id: str, limit: int = 168) -> List[Dict[str, Any]]:
        """Retrieve recent time-series history for a meter (default 168 hours = 1 week)."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT timestamp, consumption_kwh, meter_status, event_type, hardware_tamper_flag,
                       is_late, is_missing
                FROM meter_readings
                WHERE meter_id = ?
                ORDER BY timestamp DESC LIMIT ?;
            """, (meter_id, limit)).fetchall()
            # Return chronologically ascending
            return [dict(r) for r in reversed(rows)]

    def get_total_readings_count(self) -> int:
        """Count total stored readings."""
        with self._get_connection() as conn:
            res = conn.execute("SELECT COUNT(*) FROM meter_readings;").fetchone()
            return int(res[0]) if res else 0

    def get_client_counts(self) -> Dict[str, int]:
        """Get readings count grouped by client."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT client_id, COUNT(*) as count 
                FROM meter_readings 
                GROUP BY client_id;
            """).fetchall()
            return {r["client_id"]: int(r["count"]) for r in rows}
