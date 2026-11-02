"""
Backend Database Package.
Exposes database adapters and factory function.
"""

from backend.database.adapter import (
    BaseDatabaseAdapter,
    MongoDatabaseAdapter,
    InMemoryDatabaseAdapter,
    get_database_adapter,
    seed_default_metadata,
)

__all__ = [
    "BaseDatabaseAdapter",
    "MongoDatabaseAdapter",
    "InMemoryDatabaseAdapter",
    "get_database_adapter",
    "seed_default_metadata",
]
