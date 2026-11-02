"""
Backend Configuration Module.
Loads environment variables and exposes typed settings via Pydantic.
"""

import os
from pathlib import Path
from typing import List
import dotenv
from pydantic import BaseModel, Field

# Load project .env if present
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    dotenv.load_dotenv(dotenv_path=env_path, override=False)


class BackendSettings(BaseModel):
    """Application configuration settings for FastAPI backend."""

    project_name: str = Field(
        default_factory=lambda: os.getenv(
            "PROJECT_NAME",
            "Blockchain-Based Byzantine-Resilient Federated Learning for Smart Grid Energy Theft Detection",
        )
    )
    environment: str = Field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    # MongoDB Configuration
    mongodb_uri: str = Field(
        default_factory=lambda: os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    )
    mongodb_db_name: str = Field(
        default_factory=lambda: os.getenv("MONGODB_DB_NAME", "smart_grid_theft_fl")
    )
    mongodb_timeout_ms: int = Field(
        default_factory=lambda: int(os.getenv("MONGODB_TIMEOUT_MS", "2500"))
    )
    use_in_memory_db: bool = Field(
        default_factory=lambda: os.getenv("USE_IN_MEMORY_DB", "false").lower() in ("true", "1", "yes")
    )

    # Server & CORS
    backend_host: str = Field(
        default_factory=lambda: os.getenv("BACKEND_HOST", "127.0.0.1")
    )
    backend_port: int = Field(
        default_factory=lambda: int(os.getenv("BACKEND_PORT", "8000"))
    )
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        ]
    )

    # Blockchain Configuration
    blockchain_provider_type: str = Field(
        default_factory=lambda: os.getenv("BLOCKCHAIN_PROVIDER_TYPE", "eth_tester")
    )
    blockchain_rpc_url: str = Field(
        default_factory=lambda: os.getenv("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545")
    )
    reputation_registry_contract_address: str = Field(
        default_factory=lambda: os.getenv(
            "REPUTATION_REGISTRY_CONTRACT_ADDRESS",
            "0x5FbDB2315678afecb367f032d93F642f64180aa3",
        )
    )

    # FL Configuration
    fl_server_host: str = Field(
        default_factory=lambda: os.getenv("FL_SERVER_HOST", "127.0.0.1")
    )
    fl_server_port: int = Field(
        default_factory=lambda: int(os.getenv("FL_SERVER_PORT", "8080"))
    )


_settings_instance = None


def get_settings() -> BackendSettings:
    """Retrieve cached backend settings singleton."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = BackendSettings()
    return _settings_instance
