# FastAPI Backend Documentation

## 1. Overview & Architecture

The backend provides a production-grade REST API for the **Blockchain-Based Byzantine-Resilient Federated Learning for Smart Grid Energy Theft Detection** system. Built with **FastAPI** and **Pydantic v2**, it coordinates federated telemetry, multi-criteria Byzantine trust evaluation, smart meter statistics, and Ethereum-compatible on-chain audit tracking.

```
                          ┌────────────────────────┐
                          │   Frontend / Client    │
                          │ (Vite / React Dashboard)│
                          └───────────┬────────────┘
                                      │ HTTP / CORS
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │               FastAPI Application                │
             │               (backend/main.py)                  │
             │   - Security & Production Warning Middleware     │
             │   - Configured Local-Origin CORS                 │
             │   - Global Exception Handlers & Validation       │
             └───────────┬──────────────────────────┬───────────┘
                         │                          │
        ┌────────────────▼─────────┐      ┌─────────▼──────────────┐
        │       Service Layer      │      │    Database Adapter    │
        │  - meters_service.py     │      │   - MongoDatabaseAdapter│
        │  - client_service.py     │      │   - InMemoryDatabaseAdapter
        │  - fl_service.py         │      └─────────┬──────────────┘
        │  - prediction_service.py │                │
        │  - byzantine_service.py  │      ┌─────────▼──────────────┐
        │  - trust_service.py      │      │ MongoDB Atlas / Local  │
        │  - blockchain_service.py │      │     (or In-Memory)     │
        │  - simulation_service.py │      └────────────────────────┘
        └───────────┬──────────────┘
                    │
        ┌───────────▼──────────────┐
        │  Blockchain & Trust Sub  │
        │  - ReputationRegistry.sol│
        │  - canonical_hasher.py   │
        └──────────────────────────┘
```

---

## 2. Core Invariants & Security Guardrails

### 2.1 Confidentiality & Privacy Enforcement
> [!IMPORTANT]
> **Strict Privacy Invariant**: Individual smart meter consumption time-series readings (`consumption_kwh`, raw arrays) and consumer PII (names, meter IDs, account addresses) are **NEVER exposed in global API responses**. All meter and prediction endpoints return exclusively privacy-preserving statistical moments (mean, std, quartiles, median), substation totals, and fraud risk distributions. Every response includes the `X-Privacy-Enforcement: Strict - Raw Meter Readings Redacted` header.

### 2.2 Production Security Warning & Authentication Notice
> [!WARNING]
> **Production Controls Require Authentication**:
> Control actions (`POST /api/fl/start` and `POST /api/simulation/run`) directly manipulate federated training workflows and grid simulation parameters.
> In a live utility or smart distribution grid, these endpoints **MUST** be protected behind mutual TLS (mTLS), OAuth2/JWT token authentication, and strict Role-Based Access Control (RBAC).
> In this local development and research environment, controls operate unauthenticated for rapid prototyping. Every response injects the `X-Production-Warning` HTTP header and control payloads include an explicit `production_warning` property.

### 2.3 Dual Database Architecture & In-Memory Fallback
The backend implements an abstract adapter pattern (`BaseDatabaseAdapter`):
- **`MongoDatabaseAdapter`**: Connects via PyMongo with configurable timeouts (`MONGODB_TIMEOUT_MS`). Tested and operational with MongoDB Atlas and local daemons.
- **`InMemoryDatabaseAdapter`**: A fully documented, thread-safe (re-entrant lock `threading.RLock`) in-memory database adapter for development without an external MongoDB server. Supports MongoDB-style query filters (`$in`, `$gte`, `$lte`, `$gt`, `$lt`), sorting, and projection.
- **Automatic Fallback Factory**: `get_database_adapter()` automatically probes MongoDB. If connection fails or `USE_IN_MEMORY_DB=true`, it seamlessly falls back to the in-memory adapter without crashing the server.
- **Bootstrap Seeding**: Automatically seeds benchmark metadata from `data/synthetic/` and `experiments/results/` on initial boot.

---

## 3. Environment Configuration

Configuration is loaded from `.env` using typed Pydantic models in `backend/config.py`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PROJECT_NAME` | `"Blockchain-Based Byzantine-Resilient..."` | Application title reported in health check |
| `ENVIRONMENT` | `development` | Deployment environment (`development` / `production`) |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection URI (e.g. MongoDB Atlas) |
| `MONGODB_DB_NAME` | `smart_grid_theft_fl` | Primary MongoDB database name |
| `MONGODB_TIMEOUT_MS` | `2500` | Connection timeout before falling back to in-memory |
| `USE_IN_MEMORY_DB` | `false` | Force in-memory adapter (`true` / `false`) |
| `BACKEND_HOST` | `127.0.0.1` | Host address for FastAPI backend |
| `BACKEND_PORT` | `8000` | Port for FastAPI backend |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-delimited list of allowed browser origins |
| `BLOCKCHAIN_PROVIDER_TYPE` | `eth_tester` | Blockchain provider (`eth_tester`, `rpc`, `mock`) |
| `REPUTATION_REGISTRY_CONTRACT_ADDRESS` | `0x5FbDB...` | Active ReputationRegistry contract address |

---

## 4. REST API Endpoint Reference

### 1. System Health Probe
- **`GET /health`** (and **`GET /api/health`**)
- **Summary**: Readiness probe reporting health of Database, Blockchain node, and DL models.
- **Response**: `HealthResponse`
```json
{
  "status": "healthy",
  "project_name": "Blockchain-Based Byzantine-Resilient Federated Learning for Smart Grid Energy Theft Detection",
  "environment": "development",
  "database": {
    "provider": "mongodb",
    "connected": true,
    "database_name": "smart_grid_theft_fl",
    "message": "Database connected and responsive."
  },
  "blockchain": {
    "provider": "eth_tester",
    "connected": true,
    "block_number": 2,
    "contract_address": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
    "message": "Blockchain provider operational (eth_tester)."
  },
  "model_readiness": {
    "ready": true,
    "available_models": ["CNN_LSTM", "LSTM"],
    "checkpoint_dir": "fl_system\\checkpoints"
  },
  "version": "1.0.0",
  "timestamp": "2026-10-18T10:15:00Z"
}
```

### 2. Aggregated Smart Meter Statistics
- **`GET /api/meters/summary`**
- **Summary**: Statistical distribution of smart meter telemetry across the grid.
- **Privacy Enforcement**: Zero raw readings returned.
- **Response**: `MetersSummaryResponse` (mean, std, min, max, quartiles, event breakdowns, consumer profile counts, `data_source_label: "SIMULATED_DEMO_DATA"`).

### 3. Substation Partition Clients
- **`GET /api/clients`**
- **Summary**: Substation metadata for `client_A` through `client_E`.
- **Response**: `ClientListResponse`
  - Demonstrates regional load heterogeneity: `client_A` (38.32 kWh/day industrial) vs `client_E` (0.36 kWh/day rural).
  - Confirms `privacy_firewall: "ACTIVE"` and `central_fl_access: "BLOCKED"`.

### 4. Federated Learning Round Telemetry
- **`GET /api/fl/rounds`**
- **Summary**: Historical FL round progress, test loss, F1-scores, PR-AUC, communication MB, and participant quarantine lists.
- **Response**: `FLRoundsResponse` with epistemic privacy notice.

### 5. Start FL Simulation (Safe Bounded Parameters)
- **`POST /api/fl/start`**
- **Summary**: Initiates federated learning simulation.
- **Validation Constraints**:
  - `rounds`: `1 <= rounds <= 20` (default 5)
  - `epochs`: `1 <= epochs <= 5` (default 1)
  - `model`: `"cnn_lstm"` | `"lstm"`
  - `strategy`: `"trust_aware_fedavg"` | `"fedavg"` | `"coordinate_median"`
  - `attack_type`: `"none"` | `"sign_flipping"` | `"model_poisoning"` | `"random_noise"` | `"label_flipping"`
  - `malicious_fraction`: `0.0 <= val <= 0.5`
- **Response**: `FLStartResponse` containing `job_id`, status, parameters, and explicit production authentication warning.

### 6. Energy Theft Predictions
- **`GET /api/predictions`**
- **Summary**: Aggregated fraud detection performance on SGCC smart meter test partitions.
- **Response**: `PredictionsResponse` (confusion matrix, PR-AUC, ROC-AUC, risk distributions, substation breakdown, `data_source_label: "SGCC_REAL_WORLD_BENCHMARK"`).

### 7. Byzantine Anomaly Events
- **`GET /api/byzantine/events`**
- **Summary**: Detected and ground-truth Byzantine perturbation events across rounds.
- **Response**: `ByzantineEventsResponse` with mitigation summaries and epistemic research disclaimer.

### 8. Multi-Criteria Trust Scores
- **`GET /api/trust/scores`**
- **Summary**: Client-by-client component trust breakdowns (similarity, distance, validation delta, reputation score, composite trust score, normalized aggregation weights, quarantine flag).
- **Response**: `TrustScoresResponse` including active weighting parameters.

### 9. On-Chain Blockchain Audit Ledger
- **`GET /api/blockchain/audit`**
- **Summary**: Immutable audit records from the Ethereum `ReputationRegistry` contract.
- **Response**: `BlockchainAuditResponse` with canonical record hashes, block numbers, transaction hashes, and cryptographic status (`"[TAMPER-EVIDENT OK]"`).

### 10. Trigger Grid Simulation
- **`POST /api/simulation/run`**
- **Summary**: Generates synthetic smart meter grid partitions.
- **Validation Constraints**:
  - `num_meters`: `10 <= num_meters <= 2000`
  - `days`: `1 <= days <= 90`
  - `sampling_freq_minutes`: `15` | `30` | `60`
  - `theft_ratio`: `0.0 <= val <= 0.5`
- **Response**: `SimulationRunResponse` with generated statistical summary and production authentication warning.

---

## 5. How to Run & Test

### Run via Unified CLI
```powershell
# Launch FastAPI backend on host:port configured in .env (127.0.0.1:8000)
python run_project.py --mode api
```

### Run Directly via Uvicorn
```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### Run In-Memory Development Mode (No MongoDB Required)
```powershell
# Set environment variable to force documented in-memory adapter
$env:USE_IN_MEMORY_DB="true"
python run_project.py --mode api
```

### Execute Test Suite
```powershell
# Run backend unit and integration tests
python -m pytest tests/test_backend_api.py -v

# Run entire 123-test project test suite
python -m pytest
```
