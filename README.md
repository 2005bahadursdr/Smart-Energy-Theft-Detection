# Blockchain-Based Byzantine-Resilient Federated Learning for Smart Grid Energy Theft Detection

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-orange.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.33.0-yellow.svg)](https://flower.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.138-teal.svg)](https://fastapi.tiangolo.com/)
[![Web3.py](https://img.shields.io/badge/Web3.py-8.0-purple.svg)](https://web3py.readthedocs.io/)
[![Solidity](https://img.shields.io/badge/Solidity-^0.8.20-lightgrey.svg)](https://soliditylang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ⚡ Executive Summary & Research Motivation

Non-Technical Losses (NTL), predominantly driven by **electricity theft**, cost electric utilities worldwide over **$96 billion annually**, inducing severe distribution grid instability, voltage degradation, and inaccurate load dispatch. 

While machine learning classifiers can identify abnormal consumption patterns, traditional centralized data pooling violates privacy legislation (e.g., GDPR, CCPA) by exposing granular, high-frequency household consumption telemetry. Furthermore, while **Federated Learning (FL)** addresses data privacy by aggregating locally trained model parameters, standard FL aggregators (such as **FedAvg**) fail catastrophically under **Byzantine attacks** (e.g., model poisoning, sign flipping, label flipping) executed by compromised grid edge nodes or substations.

This research prototype introduces an end-to-end framework coupling:
1. **Time-Series Deep Learning Architectures** (LSTM and CNN-LSTM) tailored for raw and normalized smart meter consumption sequences.
2. **Multi-Faceted Trust & Byzantine Detection Mechanism** fusing cosine direction similarity, Euclidean gradient norm validation, and server-side loss benchmarks.
3. **Decaying Historical Reputation Engine** ensuring adaptive weighting and swift isolation of persistent adversaries.
4. **Blockchain Audit Trail (Ethereum/Solidity)** providing decentralized, tamper-proof auditability of node reputations and aggregation round proofs without ever leaking raw meter data.
5. **Interactive Full-Stack Platform** featuring a Meter Data Management System (MDMS) simulator, FastAPI asynchronous backend, and modern React/Vite/Tailwind CSS intelligence dashboard.

---

## 🏛️ System Architecture

```
                                  +---------------------------------------+
                                  |     Decentralized Ethereum Chain      |
                                  |      (ReputationRegistry.sol)         |
                                  +-------------------+-------------------+
                                                      ^
                                       Audit Hashes   |   Verify History
                                       & Reputations  |   & Quarantines
                                                      v
+-----------------------+         +-------------------+-------------------+         +-----------------------+
| Smart Meter Simulator |         |      Federated Learning Server        |         |  FastAPI Backend &    |
| (100-1000 Meters)     | ------> |  - Multi-Criteria Trust Scorer        | <-----> |  React Vite Dashboard |
| Normal + Theft Inject |         |  - Byzantine Anomaly Detector         |         |  (Live Telemetry &    |
+-----------------------+         |  - Robust Reputation Aggregator       |         |   Experiment Suite)   |
            |                     +-------------------+-------------------+         +-----------------------+
            | Raw Telemetry                           ^
            v                                         | Parameter Gradients / Updates
+-----------------------+                             v
|    MDMS Simulator     |              [ FL Clients / Substations ]
| - Regional Ingestion  |              +--------------------------+
| - Preprocessing       | -----------> | Client A (Industrial)    |
| - Feeder Partition    |              | Client B (Commercial)    |
+-----------------------+              | Client C (Residential 1) |
                                       | Client D (Residential 2) |
                                       | Client E (Rural Mixed)   |
                                       +--------------------------+
```

---

## 📂 Repository Structure

```
smart-grid-theft-detection/
├── README.md                          # Project documentation and research report
├── requirements.txt                    # Python runtime dependencies
├── .env.example                        # Environment variable configuration template
├── docker-compose.yml                  # Multi-container service definitions
├── run_project.py                      # Master CLI orchestrator
│
├── config/                             # Centralized YAML configuration files
│   ├── config.yaml                     # System paths, database, blockchain settings
│   ├── fl_config.yaml                  # FL hyperparameters and trust weights
│   ├── attack_config.yaml              # Byzantine threat parameters
│   └── model_config.yaml               # LSTM & CNN-LSTM specifications
│
├── data/
│   ├── raw/
│   │   └── smart_meter_data.csv        # Benchmark SGCC Smart Meter Dataset
│   ├── processed/                      # Regional non-IID partitioned client subsets
│   └── synthetic/                      # Procedurally generated smart meter telemetry
│
├── smart_meter_simulator/              # Stream simulator and theft event injector
├── mdms_simulator/                     # Meter Data Management System ingestion engine
├── fl_system/                          # Flower FL server, clients (A-E), and models
├── byzantine/                          # Attack simulations and anomaly detectors
├── trust/                              # Similarity, validation, and reputation engines
├── aggregation/                        # Robust Byzantine-resilient aggregation strategies
├── blockchain/                         # Web3 client, audit logger, and Solidity contracts
├── backend/                            # FastAPI REST API and MongoDB data models
├── frontend/                           # React 18, Vite, Tailwind CSS, Recharts dashboard
├── experiments/                        # Automated reproducible research experiment suite
├── notebooks/                          # Interactive analysis and evaluation notebooks
├── tests/                              # PyTest comprehensive unit and integration suite
└── docs/                               # System diagrams, research notes, and API specs
```

---

## 🚀 Quickstart & Setup

### Prerequisites
- Python 3.12+
- Node.js 20+ and npm
- MongoDB (Running locally on `mongodb://localhost:27017` or via Docker)

### 1. Environment Verification
Verify all machine learning, blockchain, and backend packages:
```bash
python run_project.py --mode check-env
```

### 2. Install Python Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Initialize Configuration
```bash
cp .env.example .env
```

### 4. Running the FastAPI Backend
Launch the FastAPI REST API server with automatic MongoDB / In-Memory database fallback:
```bash
# Using the Unified CLI runner:
python run_project.py --mode api

# Or directly using Uvicorn:
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Probe**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 5. Running the React + Vite + Tailwind CSS Dashboard
Launch the interactive intelligence dashboard:
```bash
# Using the Unified CLI runner:
python run_project.py --mode dashboard

# Or from the frontend directory:
cd frontend
npm install
npm run dev
```
The dashboard will be accessible at: **[http://127.0.0.1:5173](http://127.0.0.1:5173)**.

To compile the production bundle:
```bash
cd frontend
npm run build
```

### 6. Running the Test Suite
```bash
# Run backend API test suite:
python -m pytest tests/test_backend_api.py -v

# Run entire repository test suite (123 tests):
python -m pytest
```

---

## 📡 FastAPI REST Endpoints

| Method | Endpoint | Description | Privacy / Security Guardrail |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Health & Readiness Probe (DB, Blockchain, Models) | Standard system diagnostic |
| `GET` | `/api/meters/summary` | Aggregated smart meter statistics & event counts | **Strict Privacy**: Zero raw telemetry or PII |
| `GET` | `/api/clients` | Substation partition profiles (`client_A`–`client_E`) | Non-IID variance; raw data locked behind firewalls |
| `GET` | `/api/fl/rounds` | FL round telemetry (loss, accuracy, F1, PR-AUC) | Epistemic privacy notice included |
| `POST` | `/api/fl/start` | Launch FL simulation with safe bounded parameters | **Auth Warning**: Production requires JWT/mTLS |
| `GET` | `/api/predictions` | Aggregated energy theft prediction metrics | Aggregated metrics only; clear demo labels |
| `GET` | `/api/byzantine/events` | Detected & ground-truth Byzantine attack logs | Outliers are not proof of malicious intent |
| `GET` | `/api/trust/scores` | Multi-criteria component trust scores & weights | Dynamic weighting dampens Byzantine nodes |
| `GET` | `/api/blockchain/audit` | On-chain ReputationRegistry audit records | Cryptographic `[TAMPER-EVIDENT OK]` verification |
| `POST` | `/api/simulation/run` | Execute smart meter synthetic grid simulation | **Auth Warning**: Safe bounded parameter validation |

---

## 🔒 Security & Privacy Guarantees
- **Strict Data Privacy**: Raw smart meter time-series readings and consumer PII are NEVER exposed in global responses. Only privacy-preserving statistical moments and fraud ratios are served.
- **Production Authentication Warning**: All control endpoints (`POST /api/fl/start`, `POST /api/simulation/run`) require mutual TLS, OAuth2/JWT tokens, and RBAC in production deployments.
- **Dual Database Architecture**: Automatically connects to MongoDB (Atlas or local); seamlessly falls back to a thread-safe, documented in-memory adapter for development without MongoDB.
- **No Raw Telemetry On-Chain**: The Ethereum blockchain records exclusively cryptographic hashes of global models, round IDs, and aggregated client reputation scores.
- **Byzantine Resilience**: Up to 40% malicious or compromised nodes (sign flipping, model poisoning, label flipping) are identified and dynamically quarantined via our reputation-weighted defense.
