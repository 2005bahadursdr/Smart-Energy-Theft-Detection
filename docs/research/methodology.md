# Methodology, Threat Model, Limitations & Windows Demo Guide

## 1. System Overview & Methodology

The **Blockchain-Based Byzantine-Resilient Federated Learning for Smart Grid Energy Theft Detection** system provides an end-to-end framework integrating:
1. **Synthetic Smart Meter Simulator & MDMS Preprocessing**: Generates realistic diurnal consumption, seasonal load patterns, solar PV, legitimate zero-consumption (vacancies, outages, disconnections), and diverse hardware/cyber theft events (partial bypass, full bypass, reverse current, magnetic tamper, cyber scaling, flatline, peak clipping).
2. **Decentralized Substation Feeder Partitioning**: Regional non-IID partitions (`client_A` through `client_E`) exhibiting extreme consumption heterogeneity (industrial 38.32 kWh/day down to rural agricultural 0.36 kWh/day).
3. **Local Deep Learning Sequence Models**: PyTorch CNN-LSTM (`Conv1D -> MaxPool1d -> LSTM(64) -> Dropout -> Linear -> Sigmoid`) and LSTM classifiers trained strictly on local data.
4. **Byzantine Threat Injection**: Sign-flipping gradient inversion, extreme high-norm model poisoning, Gaussian noise, and adversarial label flipping.
5. **Multi-Criteria Trust Engine**:
   - **Directional Agreement ($S_k$)**: Cosine similarity between client gradient updates and reference median gradient.
   - **Validation Loss Delta ($V_k$)**: Server-side validation loss impact when incorporating client update.
   - **Historical Reputation ($R_k$)**: Exponential Moving Average (EMA) with forgiveness decay:
     $$R_{k}^{(t)} = \lambda \cdot R_{k}^{(t-1)} + (1 - \lambda) \cdot T_{k}^{(t)}$$
   - **Composite Trust Score ($T_k$)**:
     $$T_k = \alpha \cdot S_k + \beta \cdot V_k + \gamma \cdot R_k$$
   - **Dynamic Weighting & Quarantine**:
     $$w_k = \begin{cases} 0.0 & \text{if } T_k < \tau_{quarantine} \text{ or } S_k < 0 \\ \frac{T_k}{\sum_{j \in \mathcal{A}} T_j} & \text{otherwise} \end{cases}$$
6. **Immutable On-Chain Audit Logging**: Solidity `ReputationRegistry` smart contract deployed to an Ethereum-compatible chain via Web3.py v8.0, storing canonical SHA-256 record hashes of coordinator decisions without ever leaking raw telemetry or model weights.
7. **FastAPI Backend & React Dashboard**: Dual-database REST API (MongoDB Atlas + In-Memory fallback) coupled with an interactive React + Vite + Tailwind CSS dashboard.

---

## 2. Epistemic Principles & Threat Model

### 2.1 Anomaly != Malice
> [!IMPORTANT]
> **Epistemic Invariant**: An unusual, high-norm, or directionally divergent gradient update is **NOT definitive proof of malicious Byzantine intent**. In smart distribution grids, regional non-IID energy consumption (e.g. continuous industrial loads in `client_A` or seasonal agricultural pumps in `client_E`) naturally produces legitimate statistical outliers. The trust scoring mechanism explicitly balances outlier dampening with domain-specific non-IID variance tolerance.

### 2.2 Proof of Claim vs. Proof of Correctness
> [!NOTE]
> On-chain cryptographic hashes prove **non-repudiation and temporal ordering** of coordinator claims. Blockchain records evidence; it does not itself detect or prevent poisoning.

### 2.3 Strict Data Privacy Invariant
> [!CAUTION]
> Raw smart meter time-series consumption traces and consumer PII are strictly forbidden from being transmitted to the central FL coordinator, written to the blockchain, or exposed in global API responses. Only model parameter gradients and statistical aggregates are communicated.

---

## 3. Empirical Benchmark Results & Separation of Metrics

Controlled comparative experiments were executed across 4 experimental arms under identical initial global model parameters and non-IID data splits (Seed: 42, CNN-LSTM, 2 rounds, 1 local epoch, learning rate 0.001):

### 3.1 Model Performance Metrics

| Experimental Arm | Strategy | Attack Evaluated | Final Test Loss | Final F1 | PR-AUC | ROC-AUC | Convergence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. FedAvg Vulnerable** | FedAvg (1/N uniform) | Sign-Flipping (20% attacker) | `0.05672` | `98.6%` | `100.0%` | `100.0%` | Degraded loss |
| **2. Coordinate Median** | Coordinate-Wise Median | Sign-Flipping (20% attacker) | `0.00041` | `100.0%` | `100.0%` | `100.0%` | Fast |
| **3. Trust-Aware Defense** | Multi-Criteria FedAvg | Sign-Flipping (20% attacker) | `0.00033` | `100.0%` | `100.0%` | `100.0%` | Optimal |
| **4. Clean Target** | Trust-Aware FedAvg | None (Clean Baseline) | `0.00035` | `100.0%` | `100.0%` | `100.0%` | Optimal |

### 3.2 Byzantine Detection & Resilience Metrics

| Experimental Arm | Malicious Weight Ingested | Detection Precision | Detection Recall | Detection F1 | False Positive Rate (FPR) | Arm Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. FedAvg Vulnerable** | `20.0%` (Full poisoning) | N/A (No detector) | N/A | N/A | N/A | `~4.5s` |
| **2. Coordinate Median** | `20.0%` (Uniform median) | N/A (Heuristic) | N/A | N/A | N/A | `~4.8s` |
| **3. Trust-Aware Defense** | **`0.0%` (Quarantined)** | **`100.0%`** | **`100.0%`** | **`100.0%`** | **`0.0%`** | `~5.2s` |
| **4. Clean Target** | `0.0%` (No attackers) | N/A | N/A | N/A | `0.0%` (0 false alarms) | `~4.4s` |

---

## 4. REST API Endpoint Reference

| Method | Endpoint | Description | Guardrail |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Subsystem Readiness Probe (DB, Blockchain, DL) | Diagnostic probe |
| `GET` | `/api/meters/summary` | Aggregated consumption moments (mean, std, quartiles) | **Strict Privacy**: Zero raw telemetry |
| `GET` | `/api/clients` | Substation profiles (`client_A`..`client_E`) | Non-IID variance; firewalls ACTIVE |
| `GET` | `/api/fl/rounds` | FL round telemetry (loss, accuracy, F1, PR-AUC) | Epistemic privacy notice |
| `POST` | `/api/fl/start` | Launch FL simulation with safe parameters | Safe bounds (422) + Auth warning |
| `GET` | `/api/predictions` | Aggregated fraud detection benchmarks & risk tiers | Consumer theft != FL attacker |
| `GET` | `/api/byzantine/events` | Perturbation events & mitigation ledger | Ground-truth isolation enforced |
| `GET` | `/api/trust/scores` | Multi-criteria component trust scores & weights | Reason codes; dynamic dampening |
| `GET` | `/api/blockchain/audit` | On-chain ReputationRegistry audit records | Cryptographic `[TAMPER-EVIDENT OK]` |
| `POST` | `/api/simulation/run` | Execute smart meter synthetic grid synthesis | Safe bounds (422) + Auth warning |

---

## 5. Windows Demonstration Commands (PowerShell)

All commands are runnable from the repository root in PowerShell on Windows:

### 1. Verify Environment & Services
```powershell
python run_project.py --mode check-env
```

### 2. Execute Full Integrated Pipeline
```powershell
# Runs complete pipeline: data verification -> comparative benchmark -> blockchain audit -> DB bootstrap
python run_project.py --mode all
```

### 3. Launch Backend REST API
```powershell
# Launch FastAPI backend on http://127.0.0.1:8000 (MongoDB Atlas or In-Memory fallback)
python run_project.py --mode api

# Or force in-memory development adapter:
$env:USE_IN_MEMORY_DB="true"
python run_project.py --mode api
```

### 4. Launch React + Vite Dashboard
```powershell
# Launch interactive frontend dashboard on http://127.0.0.1:5173
python run_project.py --mode dashboard
```

### 5. Run Individual Subsystem Benchmarks
```powershell
# Multi-criteria trust benchmark under sign-flipping
python run_project.py --mode trust-benchmark --rounds 3 --attack-type sign_flipping

# Query on-chain blockchain audit records
python run_project.py --mode blockchain-audit
```

### 6. Run Complete Test Suite
```powershell
# Run all 124+ automated tests (backend, blockchain, trust, FL, models, MDMS, simulator, E2E)
python -m pytest -v
```

### 7. Compile Frontend Production Bundle
```powershell
cd frontend
npm run build
```
