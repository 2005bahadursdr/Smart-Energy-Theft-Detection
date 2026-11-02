# Research Dashboard Architecture & User Guide

## 1. Overview & Aesthetics

The **Smart Grid Byzantine-Resilient FL & Blockchain Audit Dashboard** is a research-grade intelligence interface developed with **React 18**, **Vite**, and **Tailwind CSS**. Designed for grid cybersecurity analysts and federated learning researchers, it combines a high-density dark UI theme (`#0B0F19`, slate surfaces, glassmorphism) with high-contrast indicator accents (cyan, emerald, amber, rose, purple).

```
                               ┌────────────────────────────────────────────────────────┐
                               │       Top Navigation Header (Navbar.jsx)               │
                               │  - Subsystem Health (MongoDB Atlas / In-Memory Dev)    │
                               │  - Blockchain Block Height & Connection               │
                               │  - Local Data Isolation Firewall Indicator             │
                               │  - Mandatory Production Authentication Notice          │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
          ┌───────────────────────────────────────────────┴───────────────────────────────────────────────┐
          │                                                                                               │
┌─────────▼──────────────────────┐                                              ┌─────────────────────────▼──────────────────────────┐
│     Sidebar Navigation         │                                              │               Active Page View                     │
│  - Executive Overview          │ ───► /                                  ───► │  - StatCards (KPIs & Trends)                       │
│  - Smart Meters & MDMS         │ ───► /meters                            ───► │  - Recharts Visualizations (Bar/Line/Pie/Radar)    │
│  - FL Clients & Rounds         │ ───► /fl                                ───► │  - Substation Partitions Table                     │
│  - Theft-Risk Predictions      │ ───► /predictions                       ───► │  - Interactive Simulation / FL Controllers         │
│  - Byzantine Threat Monitor    │ ───► /byzantine                         ───► │  - Loading, Empty, and Error States                │
│  - Trust & Reputation Engine   │ ───► /trust                             ───► │  - Epistemic Guardrail Notices                     │
│  - FedAvg vs Trust Defense     │ ───► /comparison                        ───► │  - Clear [SIMULATED] & [GROUND TRUTH] Badges       │
│  - Blockchain Audit Trail      │ ───► /blockchain                        ───► │  - Tamper-Evident [TAMPER-EVIDENT OK] Badges       │
│  - Experiment Benchmarks       │ ───► /experiments                       ───► └────────────────────────────────────────────────────┘
└────────────────────────────────┘
```

---

## 2. Page Directory & Features

### 1. Executive Overview (`OverviewPage.jsx`)
- **System-Wide Telemetry**: Aggregates live status from meters, federated learning rounds, Byzantine detectors, and Ethereum smart contracts.
- **Visual Architecture Flow**: Step-by-step pipeline illustrating raw meter isolation, non-IID feeder slicing, local CNN-LSTM updates, multi-criteria trust weighting, and on-chain commitment.
- **Quick Feeds**: Top substation reputation rankings and recent on-chain audit proof blocks.

### 2. Smart Meters & MDMS (`MetersPage.jsx`)
- **Strict Confidentiality**: Zero raw time-series consumption traces exposed. Displays mean (21.98 kWh), std, median, quartiles, and total reported energy.
- **Visualizations**: Operational status breakdown (Pie chart) and consumer profile distribution (Bar chart).
- **Interactive Grid Simulator Controller**: Safe form for re-generating synthetic smart grid telemetry (`num_meters`: 10–2000, `days`: 1–90, `sampling_freq`: 15/30/60m, `theft_ratio`: 0–50%).

### 3. FL Clients & Rounds (`FLClientsRoundsPage.jsx`)
- **Substation Partitions Table**: Detailed profile for substations `client_A` through `client_E`, displaying regional load intensity (industrial 38.32 kWh down to rural 0.36 kWh), trust scores, reputations, and active privacy firewalls.
- **Convergence Curves**: Interactive Recharts line chart tracking global test loss, F1-scores, accuracy, and PR-AUC over training rounds.
- **Interactive FL Experiment Dispatcher**: Safe form to trigger federated learning simulations (`rounds`: 1–20, `epochs`: 1–5, `model`: CNN-LSTM/LSTM, `strategy`: Trust-Aware/FedAvg/Median, `attack`: Sign-Flipping/Poisoning/Noise/Label-Flipping).

### 4. Theft-Risk Predictions (`PredictionsPage.jsx`)
- **Crucial Distinction Highlighted**: Dedicated architectural comparison panel clearly separating **suspicious consumer patterns** (physical/cyber meter tampering, bypass, scaling) from **suspicious FL clients** (adversarial gradient poisoning, sign-flipping).
- **Confusion Matrix**: True Positives (5,553), False Negatives (142), False Positives (102), True Negatives (61,203) with precision (98.2%) and recall (97.5%).
- **Risk Distribution**: Triage into Low Risk, Medium Warning, and Confirmed Theft tiers.

### 5. Byzantine Threat Monitor (`ByzantineMonitorPage.jsx`)
- **Perturbation Event Log**: Tracks attack modality, anomaly metrics, perturbation norm, and mitigation action (quarantine, weight reduction).
- **Epistemic Principle Display**: "An unusual, high-norm, or divergent gradient update is NOT definitive proof of malicious Byzantine intent."
- **Ground-Truth Isolation Badges**: Explicitly tagged as `[RESEARCH-ONLY GROUND TRUTH]` to verify that the server detector never had access to labels during training.

### 6. Trust & Reputation Engine (`TrustReputationPage.jsx`)
- **Component Breakdown Table**: Per-node inspection of Cosine Directional Agreement ($S_k$), Update Distance, Validation Delta ($V_k$), Historical Reputation EMA ($R_k$), Composite Trust ($T_k$), Normalized Weights ($w_k$), and Reason Codes (`NEGATIVE_SIMILARITY_INVERSION`, `HIGH_DIRECTIONAL_AGREEMENT`).
- **Formula & Thresholds**: Explains $\alpha=0.40, \beta=0.35, \gamma=0.25$, suspicious cutoff (0.45), and quarantine threshold (0.25).
- **Multi-Dimensional Radar Signature**: Visualizing honest nodes vs adversary `client_D`.

### 7. FedAvg vs. Trust-Aware Defense Comparison (`ComparisonPage.jsx`)
- **4 Evaluated Experimental Arms**:
  1. Vulnerable FedAvg under Sign-Flipping (20% malicious influence, loss 0.0567).
  2. Coordinate-Wise Median Robust Baseline (20% influence, loss 0.00041).
  3. Trust-Aware FedAvg Defense (0% malicious influence, loss 0.00033).
  4. Clean Target Benchmark (0% influence, loss 0.00033).
- **Trade-Off Analysis**: Why similarity alone is insufficient; why validation delta can be noisy on non-IID data; how reputation protects against transient noise.

### 8. Blockchain Audit Trail (`BlockchainAuditPage.jsx`)
- **ReputationRegistry Contract**: Deployed contract address, EVM block height, and Chain ID 1337.
- **On-Chain Audit Blocks Ledger**: Round, transaction hash, pseudonymous client ID, model update hash, decision, reputation change, new reputation, and canonical record hash.
- **Tamper-Evident Verification**: Real-time validation badge `[TAMPER-EVIDENT OK]`.

### 9. Experiment Results Repository (`ExperimentsPage.jsx`)
- **Artifact Inspector**: Local results from `experiments/results/` (`trust_benchmark_summary.json`, `blockchain_audit_trail.json`).
- **Reproducibility Guarantee**: Random seed 42, PyTorch 2.3.1, Flower 1.33.0, fixed non-IID partitions.

---

## 3. How to Launch

```powershell
# 1. Start the FastAPI Backend:
python run_project.py --mode api

# 2. In another terminal, start the React Dashboard:
python run_project.py --mode dashboard

# 3. Or launch Vite directly from frontend/:
cd frontend
npm run dev
```

Dashboard is accessible at **[http://127.0.0.1:5173](http://127.0.0.1:5173)**.
Production bundle can be verified at any time via `npm run build`.
