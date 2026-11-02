"""
Pydantic Schemas for FastAPI Backend.
All schemas define strict types, validation ranges, clear demo-data labels,
and strict privacy enforcement ensuring no raw meter readings are exposed.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# 1. Health Schemas
# ------------------------------------------------------------------------------

class DatabaseHealthInfo(BaseModel):
    provider: Literal["mongodb", "in_memory"]
    connected: bool
    database_name: str
    message: str


class BlockchainHealthInfo(BaseModel):
    provider: str
    connected: bool
    block_number: Optional[int] = None
    contract_address: Optional[str] = None
    message: str


class ModelReadinessInfo(BaseModel):
    ready: bool
    available_models: List[str]
    checkpoint_dir: str


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "starting"]
    project_name: str
    environment: str
    database: DatabaseHealthInfo
    blockchain: BlockchainHealthInfo
    model_readiness: ModelReadinessInfo
    version: str = "1.0.0"
    timestamp: str


# ------------------------------------------------------------------------------
# 2. Smart Meter Summary Schemas (NEVER EXPOSE RAW READINGS)
# ------------------------------------------------------------------------------

class ConsumptionMetricsAggregate(BaseModel):
    mean_kwh: float = Field(..., description="Mean reported hourly kWh across all meters")
    std_kwh: float = Field(..., description="Standard deviation of consumption")
    min_kwh: float = Field(..., description="Minimum reported consumption")
    max_kwh: float = Field(..., description="Maximum reported consumption")
    median_kwh: float = Field(..., description="Median reported consumption")
    p25_kwh: float = Field(..., description="25th percentile consumption")
    p75_kwh: float = Field(..., description="75th percentile consumption")
    total_consumed_kwh: float = Field(..., description="Aggregate grid energy reported")


class MetersSummaryResponse(BaseModel):
    total_meters: int
    total_readings_sampled: int
    sampling_frequency_minutes: int
    theft_ratio: float
    consumption_metrics: ConsumptionMetricsAggregate
    status_breakdown: Dict[str, int]
    event_breakdown: Dict[str, int]
    consumer_profiles_breakdown: Dict[str, int]
    hardware_tamper_alerts: int
    estimated_unreported_theft_kwh: float
    data_source_label: Literal["SIMULATED_DEMO_DATA", "SGCC_REAL_WORLD_BENCHMARK"]
    demo_notice: str
    privacy_notice: str = "CONFIDENTIALITY INVARIANT: Individual meter time-series and consumer PII are strictly redacted. Only privacy-preserving aggregates are exposed."
    timestamp: str


# ------------------------------------------------------------------------------
# 3. Substation Client Schemas
# ------------------------------------------------------------------------------

class ClientDetail(BaseModel):
    client_id: str
    substation_name: str
    region_type: str
    meters_count: int
    theft_meters: int
    theft_rate_pct: float
    mean_daily_kwh: float = Field(..., description="Regional load intensity showing non-IID characteristics")
    extracted_sequences: int
    current_reputation: float
    current_trust_score: float
    status: Literal["ACTIVE", "QUARANTINED", "MONITORED"]
    privacy_firewall: str = "ACTIVE"
    central_fl_access: str = "BLOCKED"
    local_data_path: str


class ClientListResponse(BaseModel):
    clients: List[ClientDetail]
    total_clients: int
    active_clients: int
    quarantined_clients: int
    non_iid_characteristics: str
    privacy_guarantee: str = "Local raw readings remain strictly partitioned at the substation level. Only model parameter gradients and metadata are communicated."
    data_source_label: Literal["SIMULATED_DEMO_DATA", "SGCC_REAL_WORLD_BENCHMARK"]
    timestamp: str


# ------------------------------------------------------------------------------
# 4. Federated Learning Rounds Schemas
# ------------------------------------------------------------------------------

class FLRoundRecord(BaseModel):
    round_number: int
    strategy: str
    loss: float
    accuracy: float
    f1_score: float
    pr_auc: float
    roc_auc: float
    communication_mb: float
    participating_clients: List[str]
    quarantined_clients: List[str]
    timestamp: str


class FLRoundsResponse(BaseModel):
    rounds: List[FLRoundRecord]
    total_rounds_completed: int
    best_f1_score: float
    total_communication_mb: float
    convergence_status: str
    active_strategy: str
    data_source_label: Literal["SIMULATED_DEMO_DATA", "SGCC_REAL_WORLD_BENCHMARK"]
    epistemic_disclaimer: str
    timestamp: str


# ------------------------------------------------------------------------------
# 5. FL Start Schemas (Safe Experiment Parameters & Auth Warning)
# ------------------------------------------------------------------------------

class FLStartRequest(BaseModel):
    rounds: int = Field(5, ge=1, le=20, description="Number of FL rounds (1-20)")
    epochs: int = Field(1, ge=1, le=5, description="Local training epochs per round (1-5)")
    model: Literal["cnn_lstm", "lstm"] = "cnn_lstm"
    strategy: Literal["trust_aware_fedavg", "fedavg", "coordinate_median"] = "trust_aware_fedavg"
    attack_type: Literal["none", "sign_flipping", "model_poisoning", "random_noise", "label_flipping"] = "none"
    malicious_fraction: float = Field(0.2, ge=0.0, le=0.5, description="Fraction of Byzantine nodes (0.0-0.5)")
    enable_blockchain: bool = True
    seed: int = Field(42, ge=1, le=99999, description="Random seed for reproducibility")


class FLStartResponse(BaseModel):
    job_id: str
    status: Literal["ACCEPTED", "RUNNING", "COMPLETED", "FAILED"]
    message: str
    parameters: Dict
    estimated_duration_sec: int
    production_warning: str = (
        "CRITICAL SECURITY WARNING: Production grid deployment requires mutual TLS, OAuth2/JWT token "
        "authentication, and role-based access control (RBAC). In this development environment, "
        "controls are unauthenticated for demonstration and simulation purposes."
    )
    timestamp: str


# ------------------------------------------------------------------------------
# 6. Predictions Schemas (Privacy & Demo Labels)
# ------------------------------------------------------------------------------

class PredictionMetrics(BaseModel):
    precision: float
    recall: float
    f1_score: float
    pr_auc: float
    roc_auc: float
    test_loss: float


class ConfusionMatrix(BaseModel):
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


class RiskDistribution(BaseModel):
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int


class SubstationPredictionSummary(BaseModel):
    client_id: str
    evaluated_windows: int
    detected_thefts: int
    fraud_rate_pct: float
    mean_anomaly_score: float


class PredictionsResponse(BaseModel):
    total_windows_evaluated: int
    normal_windows_count: int
    theft_windows_count: int
    theft_ratio_detected: float
    metrics: PredictionMetrics
    confusion_matrix: ConfusionMatrix
    risk_distribution: RiskDistribution
    substation_breakdown: List[SubstationPredictionSummary]
    data_source_label: Literal["SIMULATED_DEMO_DATA", "SGCC_REAL_WORLD_BENCHMARK"]
    demo_notice: str
    privacy_note: str = "Raw meter telemetry is not returned. Responses contain only aggregated fraud detection metrics and risk counts."
    timestamp: str


# ------------------------------------------------------------------------------
# 6b. Live Meter Risk Evaluation Schemas
# ------------------------------------------------------------------------------

class MeterRiskEvaluationRequest(BaseModel):
    substation_id: str = Field(default="client_A", description="Substation identifier (client_A to client_E)")
    meter_id: Optional[str] = Field(default=None, description="Optional meter identifier (e.g. MTR-IND-0104)")
    pattern_type: Literal[
        "NORMAL",
        "BYPASS_THEFT",
        "SCALING_THEFT",
        "REVERSE_CURRENT",
        "CUSTOM",
    ] = Field(default="NORMAL", description="Simulated consumption pattern to evaluate")
    scaling_factor: Optional[float] = Field(default=0.35, ge=0.0, le=1.0, description="Scaling factor for SCALING_THEFT")
    evening_drop_pct: Optional[float] = Field(default=0.75, ge=0.0, le=0.99, description="Drop ratio for BYPASS_THEFT")
    custom_readings: Optional[List[float]] = Field(default=None, description="Optional custom 30-day daily consumption sequence (kWh)")


class DailyConsumptionPoint(BaseModel):
    day: int
    reported_kwh: float
    expected_baseline_kwh: float
    is_anomaly: bool


class MeterRiskEvaluationResponse(BaseModel):
    meter_id: str
    substation_id: str
    substation_archetype: str
    pattern_type: str
    theft_probability: float
    theft_risk_tier: Literal["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]
    risk_label: str
    risk_score_pct: float
    anomaly_flags: List[str]
    recommendation: str
    features_analyzed: Dict[str, float]
    consumption_sequence: List[DailyConsumptionPoint]
    model_version: str
    is_synthetic_evaluation: bool = True
    demo_notice: str = "Simulation Sandbox: Used for interactive grid technician inspection and model verification."
    timestamp: str


# ------------------------------------------------------------------------------
# 7. Byzantine Events Schemas
# ------------------------------------------------------------------------------

class ByzantineEventRecord(BaseModel):
    event_id: str
    round: int
    client_id: str
    attack_type: str
    detected_by_trust: bool
    anomaly_metric: str
    perturbation_norm: float
    action_taken: Literal["QUARANTINED", "WEIGHT_REDUCED", "REPUTATION_PENALIZED", "NONE"]
    ground_truth_label: Literal["MALICIOUS_BYZANTINE", "HONEST_NON_IID"]
    timestamp: str


class ByzantineAttackInjectRequest(BaseModel):
    client_id: Literal["client_A", "client_B", "client_C", "client_D", "client_E"] = "client_A"
    attack_type: Literal["sign_flipping", "gaussian_poisoning", "scaling_poison", "zero_freeze", "none"] = "sign_flipping"
    magnitude: float = Field(3.5, ge=1.0, le=10.0)


class ByzantineEventsResponse(BaseModel):
    events: List[ByzantineEventRecord]
    total_events: int
    detected_events: int
    mitigation_summary: Dict[str, int]
    data_source_label: Literal["SIMULATED_DEMO_DATA", "SGCC_REAL_WORLD_BENCHMARK"]
    disclaimer: str = (
        "RESEARCH NOTICE: An unusual, high-norm, or divergent update is NOT definitive proof of malicious Byzantine "
        "intent. Regional non-IID load heterogeneity naturally creates statistical outliers. Trust scoring balances "
        "outlier dampening with variance tolerance."
    )
    timestamp: str


# ------------------------------------------------------------------------------
# 8. Trust Scores Schemas
# ------------------------------------------------------------------------------

class ClientTrustScoreRecord(BaseModel):
    client_id: str
    round: int
    similarity_score: float
    distance_metric: float
    validation_delta: float
    reputation_score: float
    composite_trust_score: float
    normalized_weight: float
    is_quarantined: bool
    status: Literal["TRUSTED", "SUSPICIOUS", "QUARANTINED"]
    timestamp: str


class TrustScoresResponse(BaseModel):
    scores: List[ClientTrustScoreRecord]
    weights_used: Dict[str, float]
    thresholds: Dict[str, float]
    data_source_label: Literal["SIMULATED_DEMO_DATA", "SGCC_REAL_WORLD_BENCHMARK"]
    summary: str
    timestamp: str


# ------------------------------------------------------------------------------
# 9. Blockchain Audit Schemas
# ------------------------------------------------------------------------------

class OnChainAuditRecord(BaseModel):
    round: int
    client_id: str
    model_update_hash: str
    decision: Literal["ACCEPTED", "QUARANTINED", "WEIGHT_REDUCED", "ATTENUATED", "REJECTED"]
    reputation_change: float
    new_reputation: float
    record_hash: str
    integrity_status: Literal["[TAMPER-EVIDENT OK]", "[INTEGRITY FAILURE]"]
    timestamp: str


class BlockchainAuditBlock(BaseModel):
    block_number: int
    transaction_hash: Optional[str] = "0x0"
    round: int
    contract_address: Optional[str] = "0xF2E246BB76DF876Cef8b38ae84130F4F55De395b"
    timestamp: str
    records: List[OnChainAuditRecord]


class BlockchainAuditResponse(BaseModel):
    blocks: List[BlockchainAuditBlock]
    total_blocks: int
    total_audit_records: int
    contract_address: str
    chain_id: int
    provider_type: str
    epistemic_disclaimer: str = (
        "EVIDENTIARY DISCLAIMER: Blockchain records cryptographic evidence of federated updates, canonical hashes, "
        "and reputation transitions. The blockchain does not itself detect or prevent Byzantine poisoning; "
        "it creates an immutable, verifiable audit trail for non-repudiation."
    )
    data_source_label: Literal["SIMULATED_DEMO_DATA", "SGCC_REAL_WORLD_BENCHMARK", "REAL_ON_CHAIN_AUDIT"]
    timestamp: str


# ------------------------------------------------------------------------------
# 10. Simulation Run Schemas (Safe Parameters & Auth Warning)
# ------------------------------------------------------------------------------

class SimulationRunRequest(BaseModel):
    num_meters: int = Field(100, ge=10, le=2000, description="Total meters to simulate (10-2000)")
    days: int = Field(30, ge=1, le=90, description="Duration in days (1-90)")
    sampling_freq_minutes: Literal[15, 30, 60] = 15
    theft_ratio: float = Field(0.15, ge=0.0, le=0.5, description="Theft meter fraction (0.0-0.5)")
    seed: int = Field(42, ge=1, le=99999, description="Random seed")


class SimulationRunResponse(BaseModel):
    simulation_id: str
    status: Literal["COMPLETED", "RUNNING", "FAILED"]
    parameters: Dict
    generated_summary: Dict
    production_warning: str = (
        "CRITICAL SECURITY WARNING: Production deployment requires mutual TLS, OAuth2/JWT token "
        "authentication, and role-based access control (RBAC). In this development environment, "
        "simulation execution is unauthenticated for rapid prototyping."
    )
    timestamp: str
