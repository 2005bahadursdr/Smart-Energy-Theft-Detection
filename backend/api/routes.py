"""
FastAPI Route Handlers.

Exposes all required REST endpoints:
- GET  /health
- GET  /api/meters/summary
- GET  /api/clients
- GET  /api/fl/rounds
- POST /api/fl/start
- GET  /api/predictions
- GET  /api/byzantine/events
- GET  /api/trust/scores
- GET  /api/blockchain/audit
- POST /api/simulation/run
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Annotated, Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from backend.config import BackendSettings, get_settings
from backend.database.adapter import BaseDatabaseAdapter, get_database_adapter
from backend.schemas import (
    BlockchainAuditResponse,
    BlockchainHealthInfo,
    ByzantineAttackInjectRequest,
    ByzantineEventsResponse,
    ClientListResponse,
    DatabaseHealthInfo,
    FLRoundsResponse,
    FLStartRequest,
    FLStartResponse,
    HealthResponse,
    MetersSummaryResponse,
    MeterRiskEvaluationRequest,
    MeterRiskEvaluationResponse,
    ModelReadinessInfo,
    PredictionsResponse,
    SimulationRunRequest,
    SimulationRunResponse,
    TrustScoresResponse,
)
from backend.services.blockchain_service import BlockchainService
from backend.services.byzantine_service import ByzantineService
from backend.services.client_service import ClientService
from backend.services.experiment_service import ExperimentService
from backend.services.fl_service import FLService
from backend.services.meters_service import MetersService
from backend.services.prediction_service import PredictionService
from backend.services.simulation_service import SimulationService
from backend.services.trust_service import TrustService

logger = logging.getLogger("backend.api")

# Main router for /api and direct /health
router = APIRouter()


# ------------------------------------------------------------------------------
# Dependency Providers
# ------------------------------------------------------------------------------

def get_db() -> BaseDatabaseAdapter:
    """Dependency injection provider for database adapter."""
    return get_database_adapter()


def get_meters_svc(db: Annotated[BaseDatabaseAdapter, Depends(get_db)]) -> MetersService:
    return MetersService(db=db)


def get_client_svc(db: Annotated[BaseDatabaseAdapter, Depends(get_db)]) -> ClientService:
    return ClientService(db=db)


def get_fl_svc(db: Annotated[BaseDatabaseAdapter, Depends(get_db)]) -> FLService:
    return FLService(db=db)


def get_pred_svc(db: Annotated[BaseDatabaseAdapter, Depends(get_db)]) -> PredictionService:
    return PredictionService(db=db)


def get_byz_svc(db: Annotated[BaseDatabaseAdapter, Depends(get_db)]) -> ByzantineService:
    return ByzantineService(db=db)


def get_trust_svc(db: Annotated[BaseDatabaseAdapter, Depends(get_db)]) -> TrustService:
    return TrustService(db=db)


def get_bc_svc(db: Annotated[BaseDatabaseAdapter, Depends(get_db)]) -> BlockchainService:
    return BlockchainService(db=db)


def get_sim_svc(db: Annotated[BaseDatabaseAdapter, Depends(get_db)]) -> SimulationService:
    return SimulationService(db=db)


def get_exp_svc() -> ExperimentService:
    return ExperimentService()


# ------------------------------------------------------------------------------
# 1. Health Endpoint: GET /health and GET /api/health
# ------------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Readiness Probe",
    description="Returns connectivity status of Database (MongoDB vs In-Memory), Blockchain, and Deep Learning models.",
    tags=["Diagnostics"],
)
def get_health(
    db: Annotated[BaseDatabaseAdapter, Depends(get_db)],
    settings: Annotated[BackendSettings, Depends(get_settings)],
) -> HealthResponse:
    """Check system health, database connection, blockchain node, and model checkpoints."""
    # Check Database
    db_ok = False
    try:
        db_ok = db.ping()
        db_provider = db.get_provider_name()
        db_msg = "Database connected and responsive." if db_ok else "Database unresponsive."
    except Exception as exc:
        db_provider = "in_memory"
        db_msg = f"Database error: {exc}"

    # Check Blockchain
    bc_connected = True
    bc_block = 2
    try:
        from blockchain.service import get_blockchain_adapter
        bc_adapter = get_blockchain_adapter()
        bc_connected = bc_adapter.is_connected()
        bc_block = bc_adapter.get_record_count()
        bc_msg = f"Blockchain provider operational ({settings.blockchain_provider_type})."
    except Exception as exc:
        bc_connected = False
        bc_msg = f"Blockchain offline or mocked: {exc}"

    # Check Models
    checkpoint_dir = Path("fl_system/checkpoints")
    models_ready = checkpoint_dir.exists()
    available_models = ["CNN_LSTM", "LSTM"]

    overall_status = "healthy" if db_ok else "degraded"

    return HealthResponse(
        status=overall_status,
        project_name=settings.project_name,
        environment=settings.environment,
        database=DatabaseHealthInfo(
            provider=db_provider,
            connected=db_ok,
            database_name=settings.mongodb_db_name,
            message=db_msg,
        ),
        blockchain=BlockchainHealthInfo(
            provider=settings.blockchain_provider_type,
            connected=bc_connected,
            block_number=bc_block,
            contract_address=settings.reputation_registry_contract_address,
            message=bc_msg,
        ),
        model_readiness=ModelReadinessInfo(
            ready=models_ready,
            available_models=available_models,
            checkpoint_dir=str(checkpoint_dir),
        ),
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ------------------------------------------------------------------------------
# 2. Smart Meter Summary: GET /api/meters/summary
# ------------------------------------------------------------------------------

@router.get(
    "/api/meters/summary",
    response_model=MetersSummaryResponse,
    summary="Aggregated Smart Meter Statistics",
    description=(
        "Returns privacy-preserving statistical summaries (mean, std, quartiles, total consumption) "
        "and fraud event counts across the smart grid. "
        "PRIVACY INVARIANT: Individual meter time-series readings and consumer PII are NEVER exposed."
    ),
    tags=["Smart Meters"],
)
def get_meters_summary(
    svc: Annotated[MetersService, Depends(get_meters_svc)]
) -> MetersSummaryResponse:
    try:
        return svc.get_meters_summary()
    except Exception as exc:
        logger.error(f"Error fetching meters summary: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve meter statistics: {exc}",
        )


# ------------------------------------------------------------------------------
# 3. Substation Clients: GET /api/clients
# ------------------------------------------------------------------------------

@router.get(
    "/api/clients",
    response_model=ClientListResponse,
    summary="Substation Partition Client Profiles",
    description="Returns metadata for substations client_A through client_E, highlighting regional non-IID load heterogeneity and local privacy firewalls.",
    tags=["Federated Learning"],
)
def get_clients(
    svc: Annotated[ClientService, Depends(get_client_svc)]
) -> ClientListResponse:
    try:
        return svc.get_clients()
    except Exception as exc:
        logger.error(f"Error fetching clients: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve client metadata: {exc}",
        )


# ------------------------------------------------------------------------------
# 4. FL Rounds: GET /api/fl/rounds
# ------------------------------------------------------------------------------

@router.get(
    "/api/fl/rounds",
    response_model=FLRoundsResponse,
    summary="Federated Learning Round Telemetry",
    description="Returns historical FL training rounds, validation loss, F1-scores, PR-AUC, and participant quarantine status.",
    tags=["Federated Learning"],
)
def get_fl_rounds(
    svc: Annotated[FLService, Depends(get_fl_svc)]
) -> FLRoundsResponse:
    try:
        return svc.get_fl_rounds()
    except Exception as exc:
        logger.error(f"Error fetching FL rounds: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve FL round history: {exc}",
        )


# ------------------------------------------------------------------------------
# 5. FL Start: POST /api/fl/start
# ------------------------------------------------------------------------------

@router.post(
    "/api/fl/start",
    response_model=FLStartResponse,
    summary="Trigger Federated Learning Simulation Run",
    description=(
        "Launches an asynchronous FL simulation with safe, validated experiment parameters. "
        "WARNING: In production smart grids, control actions require mutual TLS, OAuth2/JWT token authentication, "
        "and strict role-based access control (RBAC)."
    ),
    tags=["Federated Learning Controls"],
)
def start_fl_simulation(
    request: FLStartRequest,
    svc: Annotated[FLService, Depends(get_fl_svc)],
) -> FLStartResponse:
    try:
        return svc.start_fl_simulation(request)
    except Exception as exc:
        logger.error(f"Error launching FL simulation: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to launch FL simulation: {exc}",
        )


# ------------------------------------------------------------------------------
# 6. Predictions: GET /api/predictions
# ------------------------------------------------------------------------------

@router.get(
    "/api/predictions",
    response_model=PredictionsResponse,
    summary="Energy Theft Prediction Metrics",
    description=(
        "Returns privacy-preserving aggregated energy theft prediction metrics, confusion matrix, "
        "and risk distributions. NEVER exposes individual consumer meter readings."
    ),
    tags=["Predictions & Anomaly Detection"],
)
def get_predictions(
    svc: Annotated[PredictionService, Depends(get_pred_svc)]
) -> PredictionsResponse:
    try:
        return svc.get_predictions_summary()
    except Exception as exc:
        logger.error(f"Error fetching predictions: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve prediction metrics: {exc}",
        )


@router.post(
    "/api/predictions/evaluate",
    response_model=MeterRiskEvaluationResponse,
    summary="Evaluate Live Smart Meter Theft Risk",
    description=(
        "Performs live energy theft risk inference on an individual smart meter sequence or simulated pattern. "
        "Evaluates load reduction ratio, zero-current streaks, and deep neural theft probability."
    ),
    tags=["Predictions & Anomaly Detection"],
)
def evaluate_meter_risk(
    request: MeterRiskEvaluationRequest,
    svc: Annotated[PredictionService, Depends(get_pred_svc)],
) -> MeterRiskEvaluationResponse:
    try:
        return svc.evaluate_meter_risk(request)
    except Exception as exc:
        logger.error(f"Error evaluating meter risk: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate meter theft risk: {exc}",
        )


@router.get(
    "/api/predictions/samples",
    summary="Get Sample Smart Meters from Real Dataset",
    description="Returns real SGCC smart meter IDs partitioned by substation for quick interactive risk evaluation.",
    tags=["Predictions & Anomaly Detection"],
)
def get_prediction_samples(
    svc: Annotated[PredictionService, Depends(get_pred_svc)],
) -> Dict[str, Any]:
    try:
        return svc.get_sample_meters()
    except Exception as exc:
        logger.error(f"Error fetching prediction sample meters: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch sample meters: {exc}",
        )


# ------------------------------------------------------------------------------
# 7. Byzantine Events: GET /api/byzantine/events
# ------------------------------------------------------------------------------

@router.get(
    "/api/byzantine/events",
    response_model=ByzantineEventsResponse,
    summary="Byzantine Anomaly & Attack Events",
    description="Returns detected and ground-truth Byzantine perturbation events across federated training rounds.",
    tags=["Byzantine Resilience"],
)
def get_byzantine_events(
    svc: Annotated[ByzantineService, Depends(get_byz_svc)]
) -> ByzantineEventsResponse:
    try:
        return svc.get_byzantine_events()
    except Exception as exc:
        logger.error(f"Error fetching Byzantine events: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve Byzantine events: {exc}",
        )


@router.post(
    "/api/byzantine/inject",
    summary="Dynamically Inject Adversarial Byzantine Attack System-Wide",
    description="Injects an adversarial attack into any substation node and updates trust scores, FL rounds, and quarantine status across the entire system.",
    tags=["Byzantine Resilience"],
)
def inject_byzantine_attack(
    request: ByzantineAttackInjectRequest,
    svc: Annotated[ByzantineService, Depends(get_byz_svc)],
) -> Dict[str, Any]:
    try:
        return svc.inject_attack(request)
    except Exception as exc:
        logger.error(f"Error injecting Byzantine attack: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to inject Byzantine attack: {exc}",
        )


@router.post(
    "/api/byzantine/reset",
    summary="Reset Byzantine Attack State to Baseline",
    description="Resets the federated grid to default benchmark state.",
    tags=["Byzantine Resilience"],
)
def reset_byzantine_attacks(
    svc: Annotated[ByzantineService, Depends(get_byz_svc)],
) -> Dict[str, Any]:
    try:
        return svc.reset_attacks()
    except Exception as exc:
        logger.error(f"Error resetting Byzantine attacks: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset Byzantine attacks: {exc}",
        )


# ------------------------------------------------------------------------------
# 8. Trust Scores: GET /api/trust/scores
# ------------------------------------------------------------------------------

@router.get(
    "/api/trust/scores",
    response_model=TrustScoresResponse,
    summary="Client Component Trust Scores",
    description="Returns multi-criteria trust breakdowns (similarity, distance, validation, reputation, composite score) and dynamic aggregation weights.",
    tags=["Trust Mechanism"],
)
def get_trust_scores(
    svc: Annotated[TrustService, Depends(get_trust_svc)]
) -> TrustScoresResponse:
    try:
        return svc.get_trust_scores()
    except Exception as exc:
        logger.error(f"Error fetching trust scores: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve trust scores: {exc}",
        )


# ------------------------------------------------------------------------------
# 9. Blockchain Audit: GET /api/blockchain/audit
# ------------------------------------------------------------------------------

@router.get(
    "/api/blockchain/audit",
    response_model=BlockchainAuditResponse,
    summary="On-Chain Byzantine Audit Ledger",
    description=(
        "Returns tamper-evident audit records from the Ethereum ReputationRegistry smart contract, "
        "including canonical record hashes and [TAMPER-EVIDENT OK] cryptographic verification."
    ),
    tags=["Blockchain Audit"],
)
def get_blockchain_audit(
    svc: Annotated[BlockchainService, Depends(get_bc_svc)]
) -> BlockchainAuditResponse:
    try:
        return svc.get_blockchain_audit()
    except Exception as exc:
        logger.error(f"Error fetching blockchain audit: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve blockchain audit records: {exc}",
        )


# ------------------------------------------------------------------------------
# 10. Simulation Run: POST /api/simulation/run
# ------------------------------------------------------------------------------

@router.post(
    "/api/simulation/run",
    response_model=SimulationRunResponse,
    summary="Trigger Smart Meter Grid Simulation",
    description=(
        "Triggers synthetic smart meter simulation with safe, validated parameters. "
        "WARNING: In production smart grids, control actions require authentication."
    ),
    tags=["Simulation Controls"],
)
def run_simulation(
    request: SimulationRunRequest,
    svc: Annotated[SimulationService, Depends(get_sim_svc)],
) -> SimulationRunResponse:
    try:
        return svc.run_simulation(request)
    except Exception as exc:
        logger.error(f"Error executing simulation: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute simulation: {exc}",
        )


# ------------------------------------------------------------------------------
# 11. Experiments & Benchmark Results: GET /api/experiments/summary & GET /api/benchmarks/comparison
# ------------------------------------------------------------------------------

@router.get(
    "/api/experiments/summary",
    summary="Fetch Canonical Research Experiment Summaries",
    description="Loads persistent benchmark results and hyperparameters directly from experiments/results/.",
    tags=["Experiments & Benchmarks"],
)
def get_experiments_summary(
    svc: Annotated[ExperimentService, Depends(get_exp_svc)]
):
    try:
        return svc.get_experiment_summary()
    except Exception as exc:
        logger.error(f"Error fetching experiment summary: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load experiment summaries: {exc}",
        )


@router.get(
    "/api/benchmarks/comparison",
    summary="Fetch Multi-Arm Comparative Aggregation Benchmarks",
    description="Returns empirical measurements across FedAvg, Coordinate-Median, and Trust-Aware FedAvg.",
    tags=["Experiments & Benchmarks"],
)
def get_benchmarks_comparison(
    svc: Annotated[ExperimentService, Depends(get_exp_svc)]
):
    try:
        return svc.get_comparison_benchmarks()
    except Exception as exc:
        logger.error(f"Error fetching benchmark comparison: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load comparison benchmarks: {exc}",
        )
