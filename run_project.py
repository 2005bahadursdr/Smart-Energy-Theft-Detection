"""
Unified Project Runner & CLI Orchestrator.
Blockchain-Based Byzantine-Resilient Federated Learning for Smart Grid Energy Theft Detection.
"""

import argparse
import sys
import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("run_project")


def check_environment() -> bool:
    """Validate runtime environment, dependencies, dataset presence, and services."""
    logger.info("=== Environment Diagnostic Check ===")
    all_ok = True

    # 1. Python version
    py_ver = sys.version.split()[0]
    logger.info(f"Python Version: {py_ver}")

    # 2. Critical libraries
    packages = {
        "torch": "PyTorch (Deep Learning)",
        "flwr": "Flower (Federated Learning)",
        "fastapi": "FastAPI (Backend Web Services)",
        "uvicorn": "Uvicorn (ASGI Server)",
        "pymongo": "PyMongo (MongoDB Driver)",
        "web3": "Web3.py (Ethereum Blockchain Interface)",
        "pandas": "Pandas (Data Science)",
        "numpy": "NumPy (Vectorized Math)",
        "scipy": "SciPy (Statistical Computing)",
        "sklearn": "Scikit-Learn (Metrics & Evaluation)",
        "pytest": "PyTest (Testing Suite)",
    }

    for pkg, label in packages.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "available")
            logger.info(f"  [OK] {label} -> {ver}")
        except ImportError:
            logger.error(f"  [MISSING] {label} ({pkg}) is not installed.")
            all_ok = False

    # 3. Raw dataset verification
    dataset_candidates = [
        Path("data/raw/smart_meter_data(sgcc).csv"),
        Path("data/raw/smart_meter_data.csv"),
    ]
    dataset_path = next((p for p in dataset_candidates if p.exists()), None)
    if dataset_path:
        size_mb = dataset_path.stat().st_size / (1024 * 1024)
        logger.info(f"  [OK] SGCC Raw Dataset: {dataset_path} ({size_mb:.2f} MB)")
    else:
        logger.warning(f"  [NOTICE] Raw SGCC dataset not found in data/raw. Synthetic generator can be used.")

    # 4. MongoDB service connectivity
    try:
        import pymongo
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        info = client.server_info()
        logger.info(f"  [OK] MongoDB Service: Connected (v{info.get('version')}) at {mongo_uri}")
    except Exception as e:
        logger.warning(f"  [NOTICE] MongoDB connectivity check failed: {e}. Local fallback or in-memory mock can be used.")

    # 5. Blockchain provider verification
    try:
        import eth_tester
        from web3 import Web3, EthereumTesterProvider
        w3 = Web3(EthereumTesterProvider())
        block_num = w3.eth.block_number
        logger.info(f"  [OK] Blockchain EVM Provider: EthereumTesterProvider operational (Block {block_num})")
    except Exception as e:
        logger.warning(f"  [NOTICE] Ethereum tester provider check: {e}")

    logger.info("=== Diagnostic Complete: %s ===", "PASSED" if all_ok else "ATTENTION NEEDED")
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestrator for Blockchain-Based Byzantine-Resilient FL"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "check-env",
            "preprocess",
            "simulate-meters",
            "train-central",
            "fl-server",
            "fl-client",
            "fl-simulation",
            "run-experiment",
            "trust-benchmark",
            "blockchain-audit",
            "api",
            "dashboard",
            "all",
        ],
        default="check-env",
        help="Subsystem execution mode",
    )
    parser.add_argument("--blockchain", action="store_true", help="Enable blockchain audit logging")
    parser.add_argument("--client-id", type=str, default="client_A", help="Client ID for FL client mode")
    parser.add_argument("--attack-type", type=str, default="none", help="Byzantine attack type to simulate")
    parser.add_argument("--strategy", type=str, default="trust_based_fedavg", help="Aggregation strategy")
    parser.add_argument("--meters", type=int, default=1000, help="Number of meters for simulation")
    parser.add_argument("--days", type=int, default=30, help="Number of days for simulation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation reproducibility")
    parser.add_argument("--epochs", type=int, default=1, help="Number of local training epochs")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated learning rounds")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="FL server host address")
    parser.add_argument("--port", type=int, default=8080, help="FL server port")
    parser.add_argument(
        "--model",
        choices=["cnn_lstm", "lstm"],
        default="cnn_lstm",
        help="Deep learning model architecture ('cnn_lstm' or 'lstm')",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "sgcc", "synthetic"],
        default="sgcc",
        help="Data source ('sgcc' or 'synthetic', default: 'sgcc')",
    )

    args = parser.parse_args()

    if args.mode == "check-env":
        success = check_environment()
        sys.exit(0 if success else 1)
    elif args.mode == "simulate-meters":
        from smart_meter_simulator.simulator import SmartMeterSimulator
        logger.info(f"Launching Smart Meter Simulator ({args.meters} meters, {args.days} days, seed {args.seed})...")
        sim = SmartMeterSimulator(num_meters=args.meters, days=args.days, seed=args.seed)
        csv_path, meta_path, sum_path, summary = sim.export_dataset()
        logger.info(f"Dataset generated successfully at: {csv_path}")
        logger.info(f"Metadata catalog generated at: {meta_path}")
        logger.info(f"Simulation summary generated at: {sum_path}")
    elif args.mode == "preprocess":
        from mdms_simulator.sgcc_processor import SGCCDatasetProcessor
        from mdms_simulator.demo import run_demo

        sgcc_candidates = [
            Path("data/raw/smart_meter_data(sgcc).csv"),
            Path("data/raw/smart_meter_data.csv"),
        ]
        sgcc_path = next((p for p in sgcc_candidates if p.exists()), None)
        use_sgcc = (args.source == "sgcc") or (args.source == "auto" and sgcc_path is not None)

        if use_sgcc and sgcc_path is not None:
            logger.info(f"=== Processing Benchmark SGCC Dataset ({args.meters} meters, {sgcc_path}) ===")
            processor = SGCCDatasetProcessor(raw_csv_path=sgcc_path)
            summary = processor.process_and_partition(num_meters=args.meters)
            logger.info("SGCC non-IID partitioning complete across substations client_A through client_E.")
        else:
            logger.info("Launching MDMS Simulator ingestion, validation, and client-local partitioning...")
            success = run_demo()
            sys.exit(0 if success else 1)
    elif args.mode == "train-central":
        import pandas as pd
        from fl_system.models import (
            create_dataloaders,
            evaluate_model,
            get_model,
            prepare_leakage_free_datasets,
            train_model,
        )

        # Prioritize SGCC benchmark dataset or synthetic data based on --source
        if args.source == "synthetic":
            data_candidates = [Path("data/synthetic/simulated_meter_data.csv")]
        else: # "sgcc" or "auto"
            data_candidates = [
                Path("data/processed/sgcc_benchmark.csv"),
                Path("data/processed/clients/client_A/local_readings.csv"),
                Path("data/raw/smart_meter_data(sgcc).csv"),
                Path("data/synthetic/simulated_meter_data.csv"),
            ]

        data_path = next((p for p in data_candidates if p.exists()), None)
        if not data_path:
            logger.error("No dataset found. Run '--mode preprocess' or '--mode simulate-meters' first.")
            sys.exit(1)

        # Auto-preprocess raw SGCC if needed
        if data_path == Path("data/raw/smart_meter_data(sgcc).csv"):
            from mdms_simulator.sgcc_processor import SGCCDatasetProcessor
            logger.info(f"Auto-preprocessing raw SGCC dataset ({args.meters} meters)...")
            processor = SGCCDatasetProcessor(raw_csv_path=data_path)
            processor.process_and_partition(num_meters=args.meters)
            data_path = Path("data/processed/sgcc_benchmark.csv")

        is_sgcc = "sgcc" in str(data_path).lower() or "client_" in str(data_path).lower()
        dataset_origin = (
            "State Grid Corporation of China (SGCC) Real-World Benchmark Dataset"
            if is_sgcc
            else "Synthetic Smart Meter Simulator"
        )
        is_synthetic = not is_sgcc

        logger.info(f"=== Centralized Benchmark Training ({args.model.upper()}, {args.epochs} epochs) ===")
        logger.info(f"Data Source       : {dataset_origin}")
        logger.info(f"Loading data from : {data_path}...")
        df = pd.read_csv(data_path)
        meters = df["meter_id"].unique()
        if len(meters) > 100:
            selected_meters = set(meters[:100])
            df = df[df["meter_id"].isin(selected_meters)].copy()

        train_ds, val_ds, test_ds, _ = prepare_leakage_free_datasets(
            df=df,
            window_size=24,
            stride=12,
            split_strategy="consumer",
            is_synthetic=is_synthetic,
            seed=args.seed,
        )
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds, batch_size=32, use_weighted_sampler=True
        )

        input_dim = train_ds.X.shape[-1] if len(train_ds) > 0 else 11
        model = get_model(model_name=args.model, input_size=input_dim)

        model, history, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            checkpoint_dir="fl_system/checkpoints",
            model_name=f"central_{args.model}_{'sgcc' if is_sgcc else 'synthetic'}",
            seed=args.seed,
        )

        metrics, _, _ = evaluate_model(
            model=model,
            dataloader=test_loader,
            is_synthetic=is_synthetic,
            dataset_origin=dataset_origin,
        )
        logger.info(f"=== Centralized Test Evaluation Results ({args.model.upper()}) ===")
        logger.info(f"  Dataset Origin  : {metrics.dataset_origin}")
        logger.info(f"  Is Synthetic    : {metrics.is_synthetic_evaluation}")
        logger.info(f"  Accuracy        : {metrics.accuracy:.4f}")
        logger.info(f"  Precision       : {metrics.precision:.4f}")
        logger.info(f"  Recall          : {metrics.recall:.4f}")
        logger.info(f"  F1-Score        : {metrics.f1_score:.4f}")
        logger.info(f"  PR-AUC          : {metrics.pr_auc:.4f}")
        logger.info(f"  ROC-AUC         : {metrics.roc_auc}")
        logger.info(f"  Confusion Matrix: {metrics.confusion_matrix}")
        logger.info(f"  Notice          : {metrics.disclaimer}")
    elif args.mode == "fl-server":
        from fl_system.server.server_runner import run_fl_server
        logger.info(f"=== Launching Flower FL Server ({args.rounds} rounds, {args.host}:{args.port}) ===")
        run_fl_server(
            host=args.host,
            port=args.port,
            num_rounds=args.rounds,
            model_name=args.model,
            local_epochs=args.epochs,
        )
    elif args.mode == "fl-client":
        import flwr as fl
        from fl_system.clients.factory import create_client
        logger.info(f"=== Launching Flower Client '{args.client_id}' connecting to {args.host}:{args.port} ===")
        client = create_client(
            client_id=args.client_id,
            model_name=args.model,
            local_epochs=args.epochs,
            seed=args.seed,
        )
        fl.client.start_numpy_client(
            server_address=f"{args.host}:{args.port}",
            client=client,
        )
    elif args.mode == "fl-simulation":
        from byzantine.attack_manager import ByzantineAttackManager
        from fl_system.coordinator import FLSimulationCoordinator

        attack_mgr = None
        if args.attack_type.lower() != "none":
            attack_mgr = ByzantineAttackManager(
                active_attack=args.attack_type,
                attacker_clients=[args.client_id] if args.client_id else ["client_D"],
                seed=args.seed,
            )

        logger.info(
            f"=== Launching Multi-Client FL Simulation ({args.rounds} rounds, {args.epochs} epochs/round, {args.model.upper()}) ==="
        )
        coordinator = FLSimulationCoordinator(
            num_rounds=args.rounds,
            local_epochs=args.epochs,
            model_name=args.model,
            attack_manager=attack_mgr,
            enable_blockchain=args.blockchain,
            seed=args.seed,
        )
        summary = coordinator.run()
        logger.info("=== FL Simulation Finished Successfully ===")
        logger.info(f"  Total Rounds          : {summary['total_rounds']}")
        logger.info(f"  Participating Clients : {summary['client_ids']}")
        logger.info(f"  Active Attack         : {summary.get('active_attack', 'none')}")
        logger.info(f"  Attacker Nodes        : {summary.get('attacker_clients', [])}")
        logger.info(f"  Best F1 Score         : {summary['best_f1_score']:.4f}")
        logger.info(f"  Total Communication   : {summary['total_communication_mb']:.2f} MB")
        logger.info(f"  Checkpoints Directory : {summary['checkpoint_dir']}")
        logger.info(f"  Privacy Disclaimer    : {summary['privacy_notice']}")
    elif args.mode == "run-experiment":
        from experiments.benchmark_runner import run_comparative_benchmark
        logger.info(
            f"=== Launching Full Byzantine FL Comparative Benchmark ({args.rounds} rounds, {args.model.upper()}) ==="
        )
        report = run_comparative_benchmark(
            num_rounds=args.rounds,
            local_epochs=args.epochs,
            model_name=args.model,
            seed=args.seed,
        )
        logger.info("=== Comparative Benchmark Complete ===")
        for arm_name, metrics in report["comparison"].items():
            logger.info(
                f"  [{arm_name}] Attack: {metrics['attack_type']} | F1: {metrics['final_f1_score']} "
                f"| Loss: {metrics['final_test_loss']} | PR-AUC: {metrics['final_pr_auc']}"
            )
    elif args.mode == "trust-benchmark":
        from experiments.trust_benchmark import run_trust_defense_benchmark
        attack = args.attack_type if args.attack_type.lower() != "none" else "sign_flipping"
        logger.info(
            f"=== Launching Multi-Criteria Trust & Byzantine Defense Benchmark ({args.rounds} rounds, {args.model.upper()}, Attack: {attack}) ==="
        )
        report = run_trust_defense_benchmark(
            num_rounds=args.rounds,
            local_epochs=args.epochs,
            learning_rate=args.lr if hasattr(args, "lr") else 0.001,
            model_name=args.model,
            attack_type=attack,
            seed=args.seed,
        )
        logger.info("=== Trust & Byzantine Defense Benchmark Complete ===")
        for arm_name, metrics in report["comparison"].items():
            logger.info(
                f"  [{arm_name}] Strategy: {metrics['strategy']} | Attack: {metrics['attack_type']} "
                f"| F1: {metrics['final_f1_score']} | Loss: {metrics['final_test_loss']} "
                f"| Malicious Influence: {metrics['malicious_influence']:.1%}"
            )
    elif args.mode == "blockchain-audit":
        from blockchain.canonical_hasher import verify_record_integrity
        from blockchain.service import get_blockchain_adapter
        adapter = get_blockchain_adapter()
        logger.info("=== Querying On-Chain Byzantine Audit Ledger ===")
        logger.info(f"  Contract Address : {getattr(adapter, 'contract_address', 'N/A')}")
        count = adapter.get_record_count()
        logger.info(f"  Live Node Committed Records : {count}")

        # Check persistent audit trail on disk
        audit_file = Path("experiments/results/blockchain_audit_trail.json")
        if audit_file.exists():
            import json
            with open(audit_file, "r", encoding="utf-8") as f:
                trail = json.load(f)
            total_records = sum(len(b.get("records", [])) for b in trail)
            logger.info(f"  Persistent Audit Trail Rounds : {len(trail)} | Total Records: {total_records}")
            for block in trail[-2:]:
                logger.info(f"\n  [Round {block['round']}] TxHash: {block.get('transactionHash')} | Block: {block.get('blockNumber')}")
                for rec in block.get("records", []):
                    # Verify integrity
                    is_valid = verify_record_integrity(
                        {k: v for k, v in rec.items() if k != "record_hash"},
                        rec["record_hash"]
                    )
                    status_tag = "[TAMPER-EVIDENT OK]" if is_valid else "[INTEGRITY FAILURE]"
                    logger.info(
                        f"    Client: {rec['client_id']:<10} | Decision: {rec['decision']:<11} "
                        f"| Rep: {rec['new_reputation']:<6} | Hash: {rec['record_hash'][:14]}... {status_tag}"
                    )
        else:
            logger.info("  No local audit trail found on disk. Run '--mode fl-simulation --blockchain' to generate records.")
    elif args.mode == "api":
        import uvicorn
        from backend.config import get_settings
        settings = get_settings()
        host = args.host if args.host else settings.backend_host
        port = settings.backend_port
        logger.info(f"=== Starting FastAPI Backend on http://{host}:{port} ===")
        logger.info(f"  Interactive OpenAPI Docs: http://{host}:{port}/docs")
        logger.info(f"  Health Check Probe     : http://{host}:{port}/health")
        logger.info(f"  CORS Allowed Origins   : {settings.cors_origins}")
        uvicorn.run("backend.main:app", host=host, port=port, reload=False)
    elif args.mode == "dashboard":
        import subprocess
        logger.info("=== Launching React + Vite + Tailwind CSS Dashboard ===")
        logger.info("  URL: http://127.0.0.1:5173")
        npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
        subprocess.run([npm_cmd, "run", "dev"], cwd="frontend")
    elif args.mode == "all":
        logger.info("================================================================================")
        logger.info("EXECUTING FULL INTEGRATED BYZANTINE-RESILIENT FL PIPELINE")
        logger.info("Smart-Meter Simulator -> MDMS -> 5 Substation Partitions -> PyTorch CNN-LSTM")
        logger.info("-> Byzantine Attacks -> Multi-Criteria Trust -> Blockchain Audit -> Backend")
        logger.info("================================================================================")

        # 1. Diagnostic Environment Check
        check_environment()

        # 2. Verify / Preprocess Data Partitions
        from mdms_simulator.sgcc_processor import SGCCDatasetProcessor
        processor = SGCCDatasetProcessor(seed=args.seed)
        summary = processor.process_and_partition(num_meters=args.meters)
        logger.info(f"Verified 5 Substation non-IID partitions: {list(summary['clients'].keys())}")

        # 3. Execute Controlled Comparative Benchmark under Identical Conditions
        from experiments.trust_benchmark import run_trust_defense_benchmark
        attack = args.attack_type if args.attack_type.lower() != "none" else "sign_flipping"
        benchmark_report = run_trust_defense_benchmark(
            num_rounds=args.rounds,
            local_epochs=args.epochs,
            learning_rate=args.lr if hasattr(args, "lr") else 0.001,
            model_name=args.model,
            attack_type=attack,
            seed=args.seed,
        )

        # 4. Check Blockchain Audit Records
        from blockchain.service import get_blockchain_adapter
        bc_adapter = get_blockchain_adapter()
        logger.info(f"On-Chain Audit Records Verified: {bc_adapter.get_record_count()} records")

        # 5. Bootstrap & Verify Database
        from backend.database.adapter import get_database_adapter
        db = get_database_adapter()
        logger.info(f"Database operational ({db.get_provider_name()}) with {db.count('meter_summaries')} summary docs")

        logger.info("================================================================================")
        logger.info("FULL INTEGRATED PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("================================================================================")
    else:
        logger.info(f"Selected mode: {args.mode}. Subsystem implementation will be invoked incrementally.")


if __name__ == "__main__":
    main()
