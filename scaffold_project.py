"""
Project Scaffolding Script
Initializes directory structure and package __init__.py files for
Blockchain-Based Byzantine-Resilient Federated Learning for Smart Grid Energy Theft Detection.
"""

import os
from pathlib import Path

PACKAGE_DIRS = [
    "smart_meter_simulator",
    "mdms_simulator",
    "mdms_simulator/api",
    "fl_system",
    "fl_system/server",
    "fl_system/clients",
    "fl_system/clients/client_A",
    "fl_system/clients/client_B",
    "fl_system/clients/client_C",
    "fl_system/clients/client_D",
    "fl_system/clients/client_E",
    "fl_system/models",
    "byzantine",
    "byzantine/attacks",
    "byzantine/detection",
    "trust",
    "trust/similarity",
    "trust/validation",
    "trust/reputation",
    "aggregation",
    "blockchain",
    "backend",
    "backend/database",
    "backend/api",
    "backend/services",
    "tests",
]

NON_PACKAGE_DIRS = [
    "data/raw",
    "data/processed",
    "data/synthetic",
    "config",
    "blockchain/contracts",
    "fl_system/checkpoints",
    "experiments/baseline",
    "experiments/byzantine",
    "experiments/trust_based",
    "experiments/results",
    "notebooks",
    "docs/architecture",
    "docs/research",
    "docs/api",
    "docs/screenshots",
    "frontend/src/components",
    "frontend/src/pages",
    "frontend/src/services",
]


def scaffold() -> None:
    root = Path(__file__).resolve().parent

    for d in PACKAGE_DIRS:
        target_dir = root / d
        target_dir.mkdir(parents=True, exist_ok=True)
        init_file = target_dir / "__init__.py"
        if not init_file.exists():
            pkg_name = d.replace("/", ".")
            with open(init_file, "w", encoding="utf-8") as f:
                f.write(
                    f'"""{pkg_name} module.\n'
                    f'Part of Blockchain-Based Byzantine-Resilient FL for Smart Grid Theft Detection.\n'
                    f'"""\n'
                )

    for d in NON_PACKAGE_DIRS:
        target_dir = root / d
        target_dir.mkdir(parents=True, exist_ok=True)

    print("Project directory scaffolding initialized successfully.")


if __name__ == "__main__":
    scaffold()
