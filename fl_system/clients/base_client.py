"""
Smart Grid Flower Federated Learning Client.
Implements NumPyClient for substation edge nodes (client_A through client_E).
Trains local LSTM or CNN-LSTM models exclusively on local partition vaults.
Enforces strict partition isolation and transmits model parameters only (never raw readings).

CRITICAL PRIVACY DOCUMENTATION:
Federated Learning alone DOES NOT guarantee privacy. While raw consumption
telemetry remains strictly on-premise at regional substations and only model
parameter arrays and scalar metadata are transmitted to the server, standard
Federated Learning does not provide mathematical Differential Privacy (DP)
nor cryptographic Secure Multi-Party Computation (SMPC). Differential privacy
or secure aggregation must be implemented separately if formal privacy guarantees
against gradient inversion attacks are required.
"""

from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Scalar

from fl_system.models.dataset import (
    MeterSequenceDataset,
    create_dataloaders,
    prepare_leakage_free_datasets,
)
from fl_system.models.evaluate import (
    ETHICAL_VALIDATION_DISCLAIMER,
    calculate_binary_metrics,
)
from fl_system.models.model_utils import (
    compute_gradient_norm,
    get_model,
    get_parameters_as_numpy,
    set_parameters_from_numpy,
    set_reproducible_seed,
)

logger = logging.getLogger("fl_client")

FL_PRIVACY_BOUNDARY_DISCLAIMER: str = (
    "FEDERATED LEARNING PRIVACY NOTICE: Standard Federated Learning (FedAvg) transmits model parameter "
    "gradients/updates rather than raw smart meter telemetry, ensuring local data isolation at distribution "
    "substations. However, Federated Learning alone DOES NOT guarantee mathematical differential privacy "
    "or cryptographic confidentiality. Parameter updates can potentially be vulnerable to gradient inversion "
    "or reconstruction attacks unless combined with formal Differential Privacy (DP) or Secure Multi-Party "
    "Computation (SMPC) / Secure Aggregation, which must be independently audited and implemented."
)


class SmartGridFlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient representing a regional smart grid distribution substation.
    
    Attributes
    ----------
    client_id : str
        Substation identifier (e.g., 'client_A', 'client_B').
    model_name : str
        Neural network architecture ('cnn_lstm' or 'lstm').
    data_dir : Path
        Local data vault directory path.
    """

    PRIVACY_NOTICE = FL_PRIVACY_BOUNDARY_DISCLAIMER

    def __init__(
        self,
        client_id: str,
        model_name: str = "cnn_lstm",
        data_dir: Optional[Union[str, Path]] = None,
        df: Optional[pd.DataFrame] = None,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        local_epochs: int = 1,
        device: str = "cpu",
        seed: int = 42,
        pos_weight: float = 5.0,
    ):
        super().__init__()
        self.client_id = client_id
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.device = torch.device(device)
        self.seed = seed
        self.pos_weight = pos_weight

        set_reproducible_seed(seed)

        # 1. Enforce Partition Isolation
        self.data_dir: Optional[Path] = None
        if data_dir is not None:
            self.data_dir = Path(data_dir).resolve()
            self._verify_partition_isolation(self.data_dir)

        # 2. Load Local Partition Data
        self.train_dataset: Optional[MeterSequenceDataset] = None
        self.val_dataset: Optional[MeterSequenceDataset] = None
        self.test_dataset: Optional[MeterSequenceDataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

        if df is not None:
            self._init_from_dataframe(df)
        elif self.data_dir is not None:
            self._init_from_directory(self.data_dir)
        else:
            logger.warning(f"Client '{client_id}' initialized without local data. Deferring dataset load.")

        # 3. Instantiate Local PyTorch Model
        input_dim = 11
        if self.train_dataset is not None and len(self.train_dataset) > 0:
            input_dim = self.train_dataset.X.shape[-1]
        self.input_size = input_dim
        self.model = get_model(model_name=self.model_name, input_size=self.input_size).to(self.device)

        # 4. Metrics & Communication Footprint Tracking
        self.cumulative_bytes_sent = 0
        self.cumulative_bytes_received = 0
        self.fit_round_count = 0

        logger.info(
            f"Initialized SmartGridFlowerClient '{self.client_id}' | Model: {self.model_name.upper()} "
            f"| Input Dim: {self.input_size} | Local Partition: {self.data_dir or 'In-Memory DataFrame'}"
        )

    def _verify_partition_isolation(self, target_dir: Path) -> None:
        """
        Verify that client strictly accesses its own designated vault partition.
        Prevents unauthorized cross-substation telemetry leakage.
        """
        dir_name = target_dir.name
        # The directory name must match client_id or contain it as a subfolder
        if dir_name != self.client_id and self.client_id not in str(target_dir):
            raise PermissionError(
                f"Partition Isolation Violation: Client '{self.client_id}' is prohibited "
                f"from accessing unauthorized partition at '{target_dir}'."
            )

    def _init_from_directory(self, data_dir: Path) -> None:
        """Load local telemetry strictly from local partition directory."""
        csv_path = data_dir / "local_readings.csv"
        npz_path = data_dir / "sequences.npz"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Ensure no foreign client data exists inside local file
            if "client_id" in df.columns:
                unique_clients = df["client_id"].unique()
                if any(c != self.client_id for c in unique_clients):
                    raise ValueError(
                        f"Data Isolation Integrity Failure: File '{csv_path}' contains "
                        f"telemetry belonging to foreign clients: {unique_clients}."
                    )
            self._init_from_dataframe(df)
        elif npz_path.exists():
            data = np.load(npz_path)
            X, y = data["X"], data["y"]
            n = len(X)
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)

            self.train_dataset = MeterSequenceDataset(X[:n_train], y[:n_train], is_synthetic=True)
            self.val_dataset = MeterSequenceDataset(X[n_train:n_train + n_val], y[n_train:n_train + n_val], is_synthetic=True)
            self.test_dataset = MeterSequenceDataset(X[n_train + n_val:], y[n_train + n_val:], is_synthetic=True)

            self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
                self.train_dataset, self.val_dataset, self.test_dataset,
                batch_size=self.batch_size, use_weighted_sampler=True
            )
        else:
            raise FileNotFoundError(f"No local dataset found in client vault: {data_dir}")

    def _init_from_dataframe(self, df: pd.DataFrame) -> None:
        """Partition local dataframe into leakage-free train, val, test subsets."""
        is_synthetic = "simulated_theft_label" in df.columns
        train_ds, val_ds, test_ds, _ = prepare_leakage_free_datasets(
            df=df,
            window_size=24,
            stride=12,
            split_strategy="consumer",
            is_synthetic=is_synthetic,
            seed=self.seed,
        )
        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.train_dataset, self.val_dataset, self.test_dataset,
            batch_size=self.batch_size, use_weighted_sampler=True
        )

    # --------------------------------------------------------------------------
    # Flower NumPyClient Core Protocol Implementation
    # --------------------------------------------------------------------------

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """
        Extract and return local model parameters as list of NumPy arrays.
        TRANSMITS WEIGHTS ONLY — NEVER RAW READINGS.
        """
        params = get_parameters_as_numpy(self.model)
        payload_bytes = sum(p.nbytes for p in params)
        logger.debug(f"[{self.client_id}] Returning {len(params)} parameter tensors ({payload_bytes} bytes).")
        return params

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Train local model on local private dataset using received global parameters.

        Parameters
        ----------
        parameters : List[np.ndarray]
            Current global model weights sent by Flower server.
        config : Dict[str, Scalar]
            Round hyperparameters (local_epochs, learning_rate, server_round).

        Returns
        -------
        Tuple[List[np.ndarray], int, Dict[str, Scalar]]
            (updated_weights, num_train_samples, scalar_training_metrics)
        """
        # 1. Update communication telemetry (downlink from server)
        recv_bytes = sum(p.nbytes for p in parameters)
        self.cumulative_bytes_received += recv_bytes

        # 2. Synchronize local model with global parameters
        set_parameters_from_numpy(self.model, parameters)

        # 3. Parse configurable hyperparameters
        epochs = int(config.get("local_epochs", self.local_epochs))
        lr = float(config.get("learning_rate", self.learning_rate))
        server_round = int(config.get("server_round", self.fit_round_count + 1))
        self.fit_round_count += 1

        # 4. Train locally (Strict On-Premise Execution)
        start_time = time.time()
        avg_train_loss, final_grad_norm = self._train_local_epochs(epochs=epochs, lr=lr)
        duration_sec = time.time() - start_time

        # 5. Extract updated weights
        updated_parameters = get_parameters_as_numpy(self.model)
        sent_bytes = sum(p.nbytes for p in updated_parameters)
        self.cumulative_bytes_sent += sent_bytes

        num_samples = len(self.train_dataset) if self.train_dataset is not None else 0

        # 6. Structured scalar metrics only (STRICT PRIVACY: NO RAW READINGS OR IDENTIFIERS)
        metrics: Dict[str, Scalar] = {
            "client_id": self.client_id,
            "server_round": server_round,
            "train_loss": float(avg_train_loss),
            "grad_norm": float(final_grad_norm),
            "epochs_trained": epochs,
            "duration_sec": float(round(duration_sec, 3)),
            "bytes_sent": sent_bytes,
            "bytes_received": recv_bytes,
            "samples_count": num_samples,
        }

        logger.info(
            f"[{self.client_id}] Round {server_round} Fit Complete | Train Loss: {avg_train_loss:.4f} "
            f"| Grad Norm: {final_grad_norm:.4f} | Samples: {num_samples} | Uplink: {sent_bytes / 1024:.1f} KB"
        )

        return updated_parameters, num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate global model parameters against local held-out test partition.

        Returns
        -------
        Tuple[float, int, Dict[str, Scalar]]
            (loss, num_test_samples, scalar_evaluation_metrics)
        """
        recv_bytes = sum(p.nbytes for p in parameters)
        self.cumulative_bytes_received += recv_bytes

        set_parameters_from_numpy(self.model, parameters)

        if self.test_loader is None or len(self.test_dataset) == 0:
            return 0.0, 0, {"accuracy": 0.0, "client_id": self.client_id}

        self.model.eval()
        pos_weight_tensor = torch.tensor([self.pos_weight], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for X_b, y_b in self.test_loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device).float()
                logits = self.model(X_b)
                if logits.dim() == 1 and y_b.dim() > 1:
                    y_b_loss = y_b.squeeze(-1)
                elif logits.dim() > 1 and y_b.dim() == 1:
                    y_b_loss = y_b.unsqueeze(-1)
                else:
                    y_b_loss = y_b
                loss = criterion(logits, y_b_loss)
                total_loss += loss.item() * len(y_b)

                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= 0.5).astype(int)
                targets = y_b.cpu().numpy().flatten().astype(int)

                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_targets.extend(targets.tolist())

        num_samples = len(self.test_dataset)
        avg_loss = total_loss / max(num_samples, 1)

        eval_metrics = calculate_binary_metrics(
            y_true=np.array(all_targets),
            y_prob=np.array(all_probs),
            loss=avg_loss,
        )

        metrics: Dict[str, Scalar] = {
            "client_id": self.client_id,
            "accuracy": float(eval_metrics.accuracy),
            "precision": float(eval_metrics.precision),
            "recall": float(eval_metrics.recall),
            "f1_score": float(eval_metrics.f1_score),
            "pr_auc": float(eval_metrics.pr_auc),
            "roc_auc": float(eval_metrics.roc_auc if eval_metrics.roc_auc is not None else 0.0),
            "samples_count": num_samples,
        }

        logger.info(
            f"[{self.client_id}] Local Evaluation | Loss: {avg_loss:.4f} "
            f"| F1: {eval_metrics.f1_score:.4f} | PR-AUC: {eval_metrics.pr_auc:.4f}"
        )

        return float(avg_loss), num_samples, metrics

    # --------------------------------------------------------------------------
    # Internal Local Training Logic
    # --------------------------------------------------------------------------

    def _train_local_epochs(self, epochs: int, lr: float) -> Tuple[float, float]:
        """Execute local gradient descent on private DataLoader."""
        if self.train_loader is None or len(self.train_dataset) == 0:
            return 0.0, 0.0

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        pos_weight_tensor = torch.tensor([self.pos_weight], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        total_loss = 0.0
        total_batches = 0
        final_grad_norm = 0.0

        for epoch in range(epochs):
            for X_b, y_b in self.train_loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device).float().unsqueeze(-1)

                optimizer.zero_grad()
                logits = self.model(X_b)
                if logits.dim() == 1 and y_b.dim() > 1:
                    y_b = y_b.squeeze(-1)
                elif logits.dim() > 1 and y_b.dim() == 1:
                    y_b = y_b.unsqueeze(-1)
                loss = criterion(logits, y_b)
                loss.backward()

                final_grad_norm = compute_gradient_norm(self.model)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        return float(avg_loss), float(final_grad_norm)
