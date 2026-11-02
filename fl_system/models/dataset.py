"""
Smart Meter Time-Series Dataset and Feature Engineering Module.
Creates 24/48-hour sequence sliding windows, derives historical consumption statistics,
and encodes meter status/hardware event features.
Enforces strict leakage-free splitting (by consumer or time) and fits preprocessing
strictly on training data only. Explicitly tags synthetic labels.
"""

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger("fl_dataset")

# Standard categorical mappings for meter operational status and events
STATUS_MAPPING = {
    "NORMAL": 0,
    "OUTAGE": 1,
    "DISCONNECTED": 2,
    "COMM_FAILURE": 3,
    "TAMPER_ALERT": 4,
}

EVENT_MAPPING = {
    "NONE": 0,
    "GRID_OUTAGE": 1,
    "DISCONNECTION": 2,
    "VACANCY": 3,
    "SOLAR_SELF_GEN": 4,
    "SUSTAINED_DROP": 5,
    "COMM_FAILURE": 6,
    "TAMPER_EVENT": 7,
}


@dataclass
class PreprocessingState:
    """Stores fitted preprocessing statistics derived strictly from training data."""
    min_consumption: float = 0.0
    max_consumption: float = 1.0
    mean_consumption: float = 0.0
    std_consumption: float = 1.0
    median_impute_value: float = 0.0


class TimeSeriesPreprocessor:
    """
    Fits normalization and imputation statistics exclusively on training partitions
    to prevent data leakage into validation or testing sets.
    """

    def __init__(self, scaling: str = "minmax"):
        self.scaling = scaling
        self.state: Optional[PreprocessingState] = None

    def fit(self, df: pd.DataFrame, consumption_col: str = "consumption_kwh") -> "TimeSeriesPreprocessor":
        """Fit normalization parameters strictly on the training partition."""
        vals = df[consumption_col].dropna().values
        if len(vals) == 0:
            self.state = PreprocessingState()
            return self

        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        if max_val <= min_val:
            max_val = min_val + 1.0

        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        if std_val < 1e-6:
            std_val = 1.0

        median_val = float(np.median(vals))

        self.state = PreprocessingState(
            min_consumption=min_val,
            max_consumption=max_val,
            mean_consumption=mean_val,
            std_consumption=std_val,
            median_impute_value=median_val
        )
        return self

    def transform(self, df: pd.DataFrame, consumption_col: str = "consumption_kwh") -> pd.DataFrame:
        """Apply fitted parameters to any partition without recalculation."""
        if self.state is None:
            raise RuntimeError("TimeSeriesPreprocessor must be fitted on training data before transforming.")

        df_out = df.copy()
        # 1. Impute missing consumption using training median
        df_out[consumption_col] = df_out[consumption_col].fillna(self.state.median_impute_value)

        # 2. Scale consumption
        if self.scaling == "minmax":
            df_out["norm_consumption"] = (df_out[consumption_col] - self.state.min_consumption) / (
                self.state.max_consumption - self.state.min_consumption
            )
            df_out["norm_consumption"] = df_out["norm_consumption"].clip(0.0, 1.0)
        else: # standard z-score
            df_out["norm_consumption"] = (df_out[consumption_col] - self.state.mean_consumption) / self.state.std_consumption

        return df_out


class MeterSequenceDataset(Dataset):
    """
    PyTorch Dataset representing multi-feature sliding window sequences
    for smart grid energy theft / anomaly detection.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        meter_ids: List[str],
        timestamps: List[str],
        is_synthetic: bool = True,
        feature_names: Optional[List[str]] = None
    ):
        """
        Parameters
        ----------
        X : torch.Tensor
            Input feature tensor of shape (N_samples, sequence_length, num_features).
        y : torch.Tensor
            Binary label tensor of shape (N_samples,) where 0 = normal, 1 = suspicious/theft.
        meter_ids : List[str]
            Pseudonymous meter identifier corresponding to each window.
        timestamps : List[str]
            Anchor end timestamp for each sequence window.
        is_synthetic : bool
            Flag explicitly identifying that ground-truth labels are simulated/synthetic.
        feature_names : Optional[List[str]]
            List of feature names matching input dimension.
        """
        self.X = X.float()
        self.y = y.float()
        self.meter_ids = meter_ids
        self.timestamps = timestamps
        self.is_synthetic = is_synthetic
        self.feature_names = feature_names or ["norm_consumption"]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @property
    def positive_ratio(self) -> float:
        """Calculate proportion of positive (suspicious/theft) instances."""
        if len(self.y) == 0:
            return 0.0
        return float(torch.mean(self.y).item())

    @property
    def class_weights(self) -> torch.Tensor:
        """Compute recommended pos_weight for BCEWithLogitsLoss."""
        pos = int(torch.sum(self.y).item())
        neg = len(self.y) - pos
        if pos == 0:
            return torch.tensor([1.0])
        return torch.tensor([max(1.0, float(neg) / float(pos))])


def extract_multivariate_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Derives historical consumption statistics, temporal cycles,
    and operational meter status features from telemetry DataFrame.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Matrix of shape (N_records, num_features) and feature name list.
    """
    df = df.copy()

    # 1. Normalized consumption
    c_norm = df["norm_consumption"].values

    # 2. Delta consumption (rate of change: c_t - c_{t-1})
    c_delta = np.zeros_like(c_norm)
    c_delta[1:] = np.diff(c_norm)

    # 3. Rolling statistics (6-hour window)
    rolling_series = pd.Series(c_norm)
    r_mean = rolling_series.rolling(window=6, min_periods=1).mean().values
    r_std = rolling_series.rolling(window=6, min_periods=1).std().fillna(0.0).values

    # 4. Cyclical temporal features (Hour of Day & Day of Week)
    dt_series = pd.to_datetime(df["timestamp"])
    hours = dt_series.dt.hour.values
    dows = dt_series.dt.dayofweek.values

    sin_hour = np.sin(2 * np.pi * hours / 24.0)
    cos_hour = np.cos(2 * np.pi * hours / 24.0)
    sin_dow = np.sin(2 * np.pi * dows / 7.0)
    cos_dow = np.cos(2 * np.pi * dows / 7.0)

    # 5. Operational meter status encoding
    status_col = df.get("meter_status", pd.Series(["NORMAL"] * len(df)))
    status_enc = np.array([STATUS_MAPPING.get(s, 0) / 4.0 for s in status_col], dtype=np.float32)

    # 6. Event type encoding
    event_col = df.get("event_type", pd.Series(["NONE"] * len(df)))
    event_enc = np.array([EVENT_MAPPING.get(e, 0) / 7.0 for e in event_col], dtype=np.float32)

    # 7. Hardware tamper flag
    hw_flag = df.get("hardware_tamper_flag", pd.Series([0] * len(df))).values.astype(np.float32)

    features = np.column_stack([
        c_norm,
        c_delta,
        r_mean,
        r_std,
        sin_hour,
        cos_hour,
        sin_dow,
        cos_dow,
        status_enc,
        event_enc,
        hw_flag,
    ])

    feature_names = [
        "norm_consumption",
        "consumption_delta",
        "rolling_mean_6h",
        "rolling_std_6h",
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
        "meter_status_enc",
        "event_type_enc",
        "hardware_tamper_flag"
    ]
    return features.astype(np.float32), feature_names


def create_sequence_windows(
    df: pd.DataFrame,
    window_size: int = 24, # 24-hour or 48-hour sequence window
    stride: int = 12,
    multivariate: bool = True,
    is_synthetic: bool = True
) -> MeterSequenceDataset:
    """
    Constructs sliding window sequences from a preprocessed telemetry DataFrame.
    """
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meter_ids: List[str] = []
    timestamps: List[str] = []

    # Identify ground-truth label column
    label_col = "simulated_theft_label" if "simulated_theft_label" in df.columns else "FLAG"
    if label_col not in df.columns:
        label_col = "label"

    feature_names = ["norm_consumption"]

    # Group by meter_id so windows do not cross between different meters
    grouped = df.groupby("meter_id", sort=False)

    for meter_id, group in grouped:
        group = group.sort_values("timestamp")
        n_rows = len(group)
        if n_rows < window_size:
            continue

        if multivariate:
            feat_matrix, feature_names = extract_multivariate_features(group)
        else:
            feat_matrix = group["norm_consumption"].values[:, np.newaxis].astype(np.float32)
            feature_names = ["norm_consumption"]

        labels = group[label_col].values.astype(np.int64)
        ts_vals = group["timestamp"].values

        for start_idx in range(0, n_rows - window_size + 1, stride):
            end_idx = start_idx + window_size
            window = feat_matrix[start_idx:end_idx]

            # Binary label: If ANY hour in window is flagged theft, window is labeled positive (theft)
            window_label = int(np.max(labels[start_idx:end_idx]))

            X_list.append(window)
            y_list.append(window_label)
            meter_ids.append(str(meter_id))
            timestamps.append(str(ts_vals[end_idx - 1]))

    if not X_list:
        # Fallback empty dataset with proper shape
        num_feat = len(feature_names)
        return MeterSequenceDataset(
            X=torch.zeros((0, window_size, num_feat)),
            y=torch.zeros((0,)),
            meter_ids=[],
            timestamps=[],
            is_synthetic=is_synthetic,
            feature_names=feature_names
        )

    X_tensor = torch.from_numpy(np.array(X_list, dtype=np.float32))
    y_tensor = torch.from_numpy(np.array(y_list, dtype=np.float32))

    return MeterSequenceDataset(
        X=X_tensor,
        y=y_tensor,
        meter_ids=meter_ids,
        timestamps=timestamps,
        is_synthetic=is_synthetic,
        feature_names=feature_names
    )


def split_data_leakage_free(
    df: pd.DataFrame,
    split_strategy: str = "consumer", # 'consumer' or 'temporal'
    test_ratio: float = 0.20,
    val_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits telemetry data into Train, Validation, and Test partitions
    BEFORE sequence windowing to strictly eliminate data leakage.

    Strategies:
    - 'consumer': Disjoint meter subsets. Meter M_i is exclusively in train, val, OR test.
    - 'temporal': Chronological split. Train on earlier time horizon, test on future horizon.
    """
    rng = np.random.RandomState(seed)

    if split_strategy == "consumer":
        # Identify label column for stratified consumer partitioning
        label_col = None
        for col in ["simulated_theft_label", "FLAG", "label"]:
            if col in df.columns:
                label_col = col
                break

        if label_col is not None and df[label_col].nunique() > 1:
            meter_has_theft = df.groupby("meter_id")[label_col].max()
            pos_meters = meter_has_theft[meter_has_theft > 0].index.to_numpy()
            neg_meters = meter_has_theft[meter_has_theft == 0].index.to_numpy()
            rng.shuffle(pos_meters)
            rng.shuffle(neg_meters)

            def partition_meters(m_list: np.ndarray) -> Tuple[set, set, set]:
                n = len(m_list)
                n_te = int(round(n * test_ratio))
                n_va = int(round(n * val_ratio))
                n_tr = max(1 if n > 0 else 0, n - n_te - n_va)
                return set(m_list[:n_tr]), set(m_list[n_tr : n_tr + n_va]), set(m_list[n_tr + n_va :])

            pos_tr, pos_va, pos_te = partition_meters(pos_meters)
            neg_tr, neg_va, neg_te = partition_meters(neg_meters)

            train_meters = pos_tr | neg_tr
            val_meters = pos_va | neg_va
            test_meters = pos_te | neg_te
        else:
            unique_meters = df["meter_id"].unique()
            rng.shuffle(unique_meters)

            n_total = len(unique_meters)
            n_test = int(round(n_total * test_ratio))
            n_val = int(round(n_total * val_ratio))
            n_train = max(1, n_total - n_test - n_val)

            train_meters = set(unique_meters[:n_train])
            val_meters = set(unique_meters[n_train : n_train + n_val])
            test_meters = set(unique_meters[n_train + n_val :])

        train_df = df[df["meter_id"].isin(train_meters)].copy()
        val_df = df[df["meter_id"].isin(val_meters)].copy()
        test_df = df[df["meter_id"].isin(test_meters)].copy()

        logger.info(
            f"Leakage-free consumer split: {len(train_meters)} train meters, "
            f"{len(val_meters)} val meters, {len(test_meters)} test meters."
        )

    elif split_strategy == "temporal":
        # Sort chronologically by timestamp
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        unique_timestamps = df_sorted["timestamp"].unique()
        n_ts = len(unique_timestamps)

        n_test_ts = int(round(n_ts * test_ratio))
        n_val_ts = int(round(n_ts * val_ratio))
        n_train_ts = max(1, n_ts - n_test_ts - n_val_ts)

        train_ts = set(unique_timestamps[:n_train_ts])
        val_ts = set(unique_timestamps[n_train_ts : n_train_ts + n_val_ts])
        test_ts = set(unique_timestamps[n_train_ts + n_val_ts :])

        train_df = df_sorted[df_sorted["timestamp"].isin(train_ts)].copy()
        val_df = df_sorted[df_sorted["timestamp"].isin(val_ts)].copy()
        test_df = df_sorted[df_sorted["timestamp"].isin(test_ts)].copy()

        logger.info(
            f"Leakage-free temporal split: {len(train_ts)} train timestamps, "
            f"{len(val_ts)} val timestamps, {len(test_ts)} test timestamps."
        )

    else:
        raise ValueError(f"Unknown split_strategy '{split_strategy}'. Choose 'consumer' or 'temporal'.")

    return train_df, val_df, test_df


def prepare_leakage_free_datasets(
    df: pd.DataFrame,
    window_size: int = 24, # 24-hour or 48-hour sequences
    stride: int = 12,
    split_strategy: str = "consumer",
    test_ratio: float = 0.20,
    val_ratio: float = 0.10,
    scaling: str = "minmax",
    multivariate: bool = True,
    is_synthetic: bool = True,
    seed: int = 42
) -> Tuple[MeterSequenceDataset, MeterSequenceDataset, MeterSequenceDataset, TimeSeriesPreprocessor]:
    """
    End-to-end dataset pipeline:
    1. Splits raw data before windowing.
    2. Fits Preprocessor STRICTLY on training partition.
    3. Transforms train, val, test partitions using training parameters.
    4. Constructs non-leaking sequence windows.
    """
    # 1. Split BEFORE windowing
    train_df, val_df, test_df = split_data_leakage_free(
        df, split_strategy=split_strategy, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed
    )

    # 2. Fit ONLY on training data
    preprocessor = TimeSeriesPreprocessor(scaling=scaling)
    preprocessor.fit(train_df)

    # 3. Transform using training parameters
    train_proc = preprocessor.transform(train_df)
    val_proc = preprocessor.transform(val_df)
    test_proc = preprocessor.transform(test_df)

    # 4. Generate sequence windows per partition
    train_ds = create_sequence_windows(train_proc, window_size=window_size, stride=stride, multivariate=multivariate, is_synthetic=is_synthetic)
    val_ds = create_sequence_windows(val_proc, window_size=window_size, stride=stride, multivariate=multivariate, is_synthetic=is_synthetic)
    test_ds = create_sequence_windows(test_proc, window_size=window_size, stride=stride, multivariate=multivariate, is_synthetic=is_synthetic)

    logger.info(
        f"Datasets synthesized: Train={len(train_ds):,} windows, "
        f"Val={len(val_ds):,} windows, Test={len(test_ds):,} windows "
        f"(Features: {train_ds.X.shape[-1] if len(train_ds) > 0 else 0}, Window: {window_size}h)"
    )
    return train_ds, val_ds, test_ds, preprocessor


def create_dataloaders(
    train_ds: MeterSequenceDataset,
    val_ds: MeterSequenceDataset,
    test_ds: MeterSequenceDataset,
    batch_size: int = 32,
    use_weighted_sampler: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds PyTorch DataLoaders with optional WeightedRandomSampler to handle class imbalance.
    """
    train_sampler = None
    shuffle = True

    if use_weighted_sampler and len(train_ds) > 0:
        labels = train_ds.y.numpy().astype(int)
        class_counts = np.bincount(labels, minlength=2)
        total_samples = len(labels)

        # Compute sample weights inversely proportional to class frequency
        weight_per_class = total_samples / (2.0 * np.maximum(class_counts, 1))
        sample_weights = weight_per_class[labels]

        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=total_samples,
            replacement=True
        )
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
