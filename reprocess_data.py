"""
Comprehensive data reprocessing script.
Re-generates all processed client partitions and partition summary JSON
from the new data/raw/data.csv (SGCC format).

Dataset format:
  CONS_NO | FLAG | 2014/1/1 | 2014/1/10 | ... (1034 daily date columns)
  FLAG=0 → Normal consumer
  FLAG=1 → Energy thief
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Fix Unicode on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
RAW_CSV = ROOT / "data" / "raw" / "data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CLIENTS_DIR = PROCESSED_DIR / "clients"
SUMMARY_JSON = PROCESSED_DIR / "sgcc_partition_summary.json"

# ─── Client profiles (Non-IID substation characteristics) ─────────────────────
CLIENT_PROFILES = {
    "client_A": {
        "label": "Heavy Industrial",
        "theft_rate_target": 0.12,   # High theft (industrial bypass)
        "n_meters": 2000,
    },
    "client_B": {
        "label": "Commercial Business Center",
        "theft_rate_target": 0.09,
        "n_meters": 2000,
    },
    "client_C": {
        "label": "High-Density Urban Residential",
        "theft_rate_target": 0.08,
        "n_meters": 2000,
    },
    "client_D": {
        "label": "Suburban Residential",
        "theft_rate_target": 0.07,
        "n_meters": 2000,
    },
    "client_E": {
        "label": "Rural / Mixed Agricultural",
        "theft_rate_target": 0.06,
        "n_meters": 2000,
    },
}

WINDOW_SIZE = 30   # 30-day sliding window per sequence
STRIDE      = 7    # Weekly stride
SEED        = 42


def load_raw_data(path: Path) -> pd.DataFrame:
    print(f"Loading raw dataset from {path}...")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df):,}   Columns: {len(df.columns):,}")
    print(f"  FLAG distribution:\n{df['FLAG'].value_counts().to_string()}")
    return df


def partition_meters(df: pd.DataFrame, rng: np.random.RandomState):
    """
    Stratified partitioning: assign meters to clients while preserving theft proportion.
    Each client gets n_meters from the full dataset (with replacement if necessary,
    since 5 × 2000 = 10,000 < 42,372 total meters).
    """
    theft_ids  = df[df["FLAG"] == 1]["CONS_NO"].values.copy()
    normal_ids = df[df["FLAG"] == 0]["CONS_NO"].values.copy()
    rng.shuffle(theft_ids)
    rng.shuffle(normal_ids)

    assignments = {}
    theft_ptr  = 0
    normal_ptr = 0

    for client_id, profile in CLIENT_PROFILES.items():
        n = profile["n_meters"]
        target_theft_rate = profile["theft_rate_target"]
        n_theft = int(round(n * target_theft_rate))
        n_normal = n - n_theft

        # Pull from shuffled arrays (wrap around if needed)
        c_theft  = np.array([theft_ids [(theft_ptr  + i) % len(theft_ids )] for i in range(n_theft )])
        c_normal = np.array([normal_ids[(normal_ptr + i) % len(normal_ids)] for i in range(n_normal)])

        theft_ptr  = (theft_ptr  + n_theft ) % len(theft_ids)
        normal_ptr = (normal_ptr + n_normal) % len(normal_ids)

        all_ids = np.concatenate([c_theft, c_normal])
        rng.shuffle(all_ids)
        assignments[client_id] = all_ids

    return assignments


def melt_to_long(df: pd.DataFrame, cons_no: str, flag: int) -> pd.DataFrame:
    """Convert wide SGCC format to long time-series format for a single meter."""
    date_cols = [c for c in df.columns if c not in ["CONS_NO", "FLAG"]]
    row = df[df["CONS_NO"] == cons_no]
    if row.empty:
        return pd.DataFrame()
    vals = row[date_cols].values.flatten().astype(float)
    dates = pd.to_datetime(date_cols, format="%Y/%m/%d")
    long_df = pd.DataFrame({"date": dates, "consumption_kwh": vals, "label": flag})
    return long_df


def build_sequences_for_client(
    df_raw: pd.DataFrame,
    meter_ids: np.ndarray,
    meter_flag_map: dict,
    window_size: int,
    stride: int,
) -> tuple:
    """
    For each meter, fill NaN values with median, normalize, then build
    sliding window sequences of shape (window_size,).
    Returns X of shape (N_seq, window_size, 1) and y of shape (N_seq,).
    """
    date_cols = [c for c in df_raw.columns if c not in ["CONS_NO", "FLAG"]]
    n_days = len(date_cols)

    X_list, y_list = [], []

    # Pre-index the raw df
    df_indexed = df_raw.set_index("CONS_NO")

    # Fit scaler on all training values for this client
    all_vals = []
    for cid in meter_ids:
        if cid in df_indexed.index:
            row_vals = df_indexed.loc[cid, date_cols].values.astype(float)
            valid = row_vals[~np.isnan(row_vals)]
            all_vals.extend(valid.tolist())

    all_vals = np.array(all_vals)
    global_median = float(np.median(all_vals)) if len(all_vals) > 0 else 0.0
    global_min    = float(np.min(all_vals))    if len(all_vals) > 0 else 0.0
    global_max    = float(np.max(all_vals))    if len(all_vals) > 0 else 1.0
    if global_max <= global_min:
        global_max = global_min + 1.0

    for cid in meter_ids:
        if cid not in df_indexed.index:
            continue
        row_vals = df_indexed.loc[cid, date_cols].values.astype(float)
        flag = meter_flag_map.get(cid, 0)

        # Impute NaN with global_median
        row_vals = np.where(np.isnan(row_vals), global_median, row_vals)

        # Min-max normalize
        row_norm = (row_vals - global_min) / (global_max - global_min)
        row_norm = np.clip(row_norm, 0.0, 1.0)

        # Sliding windows
        for start in range(0, n_days - window_size + 1, stride):
            window = row_norm[start : start + window_size].astype(np.float32)
            X_list.append(window[:, np.newaxis])   # (T, 1)
            y_list.append(float(flag))

    if not X_list:
        return np.zeros((0, window_size, 1), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    X = np.stack(X_list, axis=0)   # (N, T, 1)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def compute_mean_daily_kwh(df_raw: pd.DataFrame, meter_ids: np.ndarray) -> float:
    date_cols = [c for c in df_raw.columns if c not in ["CONS_NO", "FLAG"]]
    df_sub = df_raw[df_raw["CONS_NO"].isin(meter_ids)]
    vals = df_sub[date_cols].values.astype(float)
    vals = vals[~np.isnan(vals)]
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


def save_client_local_csv(
    df_raw: pd.DataFrame,
    meter_ids: np.ndarray,
    meter_flag_map: dict,
    client_dir: Path,
):
    """
    Save a small local_readings.csv (wide format, subset of meters for this client)
    with real consumption values from the dataset.
    """
    df_sub = df_raw[df_raw["CONS_NO"].isin(meter_ids)].copy()
    df_sub.to_csv(client_dir / "local_readings.csv", index=False)


def main():
    rng = np.random.RandomState(SEED)

    # 1. Load raw
    df_raw = load_raw_data(RAW_CSV)
    meter_flag_map = dict(zip(df_raw["CONS_NO"], df_raw["FLAG"]))

    # 2. Partition meters across clients
    assignments = partition_meters(df_raw, rng)

    # 3. Process each client
    summary_clients = {}

    for client_id, profile in CLIENT_PROFILES.items():
        print(f"\n{'='*60}")
        print(f"Processing {client_id} ({profile['label']})...")

        meter_ids = assignments[client_id]
        client_dir = CLIENTS_DIR / client_id
        client_dir.mkdir(parents=True, exist_ok=True)

        # Compute real stats
        theft_count  = sum(1 for m in meter_ids if meter_flag_map.get(m, 0) == 1)
        normal_count = len(meter_ids) - theft_count
        theft_rate   = theft_count / len(meter_ids) if len(meter_ids) > 0 else 0.0
        mean_daily   = compute_mean_daily_kwh(df_raw, meter_ids)

        print(f"  Meters: {len(meter_ids):,}  |  Theft: {theft_count} ({theft_rate*100:.1f}%)  |  Normal: {normal_count}")
        print(f"  Mean daily kWh: {mean_daily:.2f}")

        # Build sequences
        X, y = build_sequences_for_client(
            df_raw, meter_ids, meter_flag_map,
            window_size=WINDOW_SIZE, stride=STRIDE
        )
        print(f"  Sequences: X={X.shape}, y={y.shape}, theft_windows={int(y.sum())}")

        # Save tensors
        seq_path = client_dir / "sequences.npz"
        np.savez_compressed(str(seq_path), X=X, y=y)
        print(f"  Saved sequences -> {seq_path}")

        # Save local readings CSV (wide format)
        save_client_local_csv(df_raw, meter_ids, meter_flag_map, client_dir)
        print(f"  Saved local_readings.csv -> {client_dir / 'local_readings.csv'}")

        # Save vault metadata JSON
        vault_meta = {
            "client_id": client_id,
            "label": profile["label"],
            "meters_count": int(len(meter_ids)),
            "theft_meters": int(theft_count),
            "normal_meters": int(normal_count),
            "theft_rate_pct": round(theft_rate * 100, 2),
            "mean_daily_kwh": round(mean_daily, 2),
            "extracted_sequences": int(len(X)),
            "theft_sequences": int(y.sum()),
            "window_size_days": WINDOW_SIZE,
            "stride_days": STRIDE,
            "feature_dim": int(X.shape[-1]),
            "privacy_firewall": "ACTIVE",
            "central_fl_access": "BLOCKED",
            "source_dataset": "SGCC data/raw/data.csv",
        }
        with open(client_dir / "vault_metadata.json", "w") as f:
            json.dump(vault_meta, f, indent=2)
        print(f"  Saved vault_metadata.json")

        summary_clients[client_id] = {
            "client_id": client_id,
            "label": profile["label"],
            "meters_count": int(len(meter_ids)),
            "theft_meters": int(theft_count),
            "theft_rate_pct": round(theft_rate * 100, 2),
            "mean_daily_kwh": round(mean_daily, 2),
            "extracted_sequences": int(len(X)),
            "local_csv_path": f"data/processed/clients/{client_id}/local_readings.csv",
            "tensors_path": f"data/processed/clients/{client_id}/sequences.npz",
            "privacy_firewall": "ACTIVE",
            "central_fl_access": "BLOCKED",
        }

    # 4. Write overall partition summary
    total_meters    = sum(p["n_meters"] for p in CLIENT_PROFILES.values())
    total_sequences = sum(d["extracted_sequences"] for d in summary_clients.values())

    summary = {
        "dataset": "SGCC Benchmark Dataset",
        "source_path": "data/raw/data.csv",
        "total_consumers_in_raw": int(len(df_raw)),
        "total_sampled_meters": total_meters,
        "total_extracted_sequences": total_sequences,
        "sequence_length_days": WINDOW_SIZE,
        "stride_days": STRIDE,
        "raw_dataset_stats": {
            "total_consumers": int(len(df_raw)),
            "normal_consumers": int((df_raw["FLAG"] == 0).sum()),
            "theft_consumers": int((df_raw["FLAG"] == 1).sum()),
            "theft_rate_pct": round(((df_raw["FLAG"] == 1).sum() / len(df_raw)) * 100, 2),
            "date_columns": int(len(df_raw.columns) - 2),
            "date_range_start": str(df_raw.columns[2]),
            "date_range_end": str(df_raw.columns[-1]),
        },
        "clients": summary_clients,
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Partition summary saved → {SUMMARY_JSON}")
    print(f"\nDone! Total sequences across all clients: {total_sequences:,}")


if __name__ == "__main__":
    main()
