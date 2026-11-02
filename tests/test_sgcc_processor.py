"""
Unit and Integration Tests for SGCC Benchmark Dataset Processor.
Validates loading, chronological alignment, telemetry cleansing,
non-IID regional substation clustering, 30-day sequence extraction,
and client-local privacy isolation.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from mdms_simulator.config import MDMSConfig
from mdms_simulator.partitioner import ClientDataVault, RawDataPrivacyViolationError
from mdms_simulator.sgcc_processor import SGCCDatasetProcessor


@pytest.fixture
def sgcc_processor():
    raw_path = Path("data/raw/smart_meter_data(sgcc).csv")
    if not raw_path.exists():
        pytest.skip(f"SGCC dataset not found at {raw_path}")
    return SGCCDatasetProcessor(raw_csv_path=raw_path, sequence_length=30, seed=42)


def test_sgcc_dataset_detection(sgcc_processor):
    """Verify processor locates the raw SGCC CSV file."""
    path = sgcc_processor.find_raw_dataset()
    assert path.exists()
    assert "smart_meter_data" in path.name


def test_sgcc_load_clean_and_sort(sgcc_processor):
    """Verify loading, chronological date sorting, and telemetry cleaning."""
    df, date_cols = sgcc_processor.load_and_clean_data(num_meters=50)

    # 1. Row count and columns
    assert len(df) == 50
    assert "CONS_NO" in df.columns
    assert "FLAG" in df.columns

    # 2. Chronological date sequence verification
    first_date = date_cols[0]
    last_date = date_cols[-1]
    assert first_date == "2014/1/1"
    assert last_date == "2016/10/31"
    assert len(date_cols) == 1034

    # 3. Cleaned values: non-negative and zero NaNs
    vals = df[date_cols].values
    assert np.all(vals >= 0.0)
    assert not np.any(np.isnan(vals))


def test_sgcc_non_iid_substations(sgcc_processor):
    """Verify regional clustering produces non-IID demand distributions."""
    df, date_cols = sgcc_processor.load_and_clean_data(num_meters=50)
    partitions = sgcc_processor.cluster_non_iid_substations(df, date_cols)

    # Verify all 5 clients exist and total sum is 50
    assert len(partitions) == 5
    total_assigned = 0
    for cid in ["client_A", "client_B", "client_C", "client_D", "client_E"]:
        assert cid in partitions
        assert 8 <= len(partitions[cid]) <= 12
        total_assigned += len(partitions[cid])
    assert total_assigned == 50

    # Non-IID verification: client_A (industrial) has higher average load than client_E (rural)
    mean_a = partitions["client_A"]["mean_load"].mean()
    mean_e = partitions["client_E"]["mean_load"].mean()
    assert mean_a > mean_e


def test_sgcc_sliding_window_sequences(sgcc_processor):
    """Verify extraction of 30-day sliding window sequences for deep learning models."""
    df, date_cols = sgcc_processor.load_and_clean_data(num_meters=10)
    X, y, meter_ids = sgcc_processor.extract_sliding_sequences(df, date_cols[:90])

    assert X.ndim == 3
    assert X.shape[1] == 30 # sequence_length = 30 days
    assert X.shape[2] == 1  # 1 channel (daily consumption)
    assert len(X) == len(y)
    assert set(np.unique(y)).issubset({0, 1})


def test_sgcc_client_privacy_enforcement(sgcc_processor):
    """Verify SGCC raw data is isolated in client vaults and blocked from central FL server."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = MDMSConfig(base_processed_dir=tmp_path, clients_dir=tmp_path / "clients")

        processor = SGCCDatasetProcessor(
            raw_csv_path=sgcc_processor.raw_csv_path,
            config=config,
            sequence_length=30
        )
        summary = processor.process_and_partition(num_meters=25, output_dir=tmp_path / "clients")

        vault_a = ClientDataVault("client_A", tmp_path / "clients" / "client_A")
        assert vault_a.readings_csv_path.exists()

        # Local client can read raw data
        local_df = vault_a.access_raw_readings(requesting_entity="client_A")
        assert len(local_df) > 0

        # Central FL server is blocked
        with pytest.raises(RawDataPrivacyViolationError):
            vault_a.access_raw_readings(requesting_entity="FL_SERVER")
