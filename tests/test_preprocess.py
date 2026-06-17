import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from battery_rul.data.dataset import BatteryDataset
from battery_rul.data.preprocess import FEATURE_COLUMNS, add_engineered_features, compute_soh


@pytest.fixture
def synthetic_battery_df() -> pd.DataFrame:
    n = 200
    rng = np.random.default_rng(42)
    voltage = np.linspace(4.2, 3.0, n) + rng.normal(0, 0.001, n)
    df = pd.DataFrame(
        {
            "absolute_time": np.arange(n, dtype=np.float64) * 5.0,
            "relative_time": np.arange(n, dtype=np.float64) * 5.0,
            "voltage": voltage,
            "current": rng.uniform(-4.5, 4.5, n),
            "temperature": rng.uniform(20, 30, n),
            "step_type": rng.choice([-1, 0, 1], size=n).astype(np.int8),
            "cycle_count": np.zeros(n, dtype=np.int32),
        }
    )
    df["soh"] = compute_soh(df["voltage"])
    return add_engineered_features(df)


def test_soh_monotonic_decay_stays_in_unit_range() -> None:
    voltage = pd.Series(np.linspace(4.2, 4.2 * 0.8, 100))
    soh = compute_soh(voltage)
    assert soh.iloc[0] == pytest.approx(1.0)
    assert soh.iloc[-1] == pytest.approx(0.0)
    assert soh.between(0.0, 1.0).all()


def test_feature_columns_present(synthetic_battery_df: pd.DataFrame) -> None:
    for col in FEATURE_COLUMNS:
        assert col in synthetic_battery_df.columns
    assert "soh" in synthetic_battery_df.columns


def test_scaler_round_trip(synthetic_battery_df: pd.DataFrame) -> None:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(synthetic_battery_df[FEATURE_COLUMNS])
    restored = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(
        restored, synthetic_battery_df[FEATURE_COLUMNS].to_numpy(), rtol=1e-6, atol=1e-8
    )


def test_dataset_len_and_item_shapes(tmp_path, synthetic_battery_df: pd.DataFrame) -> None:
    window_size = 50
    parquet_path = tmp_path / "RWtest.parquet"
    synthetic_battery_df.to_parquet(parquet_path, index=False)

    dataset = BatteryDataset(parquet_path, window_size=window_size)
    assert len(dataset) == len(synthetic_battery_df) - window_size + 1

    features, target = dataset[0]
    assert features.shape == (window_size, len(FEATURE_COLUMNS))
    assert target.shape == ()
