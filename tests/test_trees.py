from pathlib import Path

import numpy as np

from battery_rul.models.trees import BatteryLightGBM


def _synthetic_data(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, 4))
    y = x[:, 0] * 0.5 + x[:, 1] * 0.3 + rng.normal(0, 0.01, size=n)
    return x, y


def test_fit_predict_shape() -> None:
    x, y = _synthetic_data()
    model = BatteryLightGBM(n_estimators=20).fit(x, y)
    preds = model.predict(x)
    assert preds.shape == (len(x),)


def test_fit_predict_reduces_error() -> None:
    x, y = _synthetic_data()
    model = BatteryLightGBM(n_estimators=50).fit(x, y)
    preds = model.predict(x)
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    assert rmse < 0.1


def test_save_load_round_trip(tmp_path: Path) -> None:
    x, y = _synthetic_data()
    model = BatteryLightGBM(n_estimators=20).fit(x, y)
    path = tmp_path / "model.joblib"
    model.save(path)

    loaded = BatteryLightGBM.load(path)
    np.testing.assert_allclose(loaded.predict(x), model.predict(x))
