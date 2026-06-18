import numpy as np
import pytest

from battery_rul.evaluation.metrics import (
    mae,
    overestimation_rmse,
    r2_score,
    rmse,
    safety_ratio,
    underestimation_rmse,
)


def test_rmse_known_values() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    assert rmse(y_true, y_true) == 0.0
    assert rmse(y_true, y_true + 1.0) == pytest.approx(1.0)


def test_mae_known_values() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 5.0])
    assert mae(y_true, y_pred) == pytest.approx(1.0)


def test_r2_score_perfect_fit() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y_true, y_true) == pytest.approx(1.0)


def test_overestimation_underestimation_split() -> None:
    y_true = np.array([1.0, 1.0, 1.0, 1.0])
    y_pred = np.array([1.5, 1.5, 0.5, 0.5])
    assert overestimation_rmse(y_true, y_pred) == pytest.approx(0.5)
    assert underestimation_rmse(y_true, y_pred) == pytest.approx(0.5)


def test_safety_ratio_greater_than_one_when_more_underestimates() -> None:
    y_true = np.array([1.0, 1.0, 1.0, 1.0])
    y_pred = np.array([0.9, 0.9, 0.9, 1.1])
    assert safety_ratio(y_true, y_pred) > 1.0


def test_safety_ratio_less_than_one_when_more_overestimates() -> None:
    y_true = np.array([1.0, 1.0, 1.0, 1.0])
    y_pred = np.array([1.1, 1.1, 1.1, 0.9])
    assert safety_ratio(y_true, y_pred) < 1.0
