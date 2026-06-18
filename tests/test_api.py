import pytest
from fastapi.testclient import TestClient

from battery_rul.inference import api

VALID_PAYLOAD = {
    "voltage_history": [4.1, 4.08, 4.05, 4.02, 4.0],
    "current": -2.3,
    "temperature": 25.4,
    "step_type": "discharge",
}


class _FakePredictor:
    def predict(
        self, voltage_history: list[float], current: float, temperature: float, step_type: str
    ) -> dict[str, float | str]:
        return {"soh": 0.873, "rul_estimate": "Healthy", "confidence": "high"}


@pytest.fixture
def client() -> TestClient:
    api.app.dependency_overrides[api.get_predictor] = lambda: _FakePredictor()
    with TestClient(api.app) as test_client:
        yield test_client
    api.app.dependency_overrides.clear()


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_returns_expected_fields(client: TestClient) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    for key in ("architecture", "rmse", "training_battery", "test_batteries"):
        assert key in body


def test_predict_valid_input_returns_schema(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["soh"] == pytest.approx(0.873)
    assert body["soh_percent"] == pytest.approx(87.3)
    assert body["rul_estimate"] == "Healthy"
    assert body["model_version"] == "upgraded_dnn_v1"


def test_predict_invalid_step_type_returns_422(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "step_type": "sleeping"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_too_short_voltage_history_returns_422(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "voltage_history": [4.1]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
