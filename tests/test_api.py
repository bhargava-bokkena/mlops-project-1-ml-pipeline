from fastapi.testclient import TestClient
from src.api.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_endpoint_returns_prediction():
    # Breast cancer dataset has 30 features → send 30 dummy values
    features = [0.0] * 30

    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], int)
