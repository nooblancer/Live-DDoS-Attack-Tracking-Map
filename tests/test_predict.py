"""Unit tests for routes/predict.py — the /predict endpoint."""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ml.features import SELECTED_FEATURES
from models.schemas import PredictRequest
from routes.predict import router, get_model_service
from services.model_service import ModelService, _FIELD_TO_FEATURE


@pytest.fixture
def trained_model_service(tmp_path):
    """Create a ModelService with a trained model loaded."""
    rng = np.random.default_rng(42)
    n_features = len(SELECTED_FEATURES)
    X_train = rng.standard_normal((100, n_features))
    y_train = rng.integers(0, 2, size=100)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_scaled, y_train)

    model_path = tmp_path / "ddos_model.joblib"
    scaler_path = tmp_path / "scaler.joblib"
    metadata_path = tmp_path / "metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_type": "RandomForest",
        "features": SELECTED_FEATURES,
        "training_date": "2024-01-15",
        "metrics": {"precision": 0.95, "recall": 0.93, "f1": 0.94, "roc_auc": 0.97},
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    svc = ModelService()
    svc.load(str(model_path), str(scaler_path), str(metadata_path))
    return svc


@pytest.fixture
def unloaded_model_service():
    """Return a ModelService that has not loaded a model."""
    return ModelService()


@pytest.fixture
def client_with_model(trained_model_service):
    """Test client with a loaded model service."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_model_service] = lambda: trained_model_service
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_without_model(unloaded_model_service):
    """Test client with an unloaded model service (simulates model not available)."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_model_service] = lambda: unloaded_model_service
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_payload():
    """Return a valid prediction request payload with all 45 features."""
    rng = np.random.default_rng(123)
    return {field: float(rng.standard_normal()) for field in _FIELD_TO_FEATURE.keys()}


class TestPredictEndpointSuccess:
    """Tests for successful prediction responses."""

    def test_predict_returns_200(self, client_with_model, valid_payload):
        """POST /predict with valid features returns 200."""
        response = client_with_model.post("/predict", json=valid_payload)
        assert response.status_code == 200

    def test_predict_returns_valid_label(self, client_with_model, valid_payload):
        """Response contains label that is 'Benign' or 'DDoS'."""
        response = client_with_model.post("/predict", json=valid_payload)
        data = response.json()
        assert data["label"] in ("Benign", "DDoS")

    def test_predict_returns_valid_probability(self, client_with_model, valid_payload):
        """Response contains probability in [0.0, 1.0]."""
        response = client_with_model.post("/predict", json=valid_payload)
        data = response.json()
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_response_has_correct_keys(self, client_with_model, valid_payload):
        """Response JSON has exactly 'label' and 'probability' keys."""
        response = client_with_model.post("/predict", json=valid_payload)
        data = response.json()
        assert set(data.keys()) == {"label", "probability"}


class TestPredictEndpoint503:
    """Tests for 503 Service Unavailable when model is not loaded."""

    def test_returns_503_when_model_not_loaded(self, client_without_model, valid_payload):
        """POST /predict returns 503 when ModelService.is_loaded is False."""
        response = client_without_model.post("/predict", json=valid_payload)
        assert response.status_code == 503
        assert response.json()["detail"] == "Model not available"


class TestPredictEndpoint422:
    """Tests for 422 Unprocessable Entity on validation errors."""

    def test_returns_422_on_missing_field(self, client_with_model):
        """POST /predict with missing fields returns 422."""
        incomplete_payload = {"flow_duration": 1.0}
        response = client_with_model.post("/predict", json=incomplete_payload)
        assert response.status_code == 422

    def test_returns_422_on_non_numeric_field(self, client_with_model, valid_payload):
        """POST /predict with a non-numeric field value returns 422."""
        valid_payload["flow_duration"] = "not_a_number"
        response = client_with_model.post("/predict", json=valid_payload)
        assert response.status_code == 422

    def test_returns_422_on_empty_body(self, client_with_model):
        """POST /predict with empty JSON body returns 422."""
        response = client_with_model.post("/predict", json={})
        assert response.status_code == 422

    def test_returns_422_on_null_body(self, client_with_model):
        """POST /predict with null body returns 422."""
        response = client_with_model.post(
            "/predict", content="null", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
