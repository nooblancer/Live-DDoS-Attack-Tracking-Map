"""Unit tests for services/model_service.py."""

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ml.features import SELECTED_FEATURES
from services.model_service import ModelService, _FIELD_TO_FEATURE, _FEATURE_TO_FIELD


@pytest.fixture
def trained_artifacts(tmp_path):
    """Create a minimal trained model, scaler, and metadata on disk."""
    # Train a simple RF on random data
    rng = np.random.default_rng(42)
    n_features = len(SELECTED_FEATURES)
    X_train = rng.standard_normal((100, n_features))
    y_train = rng.integers(0, 2, size=100)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_scaled, y_train)

    # Save artifacts
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

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "metadata_path": str(metadata_path),
    }


@pytest.fixture
def sample_features():
    """Return a sample feature dict with all 45 snake_case fields."""
    rng = np.random.default_rng(123)
    return {field: float(rng.standard_normal()) for field in _FIELD_TO_FEATURE.keys()}


class TestModelServiceLoad:
    """Tests for ModelService.load()."""

    def test_load_success(self, trained_artifacts):
        """Model loads successfully with valid artifacts."""
        service = ModelService()
        service.load(**trained_artifacts)

        assert service.is_loaded is True
        assert service.model is not None
        assert service.scaler is not None
        assert len(service.feature_names) == len(SELECTED_FEATURES)

    def test_load_missing_model_file(self, trained_artifacts, tmp_path):
        """is_loaded stays False when model file doesn't exist."""
        service = ModelService()
        service.load(
            model_path=str(tmp_path / "nonexistent.joblib"),
            scaler_path=trained_artifacts["scaler_path"],
            metadata_path=trained_artifacts["metadata_path"],
        )

        assert service.is_loaded is False
        assert service.model is None

    def test_load_missing_scaler_file(self, trained_artifacts, tmp_path):
        """is_loaded stays False when scaler file doesn't exist."""
        service = ModelService()
        service.load(
            model_path=trained_artifacts["model_path"],
            scaler_path=str(tmp_path / "nonexistent.joblib"),
            metadata_path=trained_artifacts["metadata_path"],
        )

        assert service.is_loaded is False

    def test_load_missing_metadata_file(self, trained_artifacts, tmp_path):
        """is_loaded stays False when metadata file doesn't exist."""
        service = ModelService()
        service.load(
            model_path=trained_artifacts["model_path"],
            scaler_path=trained_artifacts["scaler_path"],
            metadata_path=str(tmp_path / "nonexistent.json"),
        )

        assert service.is_loaded is False

    def test_load_corrupt_model_file(self, trained_artifacts, tmp_path):
        """is_loaded stays False when model file is corrupt."""
        corrupt_path = tmp_path / "corrupt_model.joblib"
        corrupt_path.write_text("this is not a valid joblib file")

        service = ModelService()
        service.load(
            model_path=str(corrupt_path),
            scaler_path=trained_artifacts["scaler_path"],
            metadata_path=trained_artifacts["metadata_path"],
        )

        assert service.is_loaded is False

    def test_load_corrupt_metadata_file(self, trained_artifacts, tmp_path):
        """is_loaded stays False when metadata file is not valid JSON."""
        corrupt_path = tmp_path / "bad_metadata.json"
        corrupt_path.write_text("{invalid json content")

        service = ModelService()
        service.load(
            model_path=trained_artifacts["model_path"],
            scaler_path=trained_artifacts["scaler_path"],
            metadata_path=str(corrupt_path),
        )

        assert service.is_loaded is False


class TestModelServicePredict:
    """Tests for ModelService.predict()."""

    def test_predict_returns_valid_label(self, trained_artifacts, sample_features):
        """predict() returns a label that is either 'Benign' or 'DDoS'."""
        service = ModelService()
        service.load(**trained_artifacts)

        result = service.predict(sample_features)

        assert result["label"] in ("Benign", "DDoS")

    def test_predict_returns_valid_probability(self, trained_artifacts, sample_features):
        """predict() returns a probability in [0.0, 1.0]."""
        service = ModelService()
        service.load(**trained_artifacts)

        result = service.predict(sample_features)

        assert 0.0 <= result["probability"] <= 1.0

    def test_predict_raises_when_not_loaded(self, sample_features):
        """predict() raises RuntimeError when model is not loaded."""
        service = ModelService()

        with pytest.raises(RuntimeError, match="Model is not loaded"):
            service.predict(sample_features)

    def test_predict_raises_on_missing_feature(self, trained_artifacts):
        """predict() raises ValueError when a required feature is missing."""
        service = ModelService()
        service.load(**trained_artifacts)

        incomplete_features = {"flow_duration": 1.0}  # Only one feature

        with pytest.raises(ValueError, match="Missing required feature"):
            service.predict(incomplete_features)


class TestFieldMapping:
    """Tests for the field name mapping constants."""

    def test_all_selected_features_have_mapping(self):
        """Every SELECTED_FEATURE has a corresponding snake_case field."""
        for feature in SELECTED_FEATURES:
            assert feature in _FEATURE_TO_FIELD, (
                f"Feature '{feature}' has no snake_case mapping"
            )

    def test_mapping_is_bijective(self):
        """The field-to-feature mapping is one-to-one."""
        assert len(_FIELD_TO_FEATURE) == len(_FEATURE_TO_FIELD)
        assert len(_FIELD_TO_FEATURE) == len(SELECTED_FEATURES)

    def test_field_names_match_predict_request(self):
        """All field names in the mapping match PredictRequest model fields."""
        from models.schemas import PredictRequest

        request_fields = set(PredictRequest.model_fields.keys())
        mapping_fields = set(_FIELD_TO_FEATURE.keys())
        assert mapping_fields == request_fields
