"""Unit tests for services/multi_class_classifier.py."""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from models.schemas import ATTACK_TYPE_INDEX, ClassificationResult, FeatureImportance
from services.multi_class_classifier import MultiClassClassifier


@pytest.fixture
def feature_names():
    """Feature names used by the test model."""
    return ["Feature_A", "Feature_B", "Feature_C", "Feature_D", "Feature_E"]


@pytest.fixture
def multi_class_artifacts(tmp_path, feature_names):
    """Create minimal 12-class model artifacts on disk."""
    rng = np.random.default_rng(42)
    n_features = len(feature_names)
    X_train = rng.standard_normal((120, n_features))
    y_train = np.tile(np.arange(12), 10)  # 12 classes, 10 samples each

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    model.fit(X_scaled, y_train)

    model_path = tmp_path / "model.joblib"
    scaler_path = tmp_path / "scaler.pkl"
    metadata_path = tmp_path / "metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_type": "XGBoost",
        "features": feature_names,
        "metrics": {"f1": 0.96, "precision": 0.95, "recall": 0.97, "roc_auc": 0.99},
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "metadata_path": str(metadata_path),
    }


@pytest.fixture
def sample_features(feature_names):
    """Return a valid feature dict for the test model."""
    rng = np.random.default_rng(123)
    return {name: float(rng.standard_normal()) for name in feature_names}


@pytest.fixture
def loaded_classifier(multi_class_artifacts):
    """Return a loaded MultiClassClassifier."""
    clf = MultiClassClassifier()
    clf.load(**multi_class_artifacts)
    return clf


class TestMultiClassClassifierLoad:
    """Tests for MultiClassClassifier.load()."""

    def test_load_success(self, multi_class_artifacts, feature_names):
        """Classifier loads successfully with valid artifacts."""
        clf = MultiClassClassifier()
        clf.load(**multi_class_artifacts)

        assert clf.is_loaded is True
        assert clf.model is not None
        assert clf.scaler is not None
        assert clf.feature_names == feature_names

    def test_load_missing_model_file(self, multi_class_artifacts, tmp_path):
        """Enters degraded mode when model file is missing."""
        clf = MultiClassClassifier()
        clf.load(
            model_path=str(tmp_path / "nonexistent.joblib"),
            scaler_path=multi_class_artifacts["scaler_path"],
            metadata_path=multi_class_artifacts["metadata_path"],
        )

        assert clf.is_loaded is False
        assert clf.model is None

    def test_load_missing_scaler_file(self, multi_class_artifacts, tmp_path):
        """Enters degraded mode when scaler file is missing."""
        clf = MultiClassClassifier()
        clf.load(
            model_path=multi_class_artifacts["model_path"],
            scaler_path=str(tmp_path / "nonexistent.pkl"),
            metadata_path=multi_class_artifacts["metadata_path"],
        )

        assert clf.is_loaded is False

    def test_load_missing_metadata_file(self, multi_class_artifacts, tmp_path):
        """Enters degraded mode when metadata file is missing."""
        clf = MultiClassClassifier()
        clf.load(
            model_path=multi_class_artifacts["model_path"],
            scaler_path=multi_class_artifacts["scaler_path"],
            metadata_path=str(tmp_path / "nonexistent.json"),
        )

        assert clf.is_loaded is False

    def test_load_corrupt_model_file(self, multi_class_artifacts, tmp_path):
        """Enters degraded mode when model file is corrupt."""
        corrupt_path = tmp_path / "corrupt.joblib"
        corrupt_path.write_text("not a valid model file")

        clf = MultiClassClassifier()
        clf.load(
            model_path=str(corrupt_path),
            scaler_path=multi_class_artifacts["scaler_path"],
            metadata_path=multi_class_artifacts["metadata_path"],
        )

        assert clf.is_loaded is False

    def test_load_corrupt_metadata_file(self, multi_class_artifacts, tmp_path):
        """Enters degraded mode when metadata is invalid JSON."""
        corrupt_path = tmp_path / "bad.json"
        corrupt_path.write_text("{invalid json")

        clf = MultiClassClassifier()
        clf.load(
            model_path=multi_class_artifacts["model_path"],
            scaler_path=multi_class_artifacts["scaler_path"],
            metadata_path=str(corrupt_path),
        )

        assert clf.is_loaded is False


class TestMultiClassClassifierPredict:
    """Tests for MultiClassClassifier.predict()."""

    def test_predict_returns_classification_result(self, loaded_classifier, sample_features):
        """predict() returns a ClassificationResult dataclass."""
        result = loaded_classifier.predict(sample_features)
        assert isinstance(result, ClassificationResult)

    def test_predict_attack_type_in_12_classes(self, loaded_classifier, sample_features):
        """predict() returns an attack_type from the 12 defined types."""
        result = loaded_classifier.predict(sample_features)
        valid_types = set(ATTACK_TYPE_INDEX.values())
        assert result.attack_type in valid_types

    def test_predict_confidence_in_range(self, loaded_classifier, sample_features):
        """predict() returns confidence in [0.0, 1.0]."""
        result = loaded_classifier.predict(sample_features)
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_classified_in_ms_positive(self, loaded_classifier, sample_features):
        """predict() returns a positive inference time."""
        result = loaded_classifier.predict(sample_features)
        assert result.classified_in_ms > 0.0

    def test_predict_top_features_count(self, loaded_classifier, sample_features):
        """predict() returns exactly 3 top features."""
        result = loaded_classifier.predict(sample_features)
        assert len(result.top_features) == 3

    def test_predict_top_features_are_valid_names(self, loaded_classifier, sample_features, feature_names):
        """predict() top features are valid feature names from the model."""
        result = loaded_classifier.predict(sample_features)
        for name in result.top_features:
            assert name in feature_names

    def test_predict_fallback_when_not_loaded(self, sample_features):
        """predict() returns fallback result when model not loaded."""
        clf = MultiClassClassifier()
        result = clf.predict(sample_features)

        assert result.attack_type == "UNKNOWN"
        assert result.confidence == 0.0
        assert result.classified_in_ms == 0.0
        assert result.top_features == []

    def test_predict_fallback_on_missing_features(self, loaded_classifier):
        """predict() returns fallback when required features are missing."""
        result = loaded_classifier.predict({"wrong_feature": 1.0})

        assert result.attack_type == "UNKNOWN"
        assert result.confidence == 0.0

    def test_predict_handles_all_zero_features(self, loaded_classifier, feature_names):
        """predict() handles all-zero feature vectors without crash."""
        zeros = {name: 0.0 for name in feature_names}
        result = loaded_classifier.predict(zeros)

        assert result.attack_type in set(ATTACK_TYPE_INDEX.values())
        assert 0.0 <= result.confidence <= 1.0


class TestMultiClassClassifierPredictBatch:
    """Tests for MultiClassClassifier.predict_batch()."""

    def test_batch_returns_correct_count(self, loaded_classifier, sample_features):
        """predict_batch() returns one result per input."""
        batch = [sample_features, sample_features, sample_features]
        results = loaded_classifier.predict_batch(batch)
        assert len(results) == 3

    def test_batch_empty_input(self, loaded_classifier):
        """predict_batch() returns empty list for empty input."""
        results = loaded_classifier.predict_batch([])
        assert results == []

    def test_batch_mixed_valid_invalid(self, loaded_classifier, sample_features):
        """predict_batch() handles mix of valid and invalid inputs."""
        batch = [
            sample_features,  # valid
            {"wrong": 1.0},  # invalid
            sample_features,  # valid
        ]
        results = loaded_classifier.predict_batch(batch)

        assert len(results) == 3
        assert results[0].attack_type != "UNKNOWN"
        assert results[1].attack_type == "UNKNOWN"  # fallback
        assert results[2].attack_type != "UNKNOWN"

    def test_batch_fallback_when_not_loaded(self, sample_features):
        """predict_batch() returns all fallbacks when model not loaded."""
        clf = MultiClassClassifier()
        results = clf.predict_batch([sample_features, sample_features])

        assert len(results) == 2
        assert all(r.attack_type == "UNKNOWN" for r in results)
        assert all(r.confidence == 0.0 for r in results)


class TestMultiClassClassifierFeatureImportances:
    """Tests for MultiClassClassifier.feature_importances property."""

    def test_importances_sorted_descending(self, loaded_classifier):
        """feature_importances returns entries sorted by importance descending."""
        imps = loaded_classifier.feature_importances
        for i in range(len(imps) - 1):
            assert imps[i].importance >= imps[i + 1].importance

    def test_importances_are_feature_importance_objects(self, loaded_classifier):
        """feature_importances returns list of FeatureImportance."""
        imps = loaded_classifier.feature_importances
        assert all(isinstance(fi, FeatureImportance) for fi in imps)

    def test_importances_match_feature_count(self, loaded_classifier, feature_names):
        """feature_importances returns one entry per model feature."""
        imps = loaded_classifier.feature_importances
        assert len(imps) == len(feature_names)

    def test_importances_empty_when_not_loaded(self):
        """feature_importances returns empty list when model not loaded."""
        clf = MultiClassClassifier()
        assert clf.feature_importances == []
