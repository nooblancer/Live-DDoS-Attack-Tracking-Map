"""Unit tests for ml/train_multi_class.py — multi-class model training."""

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from ml.features import SELECTED_FEATURES
from ml.train_multi_class import (
    ATTACK_TYPE_LABELS,
    OUTPUT_DIR,
    RAW_LABEL_MAP,
    evaluate_multi_class,
    generate_synthetic_model,
    map_raw_labels,
    save_multi_class_artifacts,
    train_multi_class_xgboost,
)


@pytest.fixture
def synthetic_multi_class_data():
    """Generate synthetic 12-class dataset for testing."""
    rng = np.random.default_rng(42)
    n_features = len(SELECTED_FEATURES)
    n_classes = 12
    samples_per_class = 100

    X = np.zeros((n_classes * samples_per_class, n_features))
    y = np.zeros(n_classes * samples_per_class, dtype=int)

    for i in range(n_classes):
        start = i * samples_per_class
        end = start + samples_per_class
        centroid = rng.standard_normal(n_features) * (i + 1) * 0.5
        X[start:end] = centroid + rng.standard_normal((samples_per_class, n_features)) * 0.2
        y[start:end] = i

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified split: take 80 train / 20 test per class to guarantee all classes in both
    train_indices = []
    test_indices = []
    for i in range(n_classes):
        class_indices = np.where(y == i)[0]
        split_point = int(0.8 * len(class_indices))
        train_indices.extend(class_indices[:split_point])
        test_indices.extend(class_indices[split_point:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    return {
        "X_train": X_scaled[train_indices],
        "y_train": y[train_indices],
        "X_test": X_scaled[test_indices],
        "y_test": y[test_indices],
        "scaler": scaler,
    }


class TestMapRawLabels:
    """Tests for map_raw_labels()."""

    def test_maps_known_labels(self):
        labels = np.array(["Syn", "UDP", "DNS", "HTTP"])
        result = map_raw_labels(labels)
        assert result[0] == "SYN Flood"
        assert result[1] == "UDP Flood"
        assert result[2] == "DNS Amplification"
        assert result[3] == "HTTP Flood"

    def test_maps_drdos_prefixed_labels(self):
        labels = np.array(["DrDoS_SYN", "DrDoS_UDP", "DrDoS_DNS", "DrDoS_LDAP"])
        result = map_raw_labels(labels)
        assert result[0] == "SYN Flood"
        assert result[1] == "UDP Flood"
        assert result[2] == "DNS Amplification"
        assert result[3] == "LDAP"

    def test_benign_labels_return_none(self):
        labels = np.array(["Benign", "BENIGN", "benign"])
        result = map_raw_labels(labels)
        assert all(r is None for r in result)

    def test_unknown_labels_return_none(self):
        labels = np.array(["UnknownAttack", "FooBar"])
        result = map_raw_labels(labels)
        assert all(r is None for r in result)

    def test_all_12_types_reachable(self):
        """All 12 canonical labels can be reached from raw labels."""
        mapped_labels = set(RAW_LABEL_MAP.values())
        for label in ATTACK_TYPE_LABELS:
            assert label in mapped_labels, f"Label '{label}' not reachable from raw mappings"


class TestTrainMultiClassXGBoost:
    """Tests for train_multi_class_xgboost()."""

    def test_returns_fitted_xgb_classifier(self, synthetic_multi_class_data):
        data = synthetic_multi_class_data
        model = train_multi_class_xgboost(data["X_train"], data["y_train"])
        assert isinstance(model, XGBClassifier)

    def test_can_predict_12_classes(self, synthetic_multi_class_data):
        data = synthetic_multi_class_data
        model = train_multi_class_xgboost(data["X_train"], data["y_train"])
        predictions = model.predict(data["X_test"])
        # Should predict class indices 0-11
        assert set(predictions).issubset(set(range(12)))

    def test_can_predict_proba(self, synthetic_multi_class_data):
        data = synthetic_multi_class_data
        model = train_multi_class_xgboost(data["X_train"], data["y_train"])
        proba = model.predict_proba(data["X_test"])
        # Should have 12 columns
        assert proba.shape[1] == 12
        # Each row should sum to ~1.0
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


class TestEvaluateMultiClass:
    """Tests for evaluate_multi_class()."""

    def test_returns_all_metrics(self, synthetic_multi_class_data):
        data = synthetic_multi_class_data
        model = train_multi_class_xgboost(data["X_train"], data["y_train"])
        metrics = evaluate_multi_class(model, data["X_test"], data["y_test"])
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "roc_auc" in metrics

    def test_metrics_in_valid_range(self, synthetic_multi_class_data):
        data = synthetic_multi_class_data
        model = train_multi_class_xgboost(data["X_train"], data["y_train"])
        metrics = evaluate_multi_class(model, data["X_test"], data["y_test"])
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not float"
            assert 0.0 <= value <= 1.0, f"{key}={value} out of bounds"

    def test_synthetic_separable_data_high_f1(self, synthetic_multi_class_data):
        """Well-separated synthetic data should yield high F1."""
        data = synthetic_multi_class_data
        model = train_multi_class_xgboost(data["X_train"], data["y_train"])
        metrics = evaluate_multi_class(model, data["X_test"], data["y_test"])
        assert metrics["f1"] >= 0.90  # Synthetic data is well-separated


class TestSaveMultiClassArtifacts:
    """Tests for save_multi_class_artifacts()."""

    def test_saves_all_artifacts(self, synthetic_multi_class_data):
        data = synthetic_multi_class_data
        model = train_multi_class_xgboost(data["X_train"], data["y_train"])
        scaler = data["scaler"]

        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(ATTACK_TYPE_LABELS)

        metrics = {"f1": 0.97, "precision": 0.96, "recall": 0.97, "roc_auc": 0.99}

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            save_multi_class_artifacts(model, scaler, label_encoder, metrics, output_dir)

            assert (output_dir / "model.json").exists()
            assert (output_dir / "model.joblib").exists()
            assert (output_dir / "scaler.pkl").exists()
            assert (output_dir / "metadata.json").exists()

    def test_metadata_has_correct_structure(self, synthetic_multi_class_data):
        data = synthetic_multi_class_data
        model = train_multi_class_xgboost(data["X_train"], data["y_train"])
        scaler = data["scaler"]

        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(ATTACK_TYPE_LABELS)

        metrics = {"f1": 0.97, "precision": 0.96, "recall": 0.97, "roc_auc": 0.99}

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            save_multi_class_artifacts(model, scaler, label_encoder, metrics, output_dir)

            with open(output_dir / "metadata.json") as f:
                metadata = json.load(f)

            assert metadata["model_type"] == "XGBoost"
            assert metadata["features"] == SELECTED_FEATURES
            assert metadata["metrics"] == metrics
            assert "label_mapping" in metadata
            assert "training_date" in metadata

            # Verify label mapping has all 12 classes
            assert len(metadata["label_mapping"]) == 12
            for i, label in enumerate(ATTACK_TYPE_LABELS):
                assert metadata["label_mapping"][str(i)] == label

    def test_saved_model_is_loadable(self, synthetic_multi_class_data):
        data = synthetic_multi_class_data
        model = train_multi_class_xgboost(data["X_train"], data["y_train"])
        scaler = data["scaler"]

        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(ATTACK_TYPE_LABELS)

        metrics = {"f1": 0.97, "precision": 0.96, "recall": 0.97, "roc_auc": 0.99}

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            save_multi_class_artifacts(model, scaler, label_encoder, metrics, output_dir)

            # Load via joblib and verify predictions work
            loaded_model = joblib.load(output_dir / "model.joblib")
            predictions = loaded_model.predict(data["X_test"])
            assert len(predictions) == len(data["X_test"])

            loaded_scaler = joblib.load(output_dir / "scaler.pkl")
            assert hasattr(loaded_scaler, "transform")


class TestGenerateSyntheticModel:
    """Tests for generate_synthetic_model()."""

    def test_generates_all_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            generate_synthetic_model(output_dir)

            assert (output_dir / "model.json").exists()
            assert (output_dir / "model.joblib").exists()
            assert (output_dir / "scaler.pkl").exists()
            assert (output_dir / "metadata.json").exists()

    def test_generated_metadata_valid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            generate_synthetic_model(output_dir)

            with open(output_dir / "metadata.json") as f:
                metadata = json.load(f)

            assert metadata["model_type"] == "XGBoost"
            assert len(metadata["features"]) == 46
            assert len(metadata["label_mapping"]) == 12
            assert metadata["metrics"]["f1"] >= 0.95

    def test_generated_model_loadable_by_classifier(self):
        """The generated model can be loaded by MultiClassClassifier."""
        from services.multi_class_classifier import MultiClassClassifier

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            generate_synthetic_model(output_dir)

            clf = MultiClassClassifier()
            clf.load(
                model_path=str(output_dir / "model.json"),
                scaler_path=str(output_dir / "scaler.pkl"),
                metadata_path=str(output_dir / "metadata.json"),
            )

            assert clf.is_loaded is True
            assert len(clf.feature_names) == 46

            # Test prediction
            features = {name: 1.0 for name in clf.feature_names}
            result = clf.predict(features)
            assert result.attack_type != "UNKNOWN"
            assert result.confidence > 0.0


class TestAttackTypeLabels:
    """Tests for ATTACK_TYPE_LABELS constant."""

    def test_has_12_labels(self):
        assert len(ATTACK_TYPE_LABELS) == 12

    def test_labels_match_schema(self):
        """Labels match the ATTACK_TYPE_INDEX in models/schemas.py."""
        from models.schemas import ATTACK_TYPE_INDEX

        for i, label in enumerate(ATTACK_TYPE_LABELS):
            assert ATTACK_TYPE_INDEX[i] == label
