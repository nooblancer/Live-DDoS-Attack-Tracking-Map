"""Unit tests for ml/train.py — model training and evaluation."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ml.train import (
    evaluate_model,
    select_and_save_best,
    train_random_forest,
    train_xgboost,
)


@pytest.fixture
def synthetic_data():
    """Generate a small synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=500,
        n_features=45,
        n_informative=20,
        n_classes=2,
        random_state=42,
        weights=[0.7, 0.3],
    )
    # Split into train/test
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]
    return X_train, y_train, X_test, y_test


class TestTrainRandomForest:
    """Tests for train_random_forest()."""

    def test_returns_fitted_rf_classifier(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        model = train_random_forest(X_train, y_train)
        assert isinstance(model, RandomForestClassifier)

    def test_uses_balanced_class_weight(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        model = train_random_forest(X_train, y_train)
        assert model.class_weight == "balanced"

    def test_uses_200_estimators(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        model = train_random_forest(X_train, y_train)
        assert model.n_estimators == 200

    def test_can_predict(self, synthetic_data):
        X_train, y_train, X_test, _ = synthetic_data
        model = train_random_forest(X_train, y_train)
        predictions = model.predict(X_test)
        assert predictions.shape == (100,)
        assert set(predictions).issubset({0, 1})


class TestTrainXGBoost:
    """Tests for train_xgboost()."""

    def test_returns_fitted_xgb_classifier(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        model = train_xgboost(X_train, y_train)
        assert isinstance(model, XGBClassifier)

    def test_scale_pos_weight_matches_class_ratio(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        model = train_xgboost(X_train, y_train)
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        expected = n_neg / n_pos
        # XGBoost stores this as a parameter
        assert abs(model.get_params()["scale_pos_weight"] - expected) < 1e-6

    def test_can_predict(self, synthetic_data):
        X_train, y_train, X_test, _ = synthetic_data
        model = train_xgboost(X_train, y_train)
        predictions = model.predict(X_test)
        assert predictions.shape == (100,)
        assert set(predictions).issubset({0, 1})


class TestEvaluateModel:
    """Tests for evaluate_model()."""

    def test_returns_all_required_metrics(self, synthetic_data):
        X_train, y_train, X_test, y_test = synthetic_data
        model = train_random_forest(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

    def test_metrics_are_bounded_floats(self, synthetic_data):
        X_train, y_train, X_test, y_test = synthetic_data
        model = train_random_forest(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not float"
            assert 0.0 <= value <= 1.0, f"{key}={value} out of bounds"

    def test_works_with_xgboost(self, synthetic_data):
        X_train, y_train, X_test, y_test = synthetic_data
        model = train_xgboost(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        assert all(0.0 <= v <= 1.0 for v in metrics.values())


class TestSelectAndSaveBest:
    """Tests for select_and_save_best()."""

    def test_saves_model_and_metadata(self, synthetic_data):
        X_train, y_train, X_test, y_test = synthetic_data
        rf = train_random_forest(X_train, y_train)
        xgb = train_xgboost(X_train, y_train)

        models = {"RandomForest": rf, "XGBoost": xgb}
        metrics = {
            "RandomForest": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "roc_auc": 0.92},
            "XGBoost": {"precision": 0.92, "recall": 0.88, "f1": 0.90, "roc_auc": 0.95},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            select_and_save_best(models, metrics, output_dir)

            # Check model file exists
            model_path = output_dir / "ddos_model.joblib"
            assert model_path.exists()

            # Check metadata file exists and has correct structure
            metadata_path = output_dir / "metadata.json"
            assert metadata_path.exists()

            with open(metadata_path) as f:
                metadata = json.load(f)

            assert metadata["model_type"] == "XGBoost"  # Higher F1
            assert "features" in metadata
            assert isinstance(metadata["features"], list)
            assert "training_date" in metadata
            assert "metrics" in metadata
            assert metadata["metrics"]["f1"] == 0.90

    def test_selects_highest_f1(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        rf = train_random_forest(X_train, y_train)
        xgb = train_xgboost(X_train, y_train)

        models = {"RandomForest": rf, "XGBoost": xgb}
        # RF has higher F1 in this test case
        metrics = {
            "RandomForest": {"precision": 0.95, "recall": 0.93, "f1": 0.94, "roc_auc": 0.97},
            "XGBoost": {"precision": 0.92, "recall": 0.88, "f1": 0.90, "roc_auc": 0.95},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            select_and_save_best(models, metrics, output_dir)

            metadata_path = output_dir / "metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)

            assert metadata["model_type"] == "RandomForest"
