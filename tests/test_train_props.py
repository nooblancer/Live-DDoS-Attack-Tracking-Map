"""Property-based tests for ml/train.py using Hypothesis.

Validates: Requirements 3.3, 3.4
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.tree import DecisionTreeClassifier

from ml.train import evaluate_model, select_and_save_best


# ---------- Strategies ----------

# Strategy for binary labels with at least one of each class
def binary_arrays_with_both_classes(min_size=4, max_size=100):
    """Generate two binary arrays (y_true, y_pred) each with at least one 0 and one 1."""
    return st.integers(min_value=min_size, max_value=max_size).flatmap(
        lambda n: st.tuples(
            # y_true: ensure at least one 0 and one 1
            st.lists(
                st.integers(min_value=0, max_value=1),
                min_size=n,
                max_size=n,
            ).filter(lambda arr: 0 in arr and 1 in arr),
            # y_pred: ensure at least one 0 and one 1
            st.lists(
                st.integers(min_value=0, max_value=1),
                min_size=n,
                max_size=n,
            ).filter(lambda arr: 0 in arr and 1 in arr),
        )
    )


# Strategy for distinct F1 scores (floats between 0 and 1, all different)
def distinct_f1_scores(min_models=2, max_models=6):
    """Generate a list of distinct F1 scores for model metrics."""
    return st.lists(
        st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
        min_size=min_models,
        max_size=max_models,
        unique=True,
    )


# ---------- Property 8: Model evaluation produces valid bounded metrics ----------
# Feature: ddos-attack-tracking-map, Property 8: Model evaluation produces valid bounded metrics


@settings(max_examples=100, deadline=None)
@given(data=binary_arrays_with_both_classes(min_size=10, max_size=50))
def test_evaluate_model_produces_valid_bounded_metrics(data):
    """For any binary arrays y_true and y_pred (containing at least one instance of
    each class), evaluate_model() shall return a dict containing keys precision, recall,
    f1, and roc_auc, all with float values in the range [0.0, 1.0].

    **Validates: Requirements 3.3**
    """
    y_true_list, _ = data
    y_true = np.array(y_true_list)
    n_samples = len(y_true)
    n_features = 5

    # Generate random features and train a simple model to get a valid predictor
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)

    # Train a simple decision tree so we have a valid model with predict/predict_proba
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y_true)

    # Evaluate the model on the same data (we only care about output format/bounds)
    result = evaluate_model(model, X, y_true)

    # Check all required keys exist
    required_keys = {"precision", "recall", "f1", "roc_auc"}
    assert required_keys == set(result.keys()), (
        f"Expected keys {required_keys}, got {set(result.keys())}"
    )

    # Check all values are floats in [0.0, 1.0]
    for key in required_keys:
        value = result[key]
        assert isinstance(value, float), f"{key} is not float: {type(value)}"
        assert 0.0 <= value <= 1.0, f"{key} = {value} is out of range [0.0, 1.0]"


# ---------- Property 9: Best model selection maximizes F1-score ----------
# Feature: ddos-attack-tracking-map, Property 9: Best model selection maximizes F1-score


@settings(max_examples=100, deadline=None)
@given(f1_scores=distinct_f1_scores(min_models=2, max_models=6))
def test_best_model_selection_maximizes_f1(f1_scores):
    """For any collection of 2 or more models with distinct F1 scores in their metrics,
    select_and_save_best() shall select the model whose F1 metric is strictly the
    maximum among all candidates.

    **Validates: Requirements 3.4**
    """
    # Build models dict and metrics dict with distinct F1 scores
    model_names = [f"Model_{i}" for i in range(len(f1_scores))]
    models = {}
    metrics = {}

    for name, f1 in zip(model_names, f1_scores):
        # Create a mock model that can be serialized with joblib
        mock_model = DecisionTreeClassifier(random_state=0)
        # Fit on trivial data so it's serializable
        mock_model.fit([[0], [1]], [0, 1])
        models[name] = mock_model
        metrics[name] = {
            "precision": 0.9,
            "recall": 0.85,
            "f1": f1,
            "roc_auc": 0.88,
        }

    # Determine expected best
    expected_best_name = max(model_names, key=lambda n: metrics[n]["f1"])
    expected_best_f1 = metrics[expected_best_name]["f1"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        select_and_save_best(models, metrics, output_dir)

        # Verify the metadata.json was saved with the correct model
        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists(), "metadata.json was not created"

        with open(metadata_path) as f:
            metadata = json.load(f)

        # The selected model should be the one with the highest F1
        assert metadata["model_type"] == expected_best_name, (
            f"Expected {expected_best_name} (F1={expected_best_f1}), "
            f"but got {metadata['model_type']}"
        )
        assert metadata["metrics"]["f1"] == expected_best_f1

        # Verify the model file was saved
        model_path = output_dir / "ddos_model.joblib"
        assert model_path.exists(), "ddos_model.joblib was not created"

        # Verify the saved model can be loaded back
        loaded_model = joblib.load(model_path)
        assert loaded_model is not None
