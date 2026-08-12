"""Property-based tests for Multi-Class Classifier output invariants.

# Feature: v2-attack-tracking-overhaul, Property 1: Classification Output Invariants

Validates: Requirements 2.1, 2.2

For any valid flow feature vector, the Multi-Class Classifier SHALL return a result
where: attack_type is one of the 12 defined labels, confidence is a float in [0.0, 1.0],
classified_in_ms is a positive float, and top_features is a list of exactly 3 strings
that are valid feature names.
"""

import json

import joblib
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from models.schemas import ATTACK_TYPE_INDEX, ClassificationResult
from services.multi_class_classifier import MultiClassClassifier

# Feature names used by the test model (same as SELECTED_FEATURES from ml/features.py)
FEATURE_NAMES: list[str] = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
]

# Valid attack type labels from the schema
VALID_ATTACK_TYPES: set[str] = set(ATTACK_TYPE_INDEX.values())


# ---------- Hypothesis Strategy ----------

# Generate a valid feature vector as a dict mapping each feature name to a finite float
feature_vector_strategy = st.fixed_dictionaries(
    {name: st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False) for name in FEATURE_NAMES}
)


# ---------- Fixtures ----------


@pytest.fixture(scope="module")
def trained_classifier(tmp_path_factory):
    """Create a MultiClassClassifier with a GradientBoosting model trained on 12 classes.

    Uses a module-scoped fixture so the model is only trained once for all property tests.
    """
    tmp_path = tmp_path_factory.mktemp("property_classifier")
    rng = np.random.default_rng(42)
    n_features = len(FEATURE_NAMES)

    # Generate training data: 240 samples, 20 per class, 12 classes
    X_train = rng.standard_normal((240, n_features))
    y_train = np.tile(np.arange(12), 20)

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train a GradientBoostingClassifier (sklearn-compatible, no XGBoost dependency)
    model = GradientBoostingClassifier(n_estimators=20, random_state=42)
    model.fit(X_scaled, y_train)

    # Save artifacts
    model_path = tmp_path / "model.joblib"
    scaler_path = tmp_path / "scaler.pkl"
    metadata_path = tmp_path / "metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_type": "XGBoost",
        "features": FEATURE_NAMES,
        "metrics": {"f1": 0.96, "precision": 0.95, "recall": 0.97, "roc_auc": 0.99},
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    # Load classifier
    clf = MultiClassClassifier()
    clf.load(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        metadata_path=str(metadata_path),
    )
    assert clf.is_loaded, "Classifier must be loaded for property tests"
    return clf


# ---------- Property Test ----------


class TestClassificationOutputInvariants:
    """Property 1: Classification Output Invariants.

    **Validates: Requirements 2.1, 2.2**

    For any valid flow feature vector, the Multi-Class Classifier SHALL return a result
    where: attack_type is one of the 12 defined labels, confidence is a float in [0.0, 1.0],
    classified_in_ms is a positive float, and top_features is a list of exactly 3 strings
    that are valid feature names.
    """

    @given(features=feature_vector_strategy)
    @settings(max_examples=200, deadline=None)
    def test_classification_output_invariants(self, features: dict[str, float], trained_classifier):
        """For any valid feature vector, predict() returns a ClassificationResult satisfying
        all output invariants.

        **Validates: Requirements 2.1, 2.2**
        """
        result = trained_classifier.predict(features)

        # Must be a ClassificationResult
        assert isinstance(result, ClassificationResult)

        # attack_type is one of the 12 defined labels
        assert result.attack_type in VALID_ATTACK_TYPES, (
            f"attack_type '{result.attack_type}' not in valid types: {VALID_ATTACK_TYPES}"
        )

        # confidence is a float in [0.0, 1.0]
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0, (
            f"confidence {result.confidence} not in [0.0, 1.0]"
        )

        # classified_in_ms is a positive float (> 0.0)
        assert isinstance(result.classified_in_ms, float)
        assert result.classified_in_ms > 0.0, (
            f"classified_in_ms {result.classified_in_ms} must be > 0.0"
        )

        # top_features is a list of exactly 3 strings that are valid feature names
        assert isinstance(result.top_features, list)
        assert len(result.top_features) == 3, (
            f"top_features has {len(result.top_features)} items, expected 3"
        )
        for feature_name in result.top_features:
            assert isinstance(feature_name, str)
            assert feature_name in FEATURE_NAMES, (
                f"top_feature '{feature_name}' not in valid feature names"
            )
