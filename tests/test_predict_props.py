"""Property-based tests for routes/predict.py — the /predict endpoint.

Uses Hypothesis to generate random valid and invalid feature vectors and verify
that the prediction endpoint enforces its contract.
"""

# Feature: ddos-attack-tracking-map, Property 10: Prediction endpoint returns valid classification
# Feature: ddos-attack-tracking-map, Property 11: Missing features produce validation error

import json

import numpy as np
import joblib
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ml.features import SELECTED_FEATURES
from routes.predict import router, get_model_service
from services.model_service import ModelService, _FIELD_TO_FEATURE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model_service(tmp_path_factory):
    """Create a ModelService with a small trained RF model for property testing."""
    tmp_path = tmp_path_factory.mktemp("model")
    rng = np.random.default_rng(42)
    n_features = len(SELECTED_FEATURES)

    # Generate random training data with both classes
    X_train = rng.standard_normal((200, n_features))
    y_train = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)

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


@pytest.fixture(scope="module")
def client(trained_model_service):
    """TestClient wired to the predict router with a loaded model."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_model_service] = lambda: trained_model_service
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# All 46 field names from the PredictRequest schema
ALL_FIELDS = list(_FIELD_TO_FEATURE.keys())

# Strategy: a valid feature dict with all 46 fields as finite floats
valid_feature_dict = st.fixed_dictionaries(
    {field: st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
     for field in ALL_FIELDS}
)

# Strategy: a non-empty subset of fields to remove (at least 1, up to all)
fields_to_remove = st.lists(
    st.sampled_from(ALL_FIELDS),
    min_size=1,
    max_size=len(ALL_FIELDS),
    unique=True,
)


# ---------------------------------------------------------------------------
# Property 10: Prediction endpoint returns valid classification
# ---------------------------------------------------------------------------

class TestProperty10:
    """
    Property 10: Prediction endpoint returns valid classification.

    For any valid feature vector (dict with all 46 required float keys), the /predict
    endpoint shall return a response with label ∈ {"Benign", "DDoS"} and
    probability ∈ [0.0, 1.0].

    **Validates: Requirements 4.2**
    """

    @given(features=valid_feature_dict)
    @settings(max_examples=100, deadline=None)
    def test_valid_features_produce_valid_classification(self, client, features):
        """Any valid feature vector yields a valid classification response."""
        response = client.post("/predict", json=features)

        assert response.status_code == 200, (
            f"Expected 200 but got {response.status_code}: {response.text}"
        )

        data = response.json()

        # Label must be one of the two valid classes
        assert data["label"] in ("Benign", "DDoS"), (
            f"Label '{data['label']}' is not in {{'Benign', 'DDoS'}}"
        )

        # Probability must be in [0.0, 1.0]
        assert 0.0 <= data["probability"] <= 1.0, (
            f"Probability {data['probability']} is not in [0.0, 1.0]"
        )


# ---------------------------------------------------------------------------
# Property 11: Missing features produce validation error
# ---------------------------------------------------------------------------

class TestProperty11:
    """
    Property 11: Missing features produce validation error.

    For any feature dict that is missing at least one of the 46 required keys,
    the /predict endpoint shall return HTTP 422 status.

    **Validates: Requirements 4.3**
    """

    @given(features=valid_feature_dict, remove=fields_to_remove)
    @settings(max_examples=100, deadline=None)
    def test_missing_fields_produce_422(self, client, features, remove):
        """Removing any subset of required fields results in 422."""
        # Remove selected fields from the valid dict
        incomplete = {k: v for k, v in features.items() if k not in remove}

        # Ensure we actually removed at least one field
        assume(len(incomplete) < len(ALL_FIELDS))

        response = client.post("/predict", json=incomplete)

        assert response.status_code == 422, (
            f"Expected 422 but got {response.status_code} "
            f"when removing fields: {remove}"
        )
