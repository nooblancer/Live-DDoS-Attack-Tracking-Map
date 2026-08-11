"""Property-based tests for ml/features.py using Hypothesis.

Validates: Requirements 1.6, 2.1, 2.2, 2.3
"""

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.preprocessing import StandardScaler

from ml.features import SELECTED_FEATURES, fit_scaler, save_scaler, select_features


# ---------- Strategies ----------

# Strategy for column names that are NOT in SELECTED_FEATURES
_selected_set = set(SELECTED_FEATURES)

extra_column_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_"),
    min_size=1,
    max_size=15,
).filter(lambda name: name not in _selected_set)

# Strategy for finite floats suitable for numeric data
finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)

# Strategy for floats with meaningful magnitude (avoids near-zero subnormals)
meaningful_floats = st.floats(
    min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False
)


# ---------- Property 5: ML artifact serialization round-trip ----------
# Feature: ddos-attack-tracking-map, Property 5: ML artifact serialization round-trip


@settings(max_examples=100, deadline=None)
@given(
    shape=st.tuples(
        st.integers(min_value=1, max_value=20),
        st.integers(min_value=1, max_value=10),
    ),
    data=st.data(),
)
def test_numpy_array_serialization_roundtrip(shape, data):
    """For any valid NumPy array, serializing to .npz and deserializing
    shall produce an array equivalent to the original.

    **Validates: Requirements 1.6**
    """
    n_rows, n_cols = shape
    # Generate a random numeric array
    values = data.draw(
        st.lists(
            st.lists(finite_floats, min_size=n_cols, max_size=n_cols),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    original = np.array(values)

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "test_array.npz"
        np.savez_compressed(file_path, data=original)

        # Deserialize
        loaded = np.load(file_path)["data"]

    # Verify equivalence
    np.testing.assert_array_equal(original, loaded)


@settings(max_examples=100, deadline=None)
@given(
    n_features=st.integers(min_value=1, max_value=10),
    n_samples=st.integers(min_value=2, max_value=20),
    data=st.data(),
)
def test_standard_scaler_serialization_roundtrip(n_features, n_samples, data):
    """For any fitted StandardScaler, serializing to .joblib and deserializing
    shall produce a scaler equivalent to the original.

    **Validates: Requirements 2.3**
    """
    # Generate training data
    values = data.draw(
        st.lists(
            st.lists(finite_floats, min_size=n_features, max_size=n_features),
            min_size=n_samples,
            max_size=n_samples,
        )
    )
    X = np.array(values)

    # Fit a StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "scaler.joblib"
        joblib.dump(scaler, file_path)

        # Deserialize
        loaded_scaler = joblib.load(file_path)

    # Verify equivalence: means and scales should be identical
    np.testing.assert_array_equal(scaler.mean_, loaded_scaler.mean_)
    np.testing.assert_array_equal(scaler.scale_, loaded_scaler.scale_)
    np.testing.assert_array_equal(scaler.var_, loaded_scaler.var_)

    # Verify transform produces same results
    transformed_original = scaler.transform(X)
    transformed_loaded = loaded_scaler.transform(X)
    np.testing.assert_array_equal(transformed_original, transformed_loaded)


@settings(max_examples=100, deadline=None)
@given(
    metadata=st.fixed_dictionaries(
        {
            "model_type": st.sampled_from(["RandomForest", "XGBoost"]),
            "training_date": st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P")),
                min_size=1,
                max_size=30,
            ),
            "features": st.lists(
                st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"),
                min_size=1,
                max_size=10,
            ),
        },
        optional={
            "precision": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            "recall": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            "f1": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        },
    )
)
def test_metadata_dict_serialization_roundtrip(metadata):
    """For any valid metadata dict, serializing to .json and deserializing
    shall produce a dict equivalent to the original.

    **Validates: Requirements 1.6**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "metadata.json"

        # Serialize
        with open(file_path, "w") as f:
            json.dump(metadata, f)

        # Deserialize
        with open(file_path, "r") as f:
            loaded = json.load(f)

    assert metadata == loaded


# ---------- Property 6: Feature selection returns exact predefined subset ----------
# Feature: ddos-attack-tracking-map, Property 6: Feature selection returns exact predefined subset


@settings(max_examples=100, deadline=None)
@given(
    extra_cols=st.lists(extra_column_names, min_size=0, max_size=5, unique=True),
    n_rows=st.integers(min_value=1, max_value=10),
)
def test_feature_selection_returns_exact_predefined_subset(extra_cols, n_rows):
    """For any DataFrame containing all columns in SELECTED_FEATURES plus additional
    arbitrary columns, select_features() shall return a DataFrame with exactly the
    46 predefined columns and no others.

    **Validates: Requirements 2.1**
    """
    # Build DataFrame with all selected features plus extra columns
    data = {}
    for col in SELECTED_FEATURES:
        data[col] = [float(i) for i in range(n_rows)]
    for col in extra_cols:
        data[col] = [float(i) for i in range(n_rows)]

    df = pd.DataFrame(data)

    result = select_features(df)

    # Result must have exactly the SELECTED_FEATURES columns
    assert list(result.columns) == SELECTED_FEATURES
    assert len(result.columns) == 46

    # No extra columns
    extra_in_result = set(result.columns) - set(SELECTED_FEATURES)
    assert extra_in_result == set()

    # Row count preserved
    assert len(result) == n_rows


# ---------- Property 7: StandardScaler produces zero-mean unit-variance on training data ----------
# Feature: ddos-attack-tracking-map, Property 7: StandardScaler produces zero-mean unit-variance on training data


@settings(max_examples=100, deadline=None)
@given(
    n_rows=st.integers(min_value=2, max_value=50),
    n_cols=st.integers(min_value=1, max_value=10),
    data=st.data(),
)
def test_standard_scaler_zero_mean_unit_variance(n_rows, n_cols, data):
    """For any numeric array with at least 2 rows and no constant columns,
    after fitting StandardScaler and transforming the training data, each column
    shall have mean approximately 0 and standard deviation approximately 1.

    **Validates: Requirements 2.2**
    """
    # Generate non-constant columns with meaningful variance
    columns = []
    for _ in range(n_cols):
        # Draw the first value
        first_val = data.draw(meaningful_floats)
        # Draw remaining values
        rest = data.draw(
            st.lists(meaningful_floats, min_size=n_rows - 1, max_size=n_rows - 1)
        )
        col_values = [first_val] + rest

        # Ensure column is not constant: if all values are the same,
        # offset the first value by at least 1.0
        if max(col_values) - min(col_values) < 1e-10:
            col_values[0] = col_values[0] + 1.0

        columns.append(col_values)

    X = np.array(columns).T  # shape: (n_rows, n_cols)

    # Fit scaler using the implementation function
    scaler = fit_scaler(X)

    # Transform the training data
    X_scaled = scaler.transform(X)

    # Each column should have mean ≈ 0 and std ≈ 1
    col_means = X_scaled.mean(axis=0)
    col_stds = X_scaled.std(axis=0, ddof=0)  # StandardScaler uses ddof=0

    np.testing.assert_allclose(
        col_means,
        np.zeros(n_cols),
        atol=1e-7,
        err_msg="Column means are not approximately zero after scaling",
    )
    np.testing.assert_allclose(
        col_stds,
        np.ones(n_cols),
        atol=1e-7,
        err_msg="Column standard deviations are not approximately one after scaling",
    )
