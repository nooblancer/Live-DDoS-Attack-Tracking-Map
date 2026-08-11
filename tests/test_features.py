"""Unit tests for ml/features.py — feature selection and scaling."""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ml.features import SELECTED_FEATURES, fit_scaler, save_scaler, select_features


class TestSelectedFeatures:
    """Tests for the SELECTED_FEATURES constant."""

    def test_has_46_features(self):
        # Design lists 46 columns (the spec says "45" but the actual list has 46)
        assert len(SELECTED_FEATURES) == 46

    def test_no_duplicates(self):
        assert len(SELECTED_FEATURES) == len(set(SELECTED_FEATURES))

    def test_no_leading_or_trailing_whitespace(self):
        """After preprocessing, column names are stripped."""
        for feat in SELECTED_FEATURES:
            assert feat == feat.strip(), f"Feature '{feat}' has extra whitespace"


class TestSelectFeatures:
    """Tests for select_features()."""

    def test_returns_only_selected_columns(self):
        """DataFrame with extra columns gets filtered to SELECTED_FEATURES."""
        # Build a DataFrame with all selected features + some extras
        all_cols = SELECTED_FEATURES + ["Extra1", "Extra2", "Label"]
        data = np.random.randn(10, len(all_cols))
        df = pd.DataFrame(data, columns=all_cols)

        result = select_features(df)

        assert list(result.columns) == SELECTED_FEATURES
        assert result.shape == (10, 46)

    def test_preserves_data_values(self):
        """Values in selected columns are unchanged."""
        data = np.random.randn(5, len(SELECTED_FEATURES))
        df = pd.DataFrame(data, columns=SELECTED_FEATURES)

        result = select_features(df)

        pd.testing.assert_frame_equal(result, df)

    def test_raises_on_missing_columns(self):
        """KeyError raised when required features are missing."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

        with pytest.raises(KeyError, match="Missing required feature columns"):
            select_features(df)

    def test_raises_on_partially_missing_columns(self):
        """KeyError raised when only some features are present."""
        partial_cols = SELECTED_FEATURES[:10]
        data = np.random.randn(5, len(partial_cols))
        df = pd.DataFrame(data, columns=partial_cols)

        with pytest.raises(KeyError):
            select_features(df)


class TestFitScaler:
    """Tests for fit_scaler()."""

    def test_returns_fitted_standard_scaler(self):
        X = np.random.randn(100, 5)
        scaler = fit_scaler(X)

        assert isinstance(scaler, StandardScaler)
        # Scaler should have mean_ and scale_ attributes after fitting
        assert hasattr(scaler, "mean_")
        assert hasattr(scaler, "scale_")

    def test_scaler_produces_zero_mean(self):
        X = np.random.randn(200, 3) * 10 + 5  # non-zero mean, non-unit variance
        scaler = fit_scaler(X)
        X_scaled = scaler.transform(X)

        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)

    def test_scaler_produces_unit_variance(self):
        X = np.random.randn(200, 3) * 10 + 5
        scaler = fit_scaler(X)
        X_scaled = scaler.transform(X)

        np.testing.assert_allclose(X_scaled.std(axis=0, ddof=0), 1.0, atol=1e-10)


class TestSaveScaler:
    """Tests for save_scaler()."""

    def test_saves_scaler_to_disk(self):
        X = np.random.randn(50, 5)
        scaler = fit_scaler(X)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "scaler.joblib"
            save_scaler(scaler, path)

            assert path.exists()

    def test_saved_scaler_can_be_loaded(self):
        X = np.random.randn(50, 5)
        scaler = fit_scaler(X)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "scaler.joblib"
            save_scaler(scaler, path)

            loaded = joblib.load(path)
            assert isinstance(loaded, StandardScaler)
            np.testing.assert_array_equal(loaded.mean_, scaler.mean_)
            np.testing.assert_array_equal(loaded.scale_, scaler.scale_)

    def test_creates_parent_directories(self):
        X = np.random.randn(50, 5)
        scaler = fit_scaler(X)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nested" / "dir" / "scaler.joblib"
            save_scaler(scaler, path)

            assert path.exists()
