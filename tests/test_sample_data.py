"""Tests for ml/sample_data.py - sample data generation."""

import pandas as pd

from ml.features import SELECTED_FEATURES
from ml.sample_data import generate_sample_data


def test_output_shape():
    """Generated DataFrame has 200 rows and 47 columns (46 features + Label)."""
    df = generate_sample_data()
    assert df.shape == (200, len(SELECTED_FEATURES) + 1)


def test_class_distribution():
    """Generated data has exactly 100 Benign (0) and 100 DDoS (1) rows."""
    df = generate_sample_data()
    counts = df["Label"].value_counts()
    assert counts[0] == 100
    assert counts[1] == 100


def test_column_structure():
    """Columns match SELECTED_FEATURES plus Label."""
    df = generate_sample_data()
    expected_columns = SELECTED_FEATURES + ["Label"]
    assert list(df.columns) == expected_columns


def test_no_nan_or_inf():
    """Generated data contains no NaN or infinite values."""
    df = generate_sample_data()
    features = df[SELECTED_FEATURES]
    assert not features.isnull().any().any()
    assert not features.isin([float("inf"), float("-inf")]).any().any()


def test_reproducible_with_seed():
    """Same seed produces identical output."""
    df1 = generate_sample_data(seed=123)
    df2 = generate_sample_data(seed=123)
    pd.testing.assert_frame_equal(df1, df2)
