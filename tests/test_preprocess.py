"""Unit tests for ml/preprocess.py."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.preprocess import (
    UNWANTED_COLUMNS,
    clean_dataframe,
    encode_labels,
    load_csvs,
    run_preprocessing,
)


class TestLoadCsvs:
    """Tests for load_csvs function."""

    def test_loads_single_csv(self, tmp_path):
        """Should load a single CSV and strip column whitespace."""
        df = pd.DataFrame({" Col A ": [1, 2], " Col B": [3, 4]})
        df.to_csv(tmp_path / "test.csv", index=False)

        result = load_csvs(tmp_path)
        assert list(result.columns) == ["Col A", "Col B"]
        assert len(result) == 2

    def test_concatenates_multiple_csvs(self, tmp_path):
        """Should concatenate multiple CSVs from the directory."""
        df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
        df1.to_csv(tmp_path / "file1.csv", index=False)
        df2.to_csv(tmp_path / "file2.csv", index=False)

        result = load_csvs(tmp_path)
        assert len(result) == 4
        assert list(result.columns) == ["A", "B"]

    def test_raises_on_empty_directory(self, tmp_path):
        """Should raise FileNotFoundError when no CSVs exist."""
        with pytest.raises(FileNotFoundError):
            load_csvs(tmp_path)

    def test_strips_column_whitespace(self, tmp_path):
        """Should strip leading and trailing whitespace from column names."""
        df = pd.DataFrame({" Label": [1], " Flow Duration ": [2], "Normal": [3]})
        df.to_csv(tmp_path / "test.csv", index=False)

        result = load_csvs(tmp_path)
        assert "Label" in result.columns
        assert "Flow Duration" in result.columns
        assert "Normal" in result.columns


class TestCleanDataframe:
    """Tests for clean_dataframe function."""

    def test_drops_unwanted_columns(self):
        """Should drop all specified unwanted columns."""
        data = {
            "Unnamed: 0": [1, 2],
            "Flow ID": ["a", "b"],
            "Source IP": ["1.1.1.1", "2.2.2.2"],
            "Destination IP": ["3.3.3.3", "4.4.4.4"],
            "Timestamp": ["2024-01-01", "2024-01-02"],
            "SimillarHTTP": [0, 1],
            "Feature1": [10.0, 20.0],
            "Feature2": [30.0, 40.0],
        }
        df = pd.DataFrame(data)
        result = clean_dataframe(df)

        assert "Unnamed: 0" not in result.columns
        assert "Flow ID" not in result.columns
        assert "Source IP" not in result.columns
        assert "Destination IP" not in result.columns
        assert "Timestamp" not in result.columns
        assert "SimillarHTTP" not in result.columns
        assert "Feature1" in result.columns
        assert "Feature2" in result.columns

    def test_replaces_inf_and_drops_nan(self):
        """Should replace inf with NaN and drop rows containing NaN."""
        df = pd.DataFrame({
            "A": [1.0, np.inf, 3.0, 4.0],
            "B": [5.0, 6.0, -np.inf, 8.0],
            "C": [9.0, 10.0, 11.0, np.nan],
        })
        result = clean_dataframe(df)

        assert not np.any(np.isinf(result.values))
        assert not result.isna().any().any()
        # Only the first row should survive (others have inf or NaN)
        assert len(result) == 1

    def test_drops_duplicates_keeping_first(self):
        """Should remove duplicate rows, keeping the first occurrence."""
        df = pd.DataFrame({
            "A": [1, 2, 1, 3],
            "B": [4, 5, 4, 6],
        })
        result = clean_dataframe(df)

        assert len(result) == 3
        assert not result.duplicated().any()

    def test_resets_index(self):
        """Should reset index after cleaning."""
        df = pd.DataFrame({
            "A": [1.0, np.nan, 3.0],
            "B": [4.0, 5.0, 6.0],
        })
        result = clean_dataframe(df)
        assert list(result.index) == list(range(len(result)))


class TestEncodeLabels:
    """Tests for encode_labels function."""

    def test_benign_maps_to_zero(self):
        """BENIGN labels should map to 0."""
        df = pd.DataFrame({
            "Feature": [1, 2, 3],
            "Label": ["BENIGN", "BENIGN", "BENIGN"],
        })
        features, labels = encode_labels(df)
        assert all(labels == 0)

    def test_attack_maps_to_one(self):
        """Non-BENIGN labels should map to 1."""
        df = pd.DataFrame({
            "Feature": [1, 2, 3],
            "Label": ["DDoS", "DrDoS_DNS", "Syn"],
        })
        features, labels = encode_labels(df)
        assert all(labels == 1)

    def test_mixed_labels(self):
        """Mix of BENIGN and attack labels should produce correct binary encoding."""
        df = pd.DataFrame({
            "Feature": [1, 2, 3, 4],
            "Label": ["BENIGN", "DDoS", "BENIGN", "Syn"],
        })
        features, labels = encode_labels(df)
        expected = pd.Series([0, 1, 0, 1])
        pd.testing.assert_series_equal(labels, expected, check_names=False)

    def test_label_column_removed_from_features(self):
        """The label column should not be in the returned features DataFrame."""
        df = pd.DataFrame({
            "Feature1": [1, 2],
            "Feature2": [3, 4],
            "Label": ["BENIGN", "DDoS"],
        })
        features, labels = encode_labels(df)
        assert "Label" not in features.columns
        assert "Feature1" in features.columns
        assert "Feature2" in features.columns

    def test_custom_label_column(self):
        """Should work with a custom label column name."""
        df = pd.DataFrame({
            "Feature": [1, 2],
            "MyLabel": ["BENIGN", "DDoS"],
        })
        features, labels = encode_labels(df, label_col="MyLabel")
        assert labels.iloc[0] == 0
        assert labels.iloc[1] == 1


class TestRunPreprocessing:
    """Tests for run_preprocessing function."""

    def test_full_pipeline(self, tmp_path):
        """Should produce a compressed .npz file with correct arrays."""
        # Create a simple input CSV
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        output_dir = tmp_path / "processed"

        df = pd.DataFrame({
            "Unnamed: 0": [0, 1, 2, 3],
            "Flow ID": ["a", "b", "c", "d"],
            "Source IP": ["1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4"],
            "Destination IP": ["5.5.5.5", "6.6.6.6", "7.7.7.7", "8.8.8.8"],
            "Timestamp": ["t1", "t2", "t3", "t4"],
            "SimillarHTTP": [0, 0, 0, 0],
            "Feature1": [10.0, 20.0, 30.0, 40.0],
            "Feature2": [50.0, 60.0, 70.0, 80.0],
            "Label": ["BENIGN", "DDoS", "BENIGN", "Syn"],
        })
        df.to_csv(raw_dir / "test.csv", index=False)

        run_preprocessing(raw_dir, output_dir)

        # Verify output file exists and has correct structure
        npz_path = output_dir / "train_data.npz"
        assert npz_path.exists()

        data = np.load(npz_path, allow_pickle=True)
        assert "features" in data
        assert "labels" in data
        assert "feature_names" in data

        assert data["features"].shape[0] == 4
        assert data["features"].shape[1] == 2  # Feature1 and Feature2
        assert len(data["labels"]) == 4
        assert set(data["labels"]) == {0, 1}
