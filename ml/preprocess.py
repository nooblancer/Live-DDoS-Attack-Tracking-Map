"""Data loading and cleaning for the CIC-DDoS2019 dataset.

Provides functions to load raw CSVs, clean them (remove unwanted columns,
handle inf/NaN, drop duplicates), encode labels for binary classification,
and orchestrate the full preprocessing pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Columns to drop during cleaning
UNWANTED_COLUMNS = [
    "Unnamed: 0",
    "Flow ID",
    " Source IP",
    " Destination IP",
    " Timestamp",
    "SimillarHTTP",
]


def load_csvs(directory: Path) -> pd.DataFrame:
    """Load all CSVs from directory, strip column whitespace, concatenate.

    Parameters
    ----------
    directory : Path
        Directory containing one or more .csv files.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with whitespace-stripped column names.
    """
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unwanted columns, handle inf/NaN, drop duplicates.

    Steps:
    1. Drop unwanted columns (if present).
    2. Replace infinite values with NaN.
    3. Drop rows containing any NaN.
    4. Drop duplicate rows, keeping the first occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        Raw concatenated DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with reset index.
    """
    # Drop unwanted columns (only those that exist in the DataFrame)
    cols_to_drop = [col for col in UNWANTED_COLUMNS if col in df.columns]
    # Also check stripped versions for columns that have leading spaces in raw data
    for col in UNWANTED_COLUMNS:
        stripped = col.strip()
        if stripped in df.columns and stripped not in cols_to_drop:
            cols_to_drop.append(stripped)

    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Replace inf with NaN, then drop rows with any NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # Drop duplicate rows keeping first occurrence
    df = df.drop_duplicates(keep="first")

    return df.reset_index(drop=True)


def encode_labels(
    df: pd.DataFrame, label_col: str = "Label"
) -> tuple[pd.DataFrame, pd.Series]:
    """Map BENIGN→0, all others→1. Returns (features_df, labels_series).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame containing the label column.
    label_col : str
        Name of the label column (after whitespace stripping).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (features DataFrame without label column, binary labels Series)
    """
    # Strip whitespace from label values for robust matching
    labels = df[label_col].astype(str).str.strip()
    binary_labels = (labels != "BENIGN").astype(int)

    features = df.drop(columns=[label_col])
    return features, binary_labels


def run_preprocessing(train_dir: Path, output_dir: Path) -> None:
    """Full pipeline: load → clean → encode → save .npz.

    Parameters
    ----------
    train_dir : Path
        Directory containing raw training CSV files.
    output_dir : Path
        Directory where the compressed .npz file will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSVs from {train_dir}...")
    df = load_csvs(train_dir)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    print("Cleaning DataFrame...")
    df = clean_dataframe(df)
    print(f"  After cleaning: {len(df)} rows, {len(df.columns)} columns")

    print("Encoding labels...")
    features, labels = encode_labels(df)
    print(f"  Features shape: {features.shape}")
    print(f"  Label distribution: 0={int((labels == 0).sum())}, 1={int((labels == 1).sum())}")

    output_path = output_dir / "train_data.npz"
    print(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        features=features.values,
        labels=labels.values,
        feature_names=np.array(features.columns.tolist()),
    )
    print("Preprocessing complete.")


if __name__ == "__main__":
    from config import BASE_DIR

    train_directory = BASE_DIR / "data" / "raw" / "01-12"
    processed_directory = BASE_DIR / "data" / "processed"
    run_preprocessing(train_directory, processed_directory)
