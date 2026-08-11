"""Feature selection and scaling for the DDoS detection model.

Provides the predefined list of selected features, a function to filter
DataFrames to those features, and utilities to fit and persist a
StandardScaler for use during training and inference.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# The 45 selected feature columns (stripped names, matching output of
# preprocess.load_csvs() which applies df.columns.str.strip()).
SELECTED_FEATURES: list[str] = [
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


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the predefined feature columns from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the columns listed in SELECTED_FEATURES.

    Returns
    -------
    pd.DataFrame
        DataFrame filtered to only the 45 selected feature columns.

    Raises
    ------
    KeyError
        If any of the SELECTED_FEATURES columns are missing from the input.
    """
    missing = [col for col in SELECTED_FEATURES if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    return df[SELECTED_FEATURES]


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on training data only.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape (n_samples, n_features).

    Returns
    -------
    StandardScaler
        Fitted scaler ready for transforming train/test data.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_scaler(scaler: StandardScaler, path: Path) -> None:
    """Persist fitted scaler as .joblib.

    Parameters
    ----------
    scaler : StandardScaler
        A fitted StandardScaler instance.
    path : Path
        File path where the scaler will be saved (e.g. models/scaler.joblib).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
