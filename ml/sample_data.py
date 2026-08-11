"""Generate a sample dataset for development and testing.

Produces a 200-row CSV (100 Benign, 100 DDoS) with realistic-looking
network traffic features matching the preprocessed training data structure.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import SELECTED_FEATURES

# Output path relative to project root
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_data.csv"


def generate_sample_data(
    n_benign: int = 100,
    n_ddos: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a sample DataFrame with realistic network traffic features.

    Parameters
    ----------
    n_benign : int
        Number of benign (Label=0) rows to generate.
    n_ddos : int
        Number of DDoS (Label=1) rows to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with SELECTED_FEATURES columns plus a "Label" column.
    """
    rng = np.random.default_rng(seed)
    total = n_benign + n_ddos

    # --- Benign traffic patterns ---
    benign = _generate_benign(rng, n_benign)

    # --- DDoS traffic patterns ---
    ddos = _generate_ddos(rng, n_ddos)

    # Combine and add labels
    data = np.vstack([benign, ddos])
    labels = np.array([0] * n_benign + [1] * n_ddos)

    df = pd.DataFrame(data, columns=SELECTED_FEATURES)
    df["Label"] = labels

    return df


def _generate_benign(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate benign traffic feature values.

    Benign traffic tends to have:
    - Moderate flow durations
    - Lower packet counts and rates
    - Normal inter-arrival times
    - Few flag anomalies
    """
    cols = len(SELECTED_FEATURES)
    data = np.zeros((n, cols))

    col_idx = {name: i for i, name in enumerate(SELECTED_FEATURES)}

    # Flow Duration: moderate (1000 - 60_000_000 microseconds)
    data[:, col_idx["Flow Duration"]] = rng.uniform(1000, 60_000_000, n)

    # Packet counts: low to moderate
    data[:, col_idx["Total Fwd Packets"]] = rng.integers(1, 50, n).astype(float)
    data[:, col_idx["Total Backward Packets"]] = rng.integers(1, 40, n).astype(float)

    # Packet lengths
    data[:, col_idx["Total Length of Fwd Packets"]] = rng.uniform(40, 5000, n)
    data[:, col_idx["Total Length of Bwd Packets"]] = rng.uniform(40, 5000, n)
    data[:, col_idx["Fwd Packet Length Max"]] = rng.uniform(40, 1500, n)
    data[:, col_idx["Fwd Packet Length Min"]] = rng.uniform(0, 40, n)
    data[:, col_idx["Fwd Packet Length Mean"]] = rng.uniform(20, 800, n)
    data[:, col_idx["Bwd Packet Length Max"]] = rng.uniform(40, 1500, n)
    data[:, col_idx["Bwd Packet Length Min"]] = rng.uniform(0, 40, n)
    data[:, col_idx["Bwd Packet Length Mean"]] = rng.uniform(20, 800, n)

    # Flow rates: moderate
    data[:, col_idx["Flow Bytes/s"]] = rng.uniform(100, 500_000, n)
    data[:, col_idx["Flow Packets/s"]] = rng.uniform(1, 1000, n)

    # Inter-arrival times: normal distribution
    data[:, col_idx["Flow IAT Mean"]] = rng.uniform(1000, 5_000_000, n)
    data[:, col_idx["Flow IAT Std"]] = rng.uniform(0, 2_000_000, n)
    data[:, col_idx["Flow IAT Max"]] = rng.uniform(10_000, 10_000_000, n)
    data[:, col_idx["Flow IAT Min"]] = rng.uniform(0, 100_000, n)
    data[:, col_idx["Fwd IAT Total"]] = rng.uniform(1000, 60_000_000, n)
    data[:, col_idx["Fwd IAT Mean"]] = rng.uniform(1000, 5_000_000, n)
    data[:, col_idx["Bwd IAT Total"]] = rng.uniform(1000, 60_000_000, n)
    data[:, col_idx["Bwd IAT Mean"]] = rng.uniform(1000, 5_000_000, n)

    # Flags: mostly zeros for benign
    data[:, col_idx["Fwd PSH Flags"]] = rng.choice([0, 1], n, p=[0.7, 0.3])
    data[:, col_idx["Fwd URG Flags"]] = 0
    data[:, col_idx["FIN Flag Count"]] = rng.choice([0, 1], n, p=[0.6, 0.4])
    data[:, col_idx["SYN Flag Count"]] = rng.choice([0, 1], n, p=[0.5, 0.5])
    data[:, col_idx["RST Flag Count"]] = rng.choice([0, 1], n, p=[0.9, 0.1])
    data[:, col_idx["PSH Flag Count"]] = rng.choice([0, 1], n, p=[0.6, 0.4])
    data[:, col_idx["ACK Flag Count"]] = rng.choice([0, 1], n, p=[0.3, 0.7])
    data[:, col_idx["URG Flag Count"]] = 0

    # Header lengths
    data[:, col_idx["Fwd Header Length"]] = rng.integers(20, 60, n).astype(float)
    data[:, col_idx["Bwd Header Length"]] = rng.integers(20, 60, n).astype(float)

    # Packet rates
    data[:, col_idx["Fwd Packets/s"]] = rng.uniform(1, 500, n)
    data[:, col_idx["Bwd Packets/s"]] = rng.uniform(1, 500, n)

    # Packet length stats
    data[:, col_idx["Min Packet Length"]] = rng.uniform(0, 40, n)
    data[:, col_idx["Max Packet Length"]] = rng.uniform(40, 1500, n)
    data[:, col_idx["Packet Length Mean"]] = rng.uniform(20, 800, n)
    data[:, col_idx["Packet Length Std"]] = rng.uniform(0, 500, n)
    data[:, col_idx["Packet Length Variance"]] = rng.uniform(0, 250_000, n)

    # Ratios and averages
    data[:, col_idx["Down/Up Ratio"]] = rng.uniform(0, 5, n)
    data[:, col_idx["Average Packet Size"]] = rng.uniform(20, 800, n)
    data[:, col_idx["Avg Fwd Segment Size"]] = rng.uniform(20, 800, n)
    data[:, col_idx["Avg Bwd Segment Size"]] = rng.uniform(20, 800, n)

    # Window sizes
    data[:, col_idx["Init_Win_bytes_forward"]] = rng.integers(0, 65535, n).astype(float)
    data[:, col_idx["Init_Win_bytes_backward"]] = rng.integers(0, 65535, n).astype(float)

    # Active data packets and segment size
    data[:, col_idx["act_data_pkt_fwd"]] = rng.integers(0, 20, n).astype(float)
    data[:, col_idx["min_seg_size_forward"]] = rng.integers(20, 40, n).astype(float)

    return data


def _generate_ddos(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate DDoS traffic feature values.

    DDoS traffic tends to have:
    - Very short or very long flow durations
    - High packet counts and rates
    - Very low inter-arrival times (flooding)
    - Many SYN/RST flags
    """
    cols = len(SELECTED_FEATURES)
    data = np.zeros((n, cols))

    col_idx = {name: i for i, name in enumerate(SELECTED_FEATURES)}

    # Flow Duration: often very short (flooding) or very long
    data[:, col_idx["Flow Duration"]] = rng.choice(
        [rng.uniform(0, 1000, n), rng.uniform(100_000_000, 500_000_000, n)],
    ).flatten()[:n]

    # Packet counts: high
    data[:, col_idx["Total Fwd Packets"]] = rng.integers(100, 50000, n).astype(float)
    data[:, col_idx["Total Backward Packets"]] = rng.integers(0, 10, n).astype(float)

    # Packet lengths: often small uniform packets
    data[:, col_idx["Total Length of Fwd Packets"]] = rng.uniform(5000, 5_000_000, n)
    data[:, col_idx["Total Length of Bwd Packets"]] = rng.uniform(0, 500, n)
    data[:, col_idx["Fwd Packet Length Max"]] = rng.uniform(40, 200, n)
    data[:, col_idx["Fwd Packet Length Min"]] = rng.uniform(40, 100, n)
    data[:, col_idx["Fwd Packet Length Mean"]] = rng.uniform(40, 150, n)
    data[:, col_idx["Bwd Packet Length Max"]] = rng.uniform(0, 100, n)
    data[:, col_idx["Bwd Packet Length Min"]] = rng.uniform(0, 40, n)
    data[:, col_idx["Bwd Packet Length Mean"]] = rng.uniform(0, 60, n)

    # Flow rates: very high
    data[:, col_idx["Flow Bytes/s"]] = rng.uniform(500_000, 100_000_000, n)
    data[:, col_idx["Flow Packets/s"]] = rng.uniform(1000, 1_000_000, n)

    # Inter-arrival times: very low (high frequency flooding)
    data[:, col_idx["Flow IAT Mean"]] = rng.uniform(0, 1000, n)
    data[:, col_idx["Flow IAT Std"]] = rng.uniform(0, 500, n)
    data[:, col_idx["Flow IAT Max"]] = rng.uniform(0, 5000, n)
    data[:, col_idx["Flow IAT Min"]] = rng.uniform(0, 100, n)
    data[:, col_idx["Fwd IAT Total"]] = rng.uniform(0, 10_000, n)
    data[:, col_idx["Fwd IAT Mean"]] = rng.uniform(0, 1000, n)
    data[:, col_idx["Bwd IAT Total"]] = rng.uniform(0, 5000, n)
    data[:, col_idx["Bwd IAT Mean"]] = rng.uniform(0, 1000, n)

    # Flags: high SYN/RST counts (SYN floods, RST floods)
    data[:, col_idx["Fwd PSH Flags"]] = rng.choice([0, 1], n, p=[0.8, 0.2])
    data[:, col_idx["Fwd URG Flags"]] = rng.choice([0, 1], n, p=[0.95, 0.05])
    data[:, col_idx["FIN Flag Count"]] = rng.choice([0, 1], n, p=[0.8, 0.2])
    data[:, col_idx["SYN Flag Count"]] = rng.choice([0, 1], n, p=[0.2, 0.8])
    data[:, col_idx["RST Flag Count"]] = rng.choice([0, 1], n, p=[0.4, 0.6])
    data[:, col_idx["PSH Flag Count"]] = rng.choice([0, 1], n, p=[0.7, 0.3])
    data[:, col_idx["ACK Flag Count"]] = rng.choice([0, 1], n, p=[0.5, 0.5])
    data[:, col_idx["URG Flag Count"]] = rng.choice([0, 1], n, p=[0.95, 0.05])

    # Header lengths: minimal
    data[:, col_idx["Fwd Header Length"]] = rng.integers(20, 40, n).astype(float)
    data[:, col_idx["Bwd Header Length"]] = rng.integers(0, 20, n).astype(float)

    # Packet rates: very high
    data[:, col_idx["Fwd Packets/s"]] = rng.uniform(1000, 500_000, n)
    data[:, col_idx["Bwd Packets/s"]] = rng.uniform(0, 100, n)

    # Packet length stats: low variance (uniform flood packets)
    data[:, col_idx["Min Packet Length"]] = rng.uniform(40, 60, n)
    data[:, col_idx["Max Packet Length"]] = rng.uniform(60, 200, n)
    data[:, col_idx["Packet Length Mean"]] = rng.uniform(40, 150, n)
    data[:, col_idx["Packet Length Std"]] = rng.uniform(0, 50, n)
    data[:, col_idx["Packet Length Variance"]] = rng.uniform(0, 2500, n)

    # Ratios: asymmetric (mostly outbound)
    data[:, col_idx["Down/Up Ratio"]] = rng.uniform(0, 1, n)
    data[:, col_idx["Average Packet Size"]] = rng.uniform(40, 150, n)
    data[:, col_idx["Avg Fwd Segment Size"]] = rng.uniform(40, 150, n)
    data[:, col_idx["Avg Bwd Segment Size"]] = rng.uniform(0, 60, n)

    # Window sizes: often small or zero
    data[:, col_idx["Init_Win_bytes_forward"]] = rng.integers(0, 1024, n).astype(float)
    data[:, col_idx["Init_Win_bytes_backward"]] = rng.integers(0, 256, n).astype(float)

    # Active data packets
    data[:, col_idx["act_data_pkt_fwd"]] = rng.integers(0, 5, n).astype(float)
    data[:, col_idx["min_seg_size_forward"]] = rng.integers(20, 40, n).astype(float)

    return data


if __name__ == "__main__":
    df = generate_sample_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated sample data: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Label distribution:\n{df['Label'].value_counts().to_string()}")
