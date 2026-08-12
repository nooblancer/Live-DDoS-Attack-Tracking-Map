"""Multi-class XGBoost training for 12 DDoS attack types.

Trains an XGBoost multi-class classifier on the CIC-DDoS2019 dataset,
preserving the original 12 attack type labels rather than collapsing
to binary (Benign/DDoS). Saves model artifacts to ml/models/multi_class/.

Usage:
    python -m ml.train_multi_class           # Train on real CIC-DDoS2019 data
    python -m ml.train_multi_class --synthetic  # Generate synthetic model for testing

The existing binary model files are NOT modified — this creates a separate
model directory for multi-class artifacts.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

# Ensure project root is on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.features import SELECTED_FEATURES

# Output directory for multi-class model artifacts
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "ml" / "models" / "multi_class"

# The 12 attack type labels from CIC-DDoS2019
ATTACK_TYPE_LABELS: list[str] = [
    "SYN Flood",
    "UDP Flood",
    "DNS Amplification",
    "HTTP Flood",
    "LDAP",
    "NTP",
    "MSSQL",
    "NetBIOS",
    "SSDP",
    "TFTP",
    "UDPLag",
    "WebDDoS",
]

# Mapping from CIC-DDoS2019 raw label strings to our canonical labels.
# The dataset uses varying label formats across different day folders.
RAW_LABEL_MAP: dict[str, str] = {
    "Syn": "SYN Flood",
    "SYN": "SYN Flood",
    "syn": "SYN Flood",
    "DrDoS_SYN": "SYN Flood",
    "UDP": "UDP Flood",
    "DrDoS_UDP": "UDP Flood",
    "UDP-lag": "UDPLag",
    "UDPLag": "UDPLag",
    "DNS": "DNS Amplification",
    "DrDoS_DNS": "DNS Amplification",
    "HTTP": "HTTP Flood",
    "WebDDoS": "WebDDoS",
    "LDAP": "LDAP",
    "DrDoS_LDAP": "LDAP",
    "NTP": "NTP",
    "DrDoS_NTP": "NTP",
    "MSSQL": "MSSQL",
    "DrDoS_MSSQL": "MSSQL",
    "NetBIOS": "NetBIOS",
    "DrDoS_NetBIOS": "NetBIOS",
    "SSDP": "SSDP",
    "DrDoS_SSDP": "SSDP",
    "TFTP": "TFTP",
    "DrDoS_TFTP": "TFTP",
    "SNMP": "SSDP",  # Map SNMP to closest reflection category
    "DrDoS_SNMP": "SSDP",
}


def load_csvs_sampled(directory: Path, max_rows_per_file: int = 50000) -> pd.DataFrame:
    """Load CSVs from directory with per-file row sampling to avoid OOM.

    Reads each CSV in chunks and randomly samples up to max_rows_per_file rows
    from each file. This keeps memory usage bounded while preserving class
    distribution within each file.

    Parameters
    ----------
    directory : Path
        Directory containing .csv files.
    max_rows_per_file : int
        Maximum rows to sample from each CSV file.

    Returns
    -------
    pd.DataFrame
        Concatenated sampled DataFrame.
    """
    import pandas as pd

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    frames = []
    for csv_path in csv_files:
        print(f"    Loading {csv_path.name}...", end=" ", flush=True)
        # Read in chunks to avoid loading entire file into memory
        chunks = []
        total_rows = 0
        try:
            for chunk in pd.read_csv(csv_path, low_memory=False, chunksize=50000):
                chunk.columns = chunk.columns.str.strip()
                chunks.append(chunk)
                total_rows += len(chunk)
                # If we've read enough, stop early
                if total_rows >= max_rows_per_file * 3:
                    break
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if not chunks:
            print("empty")
            continue

        file_df = pd.concat(chunks, ignore_index=True)

        # Sample if larger than max_rows_per_file
        if len(file_df) > max_rows_per_file:
            file_df = file_df.sample(n=max_rows_per_file, random_state=42)

        print(f"{len(file_df)} rows")
        frames.append(file_df)

    if not frames:
        raise FileNotFoundError(f"No data loaded from {directory}")

    return pd.concat(frames, ignore_index=True)


def map_raw_labels(labels: np.ndarray) -> np.ndarray:
    """Map raw CIC-DDoS2019 label strings to canonical 12-type labels.

    Parameters
    ----------
    labels : np.ndarray
        Array of raw label strings from the dataset.

    Returns
    -------
    np.ndarray
        Array of canonical label strings. Rows with labels not in
        RAW_LABEL_MAP or 'Benign'/'BENIGN' are dropped (returned as None).
    """
    mapped = []
    for label in labels:
        label_str = str(label).strip()
        if label_str.lower() == "benign":
            mapped.append(None)  # Skip benign for multi-class attack model
        elif label_str in RAW_LABEL_MAP:
            mapped.append(RAW_LABEL_MAP[label_str])
        else:
            # Try case-insensitive match
            for key, val in RAW_LABEL_MAP.items():
                if key.lower() == label_str.lower():
                    mapped.append(val)
                    break
            else:
                mapped.append(None)  # Unknown label
    return np.array(mapped, dtype=object)


def train_multi_class_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int = 12,
) -> XGBClassifier:
    """Train multi-class XGBoost classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training feature matrix (n_samples, n_features).
    y_train : np.ndarray
        Integer-encoded training labels (n_samples,).
    n_classes : int
        Number of target classes.

    Returns
    -------
    XGBClassifier
        Fitted multi-class XGBoost model.
    """
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_multi_class(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int = 12,
) -> dict[str, float]:
    """Evaluate multi-class model on test data.

    Computes weighted F1, precision, recall, and OVR ROC-AUC.

    Parameters
    ----------
    model : XGBClassifier
        Fitted multi-class model.
    X_test : np.ndarray
        Scaled test feature matrix.
    y_test : np.ndarray
        Integer-encoded test labels.
    n_classes : int
        Number of classes.

    Returns
    -------
    dict[str, float]
        Metrics dictionary with keys: f1, precision, recall, roc_auc.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    # ROC-AUC with OVR (one-vs-rest) for multi-class
    try:
        metrics["roc_auc"] = float(
            roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        )
    except ValueError:
        # If a class has no samples in test set, OVR may fail
        metrics["roc_auc"] = 0.0

    return metrics


def save_multi_class_artifacts(
    model: XGBClassifier,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    metrics: dict[str, float],
    output_dir: Path,
) -> None:
    """Save multi-class model artifacts to disk.

    Saves:
    - model.json: XGBoost native model format
    - scaler.pkl: Fitted StandardScaler
    - metadata.json: Feature names, label mapping, metrics, training date

    Parameters
    ----------
    model : XGBClassifier
        Trained XGBoost multi-class model.
    scaler : StandardScaler
        Fitted scaler used during training.
    label_encoder : LabelEncoder
        Fitted label encoder for class index → label mapping.
    metrics : dict[str, float]
        Evaluation metrics from test set.
    output_dir : Path
        Directory to save artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model as joblib (preserves full sklearn wrapper with all attributes)
    # This ensures predict_proba and other sklearn methods work correctly on load.
    model_path = output_dir / "model.json"
    # Use joblib for reliable serialization across xgboost versions.
    # The .json extension is kept per the spec, but content is joblib-serialized.
    # The MultiClassClassifier.load() handles both .json (xgb native) and .joblib.
    # We save as .joblib alongside for compatibility.
    model_joblib_path = output_dir / "model.joblib"
    joblib.dump(model, model_joblib_path)
    print(f"  Saved model to {model_joblib_path}")

    # Also save native XGBoost JSON format (booster only) for portability
    model.get_booster().save_model(str(model_path))
    print(f"  Saved XGBoost native model to {model_path}")

    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler to {scaler_path}")

    # Build label mapping: index → attack type string
    label_mapping = {
        str(i): label for i, label in enumerate(label_encoder.classes_)
    }

    # Save metadata
    metadata = {
        "model_type": "XGBoost",
        "features": SELECTED_FEATURES,
        "label_mapping": label_mapping,
        "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": metrics,
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")


def generate_synthetic_model(output_dir: Path | None = None) -> None:
    """Generate a synthetic multi-class model for testing/development.

    Creates a small XGBoost model trained on synthetic data with 12 classes.
    The model won't have real-world accuracy but produces valid artifacts
    that the MultiClassClassifier service can load and use.

    Parameters
    ----------
    output_dir : Path or None
        Override output directory. Defaults to OUTPUT_DIR.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    print("=" * 60)
    print("Generating SYNTHETIC multi-class model for testing")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n_features = len(SELECTED_FEATURES)
    n_classes = 12
    samples_per_class = 200
    n_total = n_classes * samples_per_class

    # Generate synthetic feature data with class-separable patterns
    X = np.zeros((n_total, n_features))
    y = np.zeros(n_total, dtype=int)

    for class_idx in range(n_classes):
        start = class_idx * samples_per_class
        end = start + samples_per_class

        # Each class gets a unique centroid shift to make classes separable
        centroid = rng.standard_normal(n_features) * (class_idx + 1) * 0.5
        noise = rng.standard_normal((samples_per_class, n_features)) * 0.3

        X[start:end] = centroid + noise
        y[start:end] = class_idx

    # Split train/test (80/20)
    indices = rng.permutation(n_total)
    split = int(0.8 * n_total)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n  Training data: {X_train_scaled.shape[0]} samples, {n_features} features")
    print(f"  Test data: {X_test_scaled.shape[0]} samples")
    print(f"  Classes: {n_classes}")

    # Train model
    print("\n  Training XGBoost multi-class model...")
    model = train_multi_class_xgboost(X_train_scaled, y_train, n_classes)
    print("  Training complete.")

    # Evaluate
    metrics = evaluate_multi_class(model, X_test_scaled, y_test, n_classes)
    print(f"\n  Metrics on synthetic test set:")
    print(f"    Weighted F1:  {metrics['f1']:.4f}")
    print(f"    Precision:    {metrics['precision']:.4f}")
    print(f"    Recall:       {metrics['recall']:.4f}")
    print(f"    ROC-AUC:      {metrics['roc_auc']:.4f}")

    # Create label encoder with our canonical labels
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(ATTACK_TYPE_LABELS)

    # Save artifacts
    print(f"\n  Saving artifacts to {output_dir}/")
    save_multi_class_artifacts(model, scaler, label_encoder, metrics, output_dir)

    print("\n" + "=" * 60)
    print("Synthetic model generation complete!")
    print("=" * 60)


def main() -> None:
    """CLI entry point for multi-class model training.

    Supports two modes:
    - --synthetic: Generate a synthetic model for testing (no dataset required)
    - Default: Train on real CIC-DDoS2019 data from data/raw/ directories
    """
    parser = argparse.ArgumentParser(
        description="Train multi-class XGBoost for 12 DDoS attack types"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic model for testing (no real dataset needed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Override output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or OUTPUT_DIR

    if args.synthetic:
        generate_synthetic_model(output_dir)
        return

    # --- Real data training pipeline ---
    from config import BASE_DIR
    from ml.features import fit_scaler, select_features
    from ml.preprocess import clean_dataframe, load_csvs

    train_dir = BASE_DIR / "data" / "raw" / "01-12"
    test_dir = BASE_DIR / "data" / "raw" / "03-11"

    if not train_dir.exists():
        print(f"ERROR: Training data directory not found: {train_dir}")
        print("Run with --synthetic to generate a test model without real data.")
        sys.exit(1)

    if not test_dir.exists():
        print(f"ERROR: Test data directory not found: {test_dir}")
        print("Run with --synthetic to generate a test model without real data.")
        sys.exit(1)

    print("=" * 60)
    print("MULTI-CLASS TRAINING: 12 DDoS Attack Types from CIC-DDoS2019")
    print("=" * 60)

    # --- Load and preprocess training data (chunked to avoid OOM) ---
    print("\nSTEP 1: Loading training data (01-12) — sampled per file")
    print("-" * 40)
    train_df = load_csvs_sampled(train_dir, max_rows_per_file=50000)
    print(f"  Loaded {len(train_df)} rows (sampled)")

    train_df = clean_dataframe(train_df)
    print(f"  After cleaning: {len(train_df)} rows")

    # --- Map labels to 12 attack types ---
    print("\nSTEP 2: Mapping labels to 12 attack types")
    print("-" * 40)

    # Get the label column (CIC-DDoS2019 uses ' Label' with leading space sometimes)
    label_col = None
    for col in ["Label", " Label", "label"]:
        if col in train_df.columns:
            label_col = col
            break

    if label_col is None:
        print("ERROR: No 'Label' column found in training data")
        sys.exit(1)

    raw_labels = train_df[label_col].values
    mapped_labels = map_raw_labels(raw_labels)

    # Filter out benign (None) entries — multi-class model only classifies attacks
    attack_mask = mapped_labels != None  # noqa: E711
    train_df_attacks = train_df[attack_mask].copy()
    train_labels = mapped_labels[attack_mask]

    print(f"  Attack samples: {len(train_df_attacks)}")
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"    {label}: {count}")

    # --- Same for test data ---
    print("\nSTEP 3: Loading test data (03-11) — sampled per file")
    print("-" * 40)
    test_df = load_csvs_sampled(test_dir, max_rows_per_file=20000)
    test_df = clean_dataframe(test_df)
    print(f"  After cleaning: {len(test_df)} rows")

    raw_test_labels = test_df[label_col].values
    mapped_test_labels = map_raw_labels(raw_test_labels)

    test_attack_mask = mapped_test_labels != None  # noqa: E711
    test_df_attacks = test_df[test_attack_mask].copy()
    test_labels = mapped_test_labels[test_attack_mask]

    print(f"  Attack samples: {len(test_df_attacks)}")

    # --- Feature selection and scaling ---
    print("\nSTEP 4: Feature selection and scaling")
    print("-" * 40)

    X_train = select_features(train_df_attacks)[SELECTED_FEATURES].values
    X_test = select_features(test_df_attacks)[SELECTED_FEATURES].values

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")

    # Encode string labels to integers — fit on classes actually present in training
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)  # Fit on actual training labels
    n_classes = len(label_encoder.classes_)
    print(f"  Classes found in training data: {n_classes}")
    print(f"  Classes: {list(label_encoder.classes_)}")

    y_train = label_encoder.transform(train_labels)

    # For test set, only keep samples whose labels are in the training set
    test_mask = np.isin(test_labels, label_encoder.classes_)
    X_test = X_test[test_mask]
    test_labels_filtered = test_labels[test_mask]
    y_test = label_encoder.transform(test_labels_filtered)
    print(f"  Test samples after filtering to known classes: {len(y_test)}")

    # Fit and apply scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train ---
    print("\nSTEP 5: Training multi-class XGBoost")
    print("-" * 40)
    model = train_multi_class_xgboost(X_train_scaled, y_train, n_classes=n_classes)
    print("  Training complete.")

    # --- Evaluate ---
    print("\nSTEP 6: Evaluating on test set")
    print("-" * 40)
    metrics = evaluate_multi_class(model, X_test_scaled, y_test, n_classes=n_classes)

    print(f"  Weighted F1:  {metrics['f1']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")

    if metrics["f1"] < 0.95:
        print(f"\n  WARNING: F1 score {metrics['f1']:.4f} is below target 0.95!")
        print("  Consider tuning hyperparameters or checking data quality.")
    else:
        print(f"\n  ✓ F1 score {metrics['f1']:.4f} meets target ≥ 0.95")

    # --- Save artifacts ---
    print(f"\nSTEP 7: Saving artifacts to {output_dir}/")
    print("-" * 40)
    save_multi_class_artifacts(model, scaler, label_encoder, metrics, output_dir)

    print("\n" + "=" * 60)
    print("Multi-class training pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
