"""Model training and evaluation CLI for DDoS detection.

Trains Random Forest and XGBoost classifiers on the CIC-DDoS2019 dataset,
evaluates both on the held-out test day, selects the best by F1-score,
and persists the model, scaler, and metadata to disk.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomForestClassifier:
    """Train Random Forest with class_weight='balanced', n_estimators=200.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape (n_samples, n_features).
    y_train : np.ndarray
        Training label vector of shape (n_samples,).

    Returns
    -------
    RandomForestClassifier
        Fitted Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """Train XGBoost with scale_pos_weight calculated from class ratio.

    The scale_pos_weight is set to the ratio of negative samples to positive
    samples (count_negative / count_positive), which helps handle class
    imbalance.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape (n_samples, n_features).
    y_train : np.ndarray
        Training label vector of shape (n_samples,).

    Returns
    -------
    XGBClassifier
        Fitted XGBoost classifier.
    """
    n_negative = int((y_train == 0).sum())
    n_positive = int((y_train == 1).sum())
    scale_pos_weight = n_negative / max(n_positive, 1)

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=200,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a trained model on test data.

    Computes precision, recall, F1-score, and ROC-AUC.

    Parameters
    ----------
    model : classifier
        A fitted sklearn-compatible classifier with predict and predict_proba.
    X_test : np.ndarray
        Test feature matrix of shape (n_samples, n_features).
    y_test : np.ndarray
        Test label vector of shape (n_samples,).

    Returns
    -------
    dict
        Dictionary with keys: precision, recall, f1, roc_auc.
        All values are floats in [0.0, 1.0].
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def select_and_save_best(
    models: dict, metrics: dict, output_dir: Path
) -> None:
    """Select the model with the highest F1-score and save artifacts.

    Saves the best model as a .joblib file and a metadata.json file
    containing the model type, feature list, training date, and metrics.

    Parameters
    ----------
    models : dict
        Mapping of model name (str) to fitted classifier instance.
        e.g. {"RandomForest": rf_model, "XGBoost": xgb_model}
    metrics : dict
        Mapping of model name (str) to evaluation metrics dict.
        e.g. {"RandomForest": {"f1": 0.95, ...}, "XGBoost": {"f1": 0.97, ...}}
    output_dir : Path
        Directory where model.joblib and metadata.json will be saved.
    """
    from ml.features import SELECTED_FEATURES

    output_dir.mkdir(parents=True, exist_ok=True)

    # Select model with highest F1
    best_name = max(metrics, key=lambda name: metrics[name]["f1"])
    best_model = models[best_name]
    best_metrics = metrics[best_name]

    # Save model
    model_path = output_dir / "ddos_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved best model ({best_name}) to {model_path}")

    # Save metadata matching ModelMetadata schema
    metadata = {
        "model_type": best_name,
        "features": SELECTED_FEATURES,
        "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics": best_metrics,
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


def main() -> None:
    """CLI entry point: preprocess → train → evaluate → save.

    Pipeline steps:
    1. Load and preprocess both training (01-12) and test (03-11) datasets
    2. Apply feature selection (SELECTED_FEATURES)
    3. Fit scaler on training data, transform both train and test
    4. Train both Random Forest and XGBoost
    5. Evaluate both on test data
    6. Select best by F1 and save model + scaler + metadata
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from config import BASE_DIR, MODEL_PATH, SCALER_PATH
    from ml.features import SELECTED_FEATURES, fit_scaler, save_scaler
    from ml.preprocess import clean_dataframe, encode_labels, load_csvs

    train_dir = BASE_DIR / "data" / "raw" / "01-12"
    test_dir = BASE_DIR / "data" / "raw" / "03-11"
    output_dir = Path(MODEL_PATH).parent

    # --- Load and preprocess training data ---
    print("=" * 60)
    print("STEP 1: Loading and preprocessing training data (01-12)")
    print("=" * 60)
    train_df = load_csvs(train_dir)
    print(f"  Loaded {len(train_df)} rows, {len(train_df.columns)} columns")

    train_df = clean_dataframe(train_df)
    print(f"  After cleaning: {len(train_df)} rows")

    train_features, train_labels = encode_labels(train_df)
    print(f"  Labels: 0={int((train_labels == 0).sum())}, 1={int((train_labels == 1).sum())}")

    # --- Load and preprocess test data ---
    print("\n" + "=" * 60)
    print("STEP 2: Loading and preprocessing test data (03-11)")
    print("=" * 60)
    test_df = load_csvs(test_dir)
    print(f"  Loaded {len(test_df)} rows, {len(test_df.columns)} columns")

    test_df = clean_dataframe(test_df)
    print(f"  After cleaning: {len(test_df)} rows")

    test_features, test_labels = encode_labels(test_df)
    print(f"  Labels: 0={int((test_labels == 0).sum())}, 1={int((test_labels == 1).sum())}")

    # --- Feature selection ---
    print("\n" + "=" * 60)
    print("STEP 3: Feature selection and scaling")
    print("=" * 60)

    # Select the 45 predefined features
    missing_train = [c for c in SELECTED_FEATURES if c not in train_features.columns]
    if missing_train:
        raise KeyError(f"Training data missing features: {missing_train}")

    missing_test = [c for c in SELECTED_FEATURES if c not in test_features.columns]
    if missing_test:
        raise KeyError(f"Test data missing features: {missing_test}")

    X_train = train_features[SELECTED_FEATURES].values
    X_test = test_features[SELECTED_FEATURES].values
    y_train = train_labels.values
    y_test = test_labels.values

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")

    # Fit scaler on training data and transform both
    scaler = fit_scaler(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler
    scaler_path = Path(SCALER_PATH)
    save_scaler(scaler, scaler_path)
    print(f"  Saved scaler to {scaler_path}")

    # --- Train models ---
    print("\n" + "=" * 60)
    print("STEP 4: Training models")
    print("=" * 60)

    print("  Training Random Forest (n_estimators=200, balanced)...")
    rf_model = train_random_forest(X_train, y_train)
    print("  Random Forest training complete.")

    print("  Training XGBoost (scale_pos_weight from class ratio)...")
    xgb_model = train_xgboost(X_train, y_train)
    print("  XGBoost training complete.")

    models = {"RandomForest": rf_model, "XGBoost": xgb_model}

    # --- Evaluate models ---
    print("\n" + "=" * 60)
    print("STEP 5: Evaluating models on test data")
    print("=" * 60)

    metrics = {}
    for name, model in models.items():
        m = evaluate_model(model, X_test, y_test)
        metrics[name] = m
        print(f"  {name}:")
        print(f"    Precision: {m['precision']:.4f}")
        print(f"    Recall:    {m['recall']:.4f}")
        print(f"    F1-score:  {m['f1']:.4f}")
        print(f"    ROC-AUC:   {m['roc_auc']:.4f}")

    # --- Select best and save ---
    print("\n" + "=" * 60)
    print("STEP 6: Selecting best model and saving artifacts")
    print("=" * 60)

    select_and_save_best(models, metrics, output_dir)
    print("\nTraining pipeline complete!")


if __name__ == "__main__":
    main()
