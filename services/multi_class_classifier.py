"""Multi-class XGBoost classifier for 12 DDoS attack types.

Extends the classification capabilities beyond binary (Benign/DDoS) to identify
specific attack types from the CIC-DDoS2019 dataset with confidence scores
and feature importance.
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np

from models.schemas import (
    ATTACK_TYPE_INDEX,
    ClassificationResult,
    FeatureImportance,
)

logger = logging.getLogger(__name__)

# Fallback result returned when the classifier cannot process input
_FALLBACK_RESULT = ClassificationResult(
    attack_type="UNKNOWN",
    confidence=0.0,
    classified_in_ms=0.0,
    top_features=[],
)


class MultiClassClassifier:
    """XGBoost multi-class classifier for 12 DDoS attack types."""

    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self.feature_names: list[str] = []
        self.is_loaded: bool = False
        self._metadata: dict = {}

    def load(self, model_path: str, scaler_path: str, metadata_path: str) -> None:
        """Load multi-class model artifacts from disk.

        If any file is missing or corrupt, the classifier enters degraded mode
        with is_loaded = False. Predictions will return fallback results.

        Parameters
        ----------
        model_path : str
            Path to the XGBoost model file (model.json or .joblib).
        scaler_path : str
            Path to the fitted StandardScaler (.pkl or .joblib).
        metadata_path : str
            Path to metadata.json containing feature names and label mapping.
        """
        try:
            model_file = Path(model_path)
            scaler_file = Path(scaler_path)
            metadata_file = Path(metadata_path)

            if not model_file.exists():
                logger.warning("Multi-class model file not found: %s — running in degraded mode", model_path)
                self.is_loaded = False
                return

            if not scaler_file.exists():
                logger.warning("Scaler file not found: %s — running in degraded mode", scaler_path)
                self.is_loaded = False
                return

            if not metadata_file.exists():
                logger.warning("Metadata file not found: %s — running in degraded mode", metadata_path)
                self.is_loaded = False
                return

            # Load model — support both XGBoost native and joblib formats
            # Prefer .joblib if available alongside .json (preserves sklearn wrapper)
            joblib_path = model_file.with_suffix(".joblib")
            if model_file.suffix == ".json" and joblib_path.exists():
                self.model = joblib.load(joblib_path)
            elif model_file.suffix == ".json":
                import xgboost as xgb

                self.model = xgb.XGBClassifier()
                self.model.load_model(str(model_file))
                # Ensure n_classes_ is set for predict_proba to work
                if not hasattr(self.model, "n_classes_") or self.model.n_classes_ is None:
                    self.model.n_classes_ = len(ATTACK_TYPE_INDEX)
                    self.model.classes_ = np.arange(len(ATTACK_TYPE_INDEX))
            else:
                self.model = joblib.load(model_file)

            # Load scaler
            self.scaler = joblib.load(scaler_file)

            # Load metadata
            with open(metadata_file, "r") as f:
                self._metadata = json.load(f)

            self.feature_names = self._metadata.get("features", [])
            self.is_loaded = True

            logger.info(
                "Multi-class classifier loaded: type=%s, features=%d, classes=%d",
                self._metadata.get("model_type", "XGBoost"),
                len(self.feature_names),
                len(ATTACK_TYPE_INDEX),
            )

        except Exception as exc:
            logger.error("Failed to load multi-class model artifacts: %s", exc)
            self.model = None
            self.scaler = None
            self.feature_names = []
            self._metadata = {}
            self.is_loaded = False

    def predict(self, features: dict[str, float]) -> ClassificationResult:
        """Classify a single flow into one of 12 attack types.

        Parameters
        ----------
        features : dict[str, float]
            Dictionary mapping feature names to float values.
            Keys should match the feature names in the model metadata.

        Returns
        -------
        ClassificationResult
            Contains attack_type, confidence, classified_in_ms, and top_features.
            Returns fallback result if model is not loaded or features are invalid.
        """
        if not self.is_loaded:
            return _FALLBACK_RESULT

        try:
            start_time = time.perf_counter()

            # Build feature vector in expected order
            feature_vector = self._build_feature_vector(features)
            if feature_vector is None:
                return _FALLBACK_RESULT

            # Scale features
            X = np.array(feature_vector, dtype=np.float64).reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            # Get probabilities for all classes
            probabilities = self.model.predict_proba(X_scaled)[0]

            # Select top class
            predicted_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_idx])

            # Map index to attack type label
            attack_type = ATTACK_TYPE_INDEX.get(predicted_idx, "UNKNOWN")

            # Calculate inference time
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Get top 3 feature names by importance
            top_features = self._get_top_feature_names(n=3)

            return ClassificationResult(
                attack_type=attack_type,
                confidence=confidence,
                classified_in_ms=round(elapsed_ms, 3),
                top_features=top_features,
            )

        except Exception as exc:
            logger.warning("Prediction failed: %s — returning fallback result", exc)
            return _FALLBACK_RESULT

    def predict_batch(self, feature_batch: list[dict[str, float]]) -> list[ClassificationResult]:
        """Classify a batch of flows for throughput optimization.

        Parameters
        ----------
        feature_batch : list[dict[str, float]]
            List of feature dictionaries, each mapping feature names to floats.

        Returns
        -------
        list[ClassificationResult]
            One result per input flow. Invalid entries get fallback results.
        """
        if not self.is_loaded:
            return [_FALLBACK_RESULT for _ in feature_batch]

        if not feature_batch:
            return []

        try:
            start_time = time.perf_counter()

            # Build matrix of feature vectors
            valid_indices: list[int] = []
            feature_rows: list[list[float]] = []

            for i, features in enumerate(feature_batch):
                vector = self._build_feature_vector(features)
                if vector is not None:
                    valid_indices.append(i)
                    feature_rows.append(vector)

            # Initialize results with fallbacks
            results: list[ClassificationResult] = [_FALLBACK_RESULT for _ in feature_batch]

            if not feature_rows:
                return results

            # Scale all valid features at once
            X = np.array(feature_rows, dtype=np.float64)
            X_scaled = self.scaler.transform(X)

            # Batch prediction
            probabilities = self.model.predict_proba(X_scaled)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            per_item_ms = elapsed_ms / len(feature_rows)

            # Get top features once (same for all predictions from same model)
            top_features = self._get_top_feature_names(n=3)

            # Map results back to original indices
            for batch_idx, original_idx in enumerate(valid_indices):
                probs = probabilities[batch_idx]
                predicted_idx = int(np.argmax(probs))
                confidence = float(probs[predicted_idx])
                attack_type = ATTACK_TYPE_INDEX.get(predicted_idx, "UNKNOWN")

                results[original_idx] = ClassificationResult(
                    attack_type=attack_type,
                    confidence=confidence,
                    classified_in_ms=round(per_item_ms, 3),
                    top_features=top_features,
                )

            return results

        except Exception as exc:
            logger.warning("Batch prediction failed: %s — returning fallback results", exc)
            return [_FALLBACK_RESULT for _ in feature_batch]

    @property
    def feature_importances(self) -> list[FeatureImportance]:
        """Return top features by importance from the trained model.

        Returns
        -------
        list[FeatureImportance]
            Sorted descending by importance score. Empty if model not loaded.
        """
        if not self.is_loaded or self.model is None:
            return []

        try:
            # XGBoost and sklearn-compatible models expose feature_importances_
            importances = self.model.feature_importances_
            feature_importance_pairs = [
                FeatureImportance(name=name, importance=float(imp))
                for name, imp in zip(self.feature_names, importances)
            ]
            # Sort by importance descending
            feature_importance_pairs.sort(key=lambda x: x.importance, reverse=True)
            return feature_importance_pairs

        except Exception as exc:
            logger.warning("Could not retrieve feature importances: %s", exc)
            return []

    def _build_feature_vector(self, features: dict[str, float]) -> list[float] | None:
        """Build an ordered feature vector from input dictionary.

        Returns None if feature vector cannot be constructed (missing/invalid features).
        """
        try:
            vector = []
            for feature_name in self.feature_names:
                if feature_name in features:
                    val = float(features[feature_name])
                    # Replace NaN/Inf with 0.0
                    if np.isnan(val) or np.isinf(val):
                        val = 0.0
                    vector.append(val)
                else:
                    # Feature missing — cannot build valid vector
                    logger.debug("Missing feature '%s' in input", feature_name)
                    return None
            return vector
        except (TypeError, ValueError) as exc:
            logger.debug("Invalid feature value: %s", exc)
            return None

    def _get_top_feature_names(self, n: int = 3) -> list[str]:
        """Return the top N feature names by model importance.

        Parameters
        ----------
        n : int
            Number of top features to return.

        Returns
        -------
        list[str]
            Top N feature names sorted by importance descending.
        """
        if not self.is_loaded or self.model is None:
            return []

        try:
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:n]
            return [self.feature_names[i] for i in top_indices]
        except Exception:
            return []
