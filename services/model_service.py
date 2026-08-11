"""Model service for loading and running inference with the trained DDoS classifier.

Provides a ModelService class that loads the model, scaler, and metadata
from disk at startup, and exposes a predict() method for classifying
network flow features.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Mapping from PredictRequest snake_case field names to the actual
# SELECTED_FEATURES column names used during training.
_FIELD_TO_FEATURE: dict[str, str] = {
    "flow_duration": "Flow Duration",
    "total_fwd_packets": "Total Fwd Packets",
    "total_backward_packets": "Total Backward Packets",
    "total_length_of_fwd_packets": "Total Length of Fwd Packets",
    "total_length_of_bwd_packets": "Total Length of Bwd Packets",
    "fwd_packet_length_max": "Fwd Packet Length Max",
    "fwd_packet_length_min": "Fwd Packet Length Min",
    "fwd_packet_length_mean": "Fwd Packet Length Mean",
    "bwd_packet_length_max": "Bwd Packet Length Max",
    "bwd_packet_length_min": "Bwd Packet Length Min",
    "bwd_packet_length_mean": "Bwd Packet Length Mean",
    "flow_bytes_per_s": "Flow Bytes/s",
    "flow_packets_per_s": "Flow Packets/s",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "fwd_iat_total": "Fwd IAT Total",
    "fwd_iat_mean": "Fwd IAT Mean",
    "bwd_iat_total": "Bwd IAT Total",
    "bwd_iat_mean": "Bwd IAT Mean",
    "fwd_psh_flags": "Fwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags",
    "fwd_header_length": "Fwd Header Length",
    "bwd_header_length": "Bwd Header Length",
    "fwd_packets_per_s": "Fwd Packets/s",
    "bwd_packets_per_s": "Bwd Packets/s",
    "min_packet_length": "Min Packet Length",
    "max_packet_length": "Max Packet Length",
    "packet_length_mean": "Packet Length Mean",
    "packet_length_std": "Packet Length Std",
    "packet_length_variance": "Packet Length Variance",
    "fin_flag_count": "FIN Flag Count",
    "syn_flag_count": "SYN Flag Count",
    "rst_flag_count": "RST Flag Count",
    "psh_flag_count": "PSH Flag Count",
    "ack_flag_count": "ACK Flag Count",
    "urg_flag_count": "URG Flag Count",
    "down_up_ratio": "Down/Up Ratio",
    "average_packet_size": "Average Packet Size",
    "avg_fwd_segment_size": "Avg Fwd Segment Size",
    "avg_bwd_segment_size": "Avg Bwd Segment Size",
    "init_win_bytes_forward": "Init_Win_bytes_forward",
    "init_win_bytes_backward": "Init_Win_bytes_backward",
    "act_data_pkt_fwd": "act_data_pkt_fwd",
    "min_seg_size_forward": "min_seg_size_forward",
}

# Reverse mapping: feature name → snake_case field name
_FEATURE_TO_FIELD: dict[str, str] = {v: k for k, v in _FIELD_TO_FEATURE.items()}


class ModelService:
    """Singleton that loads model + scaler on startup."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names: list[str] = []
        self.is_loaded: bool = False

    def load(self, model_path: str, scaler_path: str, metadata_path: str) -> None:
        """Load model, scaler, and feature names from disk.

        Parameters
        ----------
        model_path : str
            Path to the trained model .joblib file.
        scaler_path : str
            Path to the fitted StandardScaler .joblib file.
        metadata_path : str
            Path to the metadata.json file containing feature names.
        """
        try:
            model_file = Path(model_path)
            scaler_file = Path(scaler_path)
            metadata_file = Path(metadata_path)

            if not model_file.exists():
                logger.error("Model file not found: %s", model_path)
                self.is_loaded = False
                return

            if not scaler_file.exists():
                logger.error("Scaler file not found: %s", scaler_path)
                self.is_loaded = False
                return

            if not metadata_file.exists():
                logger.error("Metadata file not found: %s", metadata_path)
                self.is_loaded = False
                return

            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            self.feature_names = metadata["features"]
            self.is_loaded = True
            logger.info(
                "Model loaded successfully: type=%s, features=%d",
                metadata.get("model_type", "unknown"),
                len(self.feature_names),
            )

        except Exception as exc:
            logger.error("Failed to load model artifacts: %s", exc)
            self.model = None
            self.scaler = None
            self.feature_names = []
            self.is_loaded = False

    def predict(self, features: dict[str, float]) -> dict:
        """Scale input features, run inference, return classification result.

        Parameters
        ----------
        features : dict[str, float]
            Dictionary mapping snake_case field names (from PredictRequest)
            to their float values.

        Returns
        -------
        dict
            {"label": "Benign" or "DDoS", "probability": float}

        Raises
        ------
        RuntimeError
            If the model is not loaded.
        ValueError
            If required features are missing from the input.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Cannot make predictions.")

        # Build feature vector in the order expected by the model
        feature_values = []
        for feature_name in self.feature_names:
            # Map the feature name back to the snake_case field name
            field_name = _FEATURE_TO_FIELD.get(feature_name)
            if field_name is None:
                raise ValueError(
                    f"Unknown feature in metadata: '{feature_name}'. "
                    "Cannot map to request field."
                )
            if field_name not in features:
                raise ValueError(
                    f"Missing required feature: '{field_name}' "
                    f"(maps to '{feature_name}')"
                )
            feature_values.append(features[field_name])

        # Scale features using the loaded scaler
        X = np.array(feature_values, dtype=np.float64).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Run inference
        probabilities = self.model.predict_proba(X_scaled)[0]
        # Class 0 = Benign, Class 1 = DDoS
        ddos_probability = float(probabilities[1])
        predicted_class = int(self.model.predict(X_scaled)[0])

        label = "DDoS" if predicted_class == 1 else "Benign"

        return {"label": label, "probability": ddos_probability}
