"""Pydantic schemas for API payloads, database records, and ML service contracts."""

from datetime import datetime

from pydantic import BaseModel


class AttackRecord(BaseModel):
    """Internal representation of a geolocated attack source."""

    ip_address: str
    latitude: float
    longitude: float
    country: str | None = None
    city: str | None = None
    isp: str | None = None
    threat_source: str = "firehol_level1"
    first_seen: datetime
    last_seen: datetime


class AttackEvent(BaseModel):
    """SSE payload sent to dashboard clients."""

    ip_address: str
    latitude: float
    longitude: float
    country: str | None = None
    city: str | None = None
    isp: str | None = None
    timestamp: str  # ISO 8601


class DashboardStats(BaseModel):
    """Statistics panel data."""

    total_ips: int
    countries: int
    attacks_last_hour: int
    latest_timestamp: str | None


class TimelineBucket(BaseModel):
    """Single hourly bucket in the timeline chart."""

    hour: str  # ISO 8601 hour start
    count: int


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str  # "ok" | "degraded"
    model_loaded: bool
    database_records: int


class ModelMetadata(BaseModel):
    """Saved alongside the trained model."""

    model_type: str  # "RandomForest" | "XGBoost"
    features: list[str]
    training_date: str
    metrics: dict[str, float]  # precision, recall, f1, roc_auc


class PredictRequest(BaseModel):
    """Flow features for prediction. All fields are floats matching SELECTED_FEATURES."""

    flow_duration: float
    total_fwd_packets: float
    total_backward_packets: float
    total_length_of_fwd_packets: float
    total_length_of_bwd_packets: float
    fwd_packet_length_max: float
    fwd_packet_length_min: float
    fwd_packet_length_mean: float
    bwd_packet_length_max: float
    bwd_packet_length_min: float
    bwd_packet_length_mean: float
    flow_bytes_per_s: float
    flow_packets_per_s: float
    flow_iat_mean: float
    flow_iat_std: float
    flow_iat_max: float
    flow_iat_min: float
    fwd_iat_total: float
    fwd_iat_mean: float
    bwd_iat_total: float
    bwd_iat_mean: float
    fwd_psh_flags: float
    fwd_urg_flags: float
    fwd_header_length: float
    bwd_header_length: float
    fwd_packets_per_s: float
    bwd_packets_per_s: float
    min_packet_length: float
    max_packet_length: float
    packet_length_mean: float
    packet_length_std: float
    packet_length_variance: float
    fin_flag_count: float
    syn_flag_count: float
    rst_flag_count: float
    psh_flag_count: float
    ack_flag_count: float
    urg_flag_count: float
    down_up_ratio: float
    average_packet_size: float
    avg_fwd_segment_size: float
    avg_bwd_segment_size: float
    init_win_bytes_forward: float
    init_win_bytes_backward: float
    act_data_pkt_fwd: float
    min_seg_size_forward: float


class PredictResponse(BaseModel):
    """Prediction result from the ML model."""

    label: str  # "Benign" or "DDoS"
    probability: float
