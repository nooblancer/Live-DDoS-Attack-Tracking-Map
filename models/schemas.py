"""Pydantic schemas for API payloads, database records, and ML service contracts."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# V2: Attack type definitions and mappings
# ---------------------------------------------------------------------------


class AttackType(str, Enum):
    """12-class DDoS attack types from CIC-DDoS2019 dataset."""

    SYN_FLOOD = "SYN Flood"
    UDP_FLOOD = "UDP Flood"
    DNS_AMPLIFICATION = "DNS Amplification"
    HTTP_FLOOD = "HTTP Flood"
    LDAP = "LDAP"
    NTP = "NTP"
    MSSQL = "MSSQL"
    NETBIOS = "NetBIOS"
    SSDP = "SSDP"
    TFTP = "TFTP"
    UDPLAG = "UDPLag"
    WEBDDOS = "WebDDoS"


# Index → label lookup (matches model output indices)
ATTACK_TYPE_INDEX: dict[int, str] = {
    0: AttackType.SYN_FLOOD.value,
    1: AttackType.UDP_FLOOD.value,
    2: AttackType.DNS_AMPLIFICATION.value,
    3: AttackType.HTTP_FLOOD.value,
    4: AttackType.LDAP.value,
    5: AttackType.NTP.value,
    6: AttackType.MSSQL.value,
    7: AttackType.NETBIOS.value,
    8: AttackType.SSDP.value,
    9: AttackType.TFTP.value,
    10: AttackType.UDPLAG.value,
    11: AttackType.WEBDDOS.value,
}

# Attack type → hex color for globe arcs and UI elements
ATTACK_TYPE_COLORS: dict[str, str] = {
    AttackType.SYN_FLOOD.value: "#FF0040",
    AttackType.UDP_FLOOD.value: "#FF8C00",
    AttackType.DNS_AMPLIFICATION.value: "#9B59B6",
    AttackType.HTTP_FLOOD.value: "#FFD700",
    AttackType.LDAP.value: "#00E5FF",
    AttackType.NTP.value: "#00E5FF",
    AttackType.MSSQL.value: "#00E5FF",
    AttackType.NETBIOS.value: "#00FF41",
    AttackType.SSDP.value: "#00FF41",
    AttackType.TFTP.value: "#00FF41",
    AttackType.UDPLAG.value: "#00FF41",
    AttackType.WEBDDOS.value: "#FFD700",
}


def confidence_to_severity(confidence: float) -> str:
    """Map a confidence score to a severity level.

    - confidence >= 0.95 → "critical"
    - confidence >= 0.80 → "high"
    - confidence >= 0.60 → "medium"
    - confidence < 0.60  → "low"
    """
    if confidence >= 0.95:
        return "critical"
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# V2: Classification result (dataclass for internal use)
# ---------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    """Result of a single flow classification by the multi-class model."""

    attack_type: str  # One of 12 attack types
    confidence: float  # 0.0 to 1.0
    classified_in_ms: float  # Inference time
    top_features: list[str]  # Top 3 contributing features


# ---------------------------------------------------------------------------
# V2: Enhanced Pydantic models for API responses
# ---------------------------------------------------------------------------


class FeatureImportance(BaseModel):
    """Single feature importance entry for model stats."""

    name: str
    importance: float


class EnhancedAttackEvent(BaseModel):
    """V2 SSE payload with full classification details."""

    ip_address: str
    latitude: float
    longitude: float
    country: str | None = None
    city: str | None = None
    isp: str | None = None
    attack_type: str  # One of 12 types
    confidence: float  # 0.0 to 1.0
    severity: str  # "low" | "medium" | "high" | "critical"
    classified_in_ms: float  # Inference time
    top_features: list[str]  # Top 3 feature names
    source_channel: str  # "replay" | "live"
    timestamp: str  # ISO 8601


class ReplayStatusResponse(BaseModel):
    """GET /api/replay/status response."""

    state: str  # "running" | "stopped"
    speed_multiplier: int  # 1, 10, 100, or 1000
    elapsed_seconds: float  # Time since start
    flows_processed: int  # Total flows sent to classifier
    loops_completed: int  # Times dataset has been replayed
    events_per_second: float  # Current emission rate


class ModelStatsResponse(BaseModel):
    """GET /api/model-stats response."""

    model_type: str  # "XGBoost"
    f1_score: float
    precision: float
    recall: float
    roc_auc: float
    total_predictions: int
    attacks_detected: int
    benign_classified: int
    predictions_per_second: float  # Sliding 60s window
    top_features: list[FeatureImportance]  # Top 10
    attack_type_breakdown: dict[str, int]  # Type → count


class TopAttackerEntry(BaseModel):
    """Single row in the top attackers table."""

    rank: int
    ip_address: str  # Partially masked: "185.220.x.x"
    country: str | None
    attack_count: int
    last_seen: str  # ISO 8601


class ThreatIP(BaseModel):
    """Internal model for aggregated threat intelligence."""

    ip_address: str
    source: str  # "abuseipdb" | "firehol" | "feodo" | "emerging_threats"
    confidence: float  # 0.0 to 1.0
    tags: list[str]  # e.g. ["botnet_c2", "scanner"]
    first_seen: datetime | None = None


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
