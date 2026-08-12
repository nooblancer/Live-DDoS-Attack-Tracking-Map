"""Pydantic data models for the DDoS Attack Tracking Map."""

from models.schemas import (
    AttackRecord,
    AttackEvent,
    DashboardStats,
    TimelineBucket,
    HealthResponse,
    ModelMetadata,
    PredictRequest,
    PredictResponse,
    # V2 models
    AttackType,
    ATTACK_TYPE_INDEX,
    ATTACK_TYPE_COLORS,
    confidence_to_severity,
    ClassificationResult,
    FeatureImportance,
    EnhancedAttackEvent,
    ReplayStatusResponse,
    ModelStatsResponse,
    TopAttackerEntry,
    ThreatIP,
)

__all__ = [
    "AttackRecord",
    "AttackEvent",
    "DashboardStats",
    "TimelineBucket",
    "HealthResponse",
    "ModelMetadata",
    "PredictRequest",
    "PredictResponse",
    # V2 models
    "AttackType",
    "ATTACK_TYPE_INDEX",
    "ATTACK_TYPE_COLORS",
    "confidence_to_severity",
    "ClassificationResult",
    "FeatureImportance",
    "EnhancedAttackEvent",
    "ReplayStatusResponse",
    "ModelStatsResponse",
    "TopAttackerEntry",
    "ThreatIP",
]
