"""Property-based tests for SSE event schema completeness."""

# Feature: v2-attack-tracking-overhaul, Property 5: SSE Event Schema Completeness

from hypothesis import given, settings
from hypothesis import strategies as st

from models.schemas import AttackType, EnhancedAttackEvent, confidence_to_severity


# --- Constants ---

# All 13 enhanced fields required in the serialized event
ENHANCED_FIELDS = {
    "ip_address",
    "latitude",
    "longitude",
    "country",
    "city",
    "isp",
    "attack_type",
    "confidence",
    "severity",
    "classified_in_ms",
    "top_features",
    "source_channel",
    "timestamp",
}

# Legacy fields that must also be present for backward compatibility
LEGACY_FIELDS = {
    "ip_address",
    "latitude",
    "longitude",
    "country",
    "city",
    "isp",
    "timestamp",
}


# --- Strategies ---

# Strategy for valid IPv4 addresses
ip_strategy = st.from_regex(
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", fullmatch=True
).filter(lambda ip: all(0 <= int(octet) <= 255 for octet in ip.split(".")))

# Strategy for latitude (-90 to 90)
latitude_strategy = st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False)

# Strategy for longitude (-180 to 180)
longitude_strategy = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)

# Strategy for country names (nullable)
country_strategy = st.one_of(st.none(), st.text(min_size=1, max_size=50))

# Strategy for city names (nullable)
city_strategy = st.one_of(st.none(), st.text(min_size=1, max_size=50))

# Strategy for ISP names (nullable)
isp_strategy = st.one_of(st.none(), st.text(min_size=1, max_size=100))

# Strategy for attack types (one of 12 valid types)
attack_type_strategy = st.sampled_from([at.value for at in AttackType])

# Strategy for confidence (0.0 to 1.0)
confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for classification time in ms (positive float)
classified_in_ms_strategy = st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)

# Strategy for top features (list of 3 feature name strings)
top_features_strategy = st.lists(
    st.text(min_size=1, max_size=50),
    min_size=3,
    max_size=3,
)

# Strategy for source channel
source_channel_strategy = st.sampled_from(["replay", "live"])

# Strategy for timestamp (ISO 8601 string)
timestamp_strategy = st.from_regex(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", fullmatch=True
)

# Composite strategy: build a valid EnhancedAttackEvent
enhanced_event_strategy = st.builds(
    EnhancedAttackEvent,
    ip_address=ip_strategy,
    latitude=latitude_strategy,
    longitude=longitude_strategy,
    country=country_strategy,
    city=city_strategy,
    isp=isp_strategy,
    attack_type=attack_type_strategy,
    confidence=confidence_strategy,
    severity=confidence_strategy.map(confidence_to_severity),
    classified_in_ms=classified_in_ms_strategy,
    top_features=top_features_strategy,
    source_channel=source_channel_strategy,
    timestamp=timestamp_strategy,
)


# --- Property Tests ---


@settings(max_examples=200)
@given(event=enhanced_event_strategy)
def test_serialized_event_contains_all_13_enhanced_fields(event: EnhancedAttackEvent):
    """Property 5: For any valid EnhancedAttackEvent, model_dump() contains all 13 required fields.

    For any event published to the SSE bus, the serialized event SHALL contain all 13
    enhanced fields (ip_address, latitude, longitude, country, city, isp, attack_type,
    confidence, severity, classified_in_ms, top_features, source_channel, timestamp).

    **Validates: Requirements 4.1, 16.2**
    """
    serialized = event.model_dump()

    missing = ENHANCED_FIELDS - set(serialized.keys())
    assert not missing, (
        f"Serialized event is missing enhanced fields: {missing}. "
        f"Got keys: {sorted(serialized.keys())}"
    )


@settings(max_examples=200)
@given(event=enhanced_event_strategy)
def test_serialized_event_contains_all_legacy_fields(event: EnhancedAttackEvent):
    """Property 5: The serialized dict also contains all legacy fields for backward compatibility.

    For any event published to the SSE bus, the serialized event SHALL contain all
    legacy fields (ip_address, latitude, longitude, country, city, isp, timestamp).

    **Validates: Requirements 4.1, 16.2**
    """
    serialized = event.model_dump()

    missing = LEGACY_FIELDS - set(serialized.keys())
    assert not missing, (
        f"Serialized event is missing legacy fields: {missing}. "
        f"Got keys: {sorted(serialized.keys())}"
    )


@settings(max_examples=200)
@given(event=enhanced_event_strategy)
def test_serialized_event_field_types_are_correct(event: EnhancedAttackEvent):
    """Property 5: All field types in the serialized dict are correct.

    Strings for ip/country/city/isp/attack_type/severity/source_channel/timestamp,
    float for lat/lon/confidence/classified_in_ms, list for top_features.

    **Validates: Requirements 4.1, 16.2**
    """
    serialized = event.model_dump()

    # String fields (some may be None for nullable fields)
    string_fields_required = ["ip_address", "attack_type", "severity", "source_channel", "timestamp"]
    for field in string_fields_required:
        assert isinstance(serialized[field], str), (
            f"Field '{field}' should be str, got {type(serialized[field]).__name__}: {serialized[field]}"
        )

    # Nullable string fields (str or None)
    nullable_string_fields = ["country", "city", "isp"]
    for field in nullable_string_fields:
        assert serialized[field] is None or isinstance(serialized[field], str), (
            f"Field '{field}' should be str or None, got {type(serialized[field]).__name__}: {serialized[field]}"
        )

    # Float fields
    float_fields = ["latitude", "longitude", "confidence", "classified_in_ms"]
    for field in float_fields:
        assert isinstance(serialized[field], float), (
            f"Field '{field}' should be float, got {type(serialized[field]).__name__}: {serialized[field]}"
        )

    # List field
    assert isinstance(serialized["top_features"], list), (
        f"Field 'top_features' should be list, got {type(serialized['top_features']).__name__}"
    )
    for item in serialized["top_features"]:
        assert isinstance(item, str), (
            f"Each item in 'top_features' should be str, got {type(item).__name__}: {item}"
        )
