"""Property-based tests for log entry formatting completeness."""

# Feature: v2-attack-tracking-overhaul, Property 13: Log Entry Formatting Completeness

from hypothesis import given, settings
from hypothesis import strategies as st

from models.schemas import AttackType, EnhancedAttackEvent


# --- Log Formatting Helper ---


def mask_ip(ip_address: str) -> str:
    """Mask an IP address by replacing the last two octets with 'x.x'.

    Example: "185.220.101.34" → "185.220.x.x"
    """
    parts = ip_address.split(".")
    return f"{parts[0]}.{parts[1]}.x.x"


def format_log_entry(event: EnhancedAttackEvent) -> str:
    """Format an EnhancedAttackEvent into a terminal-style log entry string.

    The formatted string contains:
    - Timestamp (HH:MM:SS extracted from ISO 8601)
    - Attack type label
    - Partially masked IP (last two octets → "x.x")
    - Country code (if provided)
    - Confidence as a percentage (e.g., "95%")
    - Classification time in milliseconds (e.g., "1.2ms")
    """
    # Extract HH:MM:SS from ISO 8601 timestamp
    # Format: "2024-01-15T14:32:07Z" or "2024-01-15T14:32:07.123Z"
    time_part = event.timestamp.split("T")[1] if "T" in event.timestamp else event.timestamp
    # Remove trailing Z or timezone info to get time
    time_clean = time_part.rstrip("Z")
    # Handle timezone offset like +00:00
    if "+" in time_clean:
        time_clean = time_clean.split("+")[0]
    elif time_clean.count("-") > 0:
        # Could be negative offset — only strip if after the time portion
        pass
    # Take HH:MM:SS portion
    hms = time_clean.split(".")[0]  # Remove fractional seconds

    masked_ip = mask_ip(event.ip_address)
    confidence_pct = f"{int(event.confidence * 100)}%"
    classification_time = f"{event.classified_in_ms:.1f}ms"
    country = event.country if event.country else "??"

    return f"> {hms} {event.attack_type} {masked_ip} {country} {confidence_pct} {classification_time}"


# --- Strategies ---

attack_type_strategy = st.sampled_from([t.value for t in AttackType])

ip_octet = st.integers(min_value=0, max_value=255)
ip_strategy = st.tuples(ip_octet, ip_octet, ip_octet, ip_octet).map(
    lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}"
)

# Generate valid ISO 8601 timestamps
timestamp_strategy = st.tuples(
    st.integers(min_value=2020, max_value=2030),  # year
    st.integers(min_value=1, max_value=12),  # month
    st.integers(min_value=1, max_value=28),  # day (safe max)
    st.integers(min_value=0, max_value=23),  # hour
    st.integers(min_value=0, max_value=59),  # minute
    st.integers(min_value=0, max_value=59),  # second
).map(
    lambda t: f"{t[0]:04d}-{t[1]:02d}-{t[2]:02d}T{t[3]:02d}:{t[4]:02d}:{t[5]:02d}Z"
)

confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

severity_strategy = st.sampled_from(["low", "medium", "high", "critical"])

country_strategy = st.one_of(
    st.just(None),
    st.text(alphabet=st.characters(whitelist_categories=("Lu",)), min_size=2, max_size=2),
)

classified_in_ms_strategy = st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)

feature_strategy = st.lists(
    st.sampled_from([
        "flow_duration", "total_fwd_packets", "total_backward_packets",
        "fwd_packet_length_max", "flow_bytes_per_s", "flow_packets_per_s",
        "flow_iat_mean", "flow_iat_std", "fwd_iat_total", "bwd_iat_total",
    ]),
    min_size=3,
    max_size=3,
)

source_channel_strategy = st.sampled_from(["replay", "live"])

# Full EnhancedAttackEvent strategy
enhanced_event_strategy = st.builds(
    EnhancedAttackEvent,
    ip_address=ip_strategy,
    latitude=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
    longitude=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
    country=country_strategy,
    city=st.one_of(st.just(None), st.text(min_size=1, max_size=20)),
    isp=st.one_of(st.just(None), st.text(min_size=1, max_size=30)),
    attack_type=attack_type_strategy,
    confidence=confidence_strategy,
    severity=severity_strategy,
    classified_in_ms=classified_in_ms_strategy,
    top_features=feature_strategy,
    source_channel=source_channel_strategy,
    timestamp=timestamp_strategy,
)


# --- Property Tests ---


@settings(max_examples=200)
@given(event=enhanced_event_strategy)
def test_log_entry_contains_timestamp(event: EnhancedAttackEvent):
    """Property 13: The formatted log entry contains a timestamp substring (HH:MM:SS).

    For any valid EnhancedAttackEvent, the formatted log entry SHALL contain
    a timestamp portion extracted from the event's ISO 8601 timestamp.

    **Validates: Requirements 8.1**
    """
    log_entry = format_log_entry(event)

    # Extract expected HH:MM:SS from the timestamp
    time_part = event.timestamp.split("T")[1].rstrip("Z").split(".")[0]

    assert time_part in log_entry, (
        f"Expected timestamp '{time_part}' not found in log entry: '{log_entry}'"
    )


@settings(max_examples=200)
@given(event=enhanced_event_strategy)
def test_log_entry_contains_attack_type(event: EnhancedAttackEvent):
    """Property 13: The formatted log entry contains the attack type label.

    For any valid EnhancedAttackEvent, the formatted log entry SHALL contain
    the full attack type label string.

    **Validates: Requirements 8.1**
    """
    log_entry = format_log_entry(event)

    assert event.attack_type in log_entry, (
        f"Expected attack type '{event.attack_type}' not found in log entry: '{log_entry}'"
    )


@settings(max_examples=200)
@given(event=enhanced_event_strategy)
def test_log_entry_contains_masked_ip(event: EnhancedAttackEvent):
    """Property 13: The formatted log entry contains a partially masked IP.

    For any valid EnhancedAttackEvent, the formatted log entry SHALL contain
    the IP address with the last two octets replaced with "x.x".

    **Validates: Requirements 8.1**
    """
    log_entry = format_log_entry(event)

    expected_masked = mask_ip(event.ip_address)

    assert expected_masked in log_entry, (
        f"Expected masked IP '{expected_masked}' not found in log entry: '{log_entry}'"
    )
    # Verify the masked IP has format "A.B.x.x"
    parts = expected_masked.split(".")
    assert parts[2] == "x" and parts[3] == "x", (
        f"Masked IP '{expected_masked}' does not end with 'x.x'"
    )


@settings(max_examples=200)
@given(event=enhanced_event_strategy)
def test_log_entry_contains_country(event: EnhancedAttackEvent):
    """Property 13: The formatted log entry contains the country code.

    For any valid EnhancedAttackEvent with a country, the formatted log entry
    SHALL contain the country code. If country is None, "??" is used.

    **Validates: Requirements 8.1**
    """
    log_entry = format_log_entry(event)

    if event.country:
        assert event.country in log_entry, (
            f"Expected country '{event.country}' not found in log entry: '{log_entry}'"
        )
    else:
        assert "??" in log_entry, (
            f"Expected '??' for null country not found in log entry: '{log_entry}'"
        )


@settings(max_examples=200)
@given(event=enhanced_event_strategy)
def test_log_entry_contains_confidence_percentage(event: EnhancedAttackEvent):
    """Property 13: The formatted log entry contains confidence as a percentage.

    For any valid EnhancedAttackEvent, the formatted log entry SHALL contain
    the confidence value expressed as a percentage (e.g., "95%").

    **Validates: Requirements 8.1**
    """
    log_entry = format_log_entry(event)

    expected_pct = f"{int(event.confidence * 100)}%"

    assert expected_pct in log_entry, (
        f"Expected confidence percentage '{expected_pct}' not found in log entry: '{log_entry}'"
    )


@settings(max_examples=200)
@given(event=enhanced_event_strategy)
def test_log_entry_contains_classification_time_ms(event: EnhancedAttackEvent):
    """Property 13: The formatted log entry contains classification time in milliseconds.

    For any valid EnhancedAttackEvent, the formatted log entry SHALL contain
    the classification time formatted as milliseconds (e.g., "1.2ms").

    **Validates: Requirements 8.1**
    """
    log_entry = format_log_entry(event)

    expected_ms = f"{event.classified_in_ms:.1f}ms"

    assert expected_ms in log_entry, (
        f"Expected classification time '{expected_ms}' not found in log entry: '{log_entry}'"
    )


@settings(max_examples=200)
@given(event=enhanced_event_strategy)
def test_log_entry_formatting_completeness(event: EnhancedAttackEvent):
    """Property 13: Full completeness check — all required fields present in one entry.

    For any valid EnhancedAttackEvent, the formatted log entry string SHALL
    contain ALL of: timestamp, attack type, masked IP, country, confidence
    percentage, and classification time in ms.

    **Validates: Requirements 8.1**
    """
    log_entry = format_log_entry(event)

    # 1. Timestamp (HH:MM:SS)
    time_part = event.timestamp.split("T")[1].rstrip("Z").split(".")[0]
    assert time_part in log_entry, f"Missing timestamp in: '{log_entry}'"

    # 2. Attack type
    assert event.attack_type in log_entry, f"Missing attack type in: '{log_entry}'"

    # 3. Masked IP
    masked = mask_ip(event.ip_address)
    assert masked in log_entry, f"Missing masked IP in: '{log_entry}'"

    # 4. Country
    country = event.country if event.country else "??"
    assert country in log_entry, f"Missing country in: '{log_entry}'"

    # 5. Confidence percentage
    conf_pct = f"{int(event.confidence * 100)}%"
    assert conf_pct in log_entry, f"Missing confidence pct in: '{log_entry}'"

    # 6. Classification time in ms
    class_time = f"{event.classified_in_ms:.1f}ms"
    assert class_time in log_entry, f"Missing classification time in: '{log_entry}'"
