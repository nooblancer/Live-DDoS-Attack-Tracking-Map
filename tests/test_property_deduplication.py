"""Property-based tests for deduplication retaining maximum confidence."""

# Feature: v2-attack-tracking-overhaul, Property 3: Deduplication Retains Maximum Confidence

from hypothesis import given, settings
from hypothesis import strategies as st

from models.schemas import ThreatIP
from services.threat_aggregator import ThreatAggregator


# --- Strategies ---

# Valid sources for ThreatIP entries
VALID_SOURCES = ["abuseipdb", "firehol", "feodo", "emerging_threats"]

# Strategy for generating a valid IPv4 address string
ipv4_strategy = st.tuples(
    st.integers(min_value=1, max_value=254),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=1, max_value=254),
).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}")

# Strategy for a single tag (use sampled_from for speed)
tag_strategy = st.sampled_from([
    "botnet_c2", "scanner", "blacklisted", "compromised", "malware",
    "firehol_level1", "firehol_level2", "firehol_level3", "spam", "brute_force",
    "phishing", "tor_exit", "proxy", "vpn", "ddos",
])

# Strategy for generating a single ThreatIP entry
threat_ip_strategy = st.builds(
    ThreatIP,
    ip_address=ipv4_strategy,
    source=st.sampled_from(VALID_SOURCES),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    tags=st.lists(tag_strategy, min_size=1, max_size=5),
    first_seen=st.none(),
)

# Strategy for a list of ThreatIP entries with guaranteed duplicates
# Draws a small pool of IPs and assigns them to multiple entries
def threat_ip_list_with_duplicates():
    """Generate a list of ThreatIP entries where some IPs are duplicated."""
    return st.lists(
        st.tuples(
            # Use a small IP pool to increase duplicate likelihood
            st.sampled_from(["10.0.0.1", "10.0.0.2", "10.0.0.3", "192.168.1.1", "172.16.0.1"]),
            st.sampled_from(VALID_SOURCES),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            st.lists(tag_strategy, min_size=1, max_size=3),
        ),
        min_size=2,
        max_size=30,
    ).map(
        lambda entries: [
            ThreatIP(
                ip_address=ip,
                source=source,
                confidence=conf,
                tags=tags,
                first_seen=None,
            )
            for ip, source, conf, tags in entries
        ]
    )


# --- Helper ---

def _create_aggregator() -> ThreatAggregator:
    """Create a minimal ThreatAggregator for testing deduplicate().

    The deduplicate() method is a pure function that doesn't access any
    instance attributes, so we can pass None for all dependencies.
    """
    return ThreatAggregator(
        geo_service=None,  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
        event_bus=None,  # type: ignore[arg-type]
        stats=None,
    )


# --- Property Tests ---


@settings(max_examples=200)
@given(entries=threat_ip_list_with_duplicates())
def test_deduplicated_ips_are_unique(entries: list[ThreatIP]):
    """Property 3: After deduplication, each IP appears exactly once.

    For any list of ThreatIP entries (with duplicate IPs having different confidence
    scores), after deduplication each IP appears exactly once.

    **Validates: Requirements 3.3**
    """
    aggregator = _create_aggregator()
    result = aggregator.deduplicate(entries)

    # Collect IPs from result
    result_ips = [t.ip_address for t in result]

    # Each IP should appear exactly once
    assert len(result_ips) == len(set(result_ips)), (
        f"Duplicate IPs found in deduplicated result: {result_ips}"
    )

    # All unique IPs from input should be present in output
    input_ips = set(t.ip_address for t in entries)
    output_ips = set(result_ips)
    assert input_ips == output_ips, (
        f"Input IPs {input_ips} != output IPs {output_ips}"
    )


@settings(max_examples=200)
@given(entries=threat_ip_list_with_duplicates())
def test_deduplicated_confidence_is_maximum(entries: list[ThreatIP]):
    """Property 3: The confidence for each deduplicated IP equals the maximum across all inputs.

    For any list of ThreatIP entries containing duplicate IP addresses with varying
    confidence scores, the deduplication function SHALL produce a list where each IP's
    confidence score equals the maximum confidence across all occurrences of that IP.

    **Validates: Requirements 3.3**
    """
    aggregator = _create_aggregator()
    result = aggregator.deduplicate(entries)

    # Build expected max confidence per IP from input
    expected_max: dict[str, float] = {}
    for entry in entries:
        if entry.ip_address not in expected_max:
            expected_max[entry.ip_address] = entry.confidence
        else:
            expected_max[entry.ip_address] = max(expected_max[entry.ip_address], entry.confidence)

    # Verify each result entry has the maximum confidence
    for threat in result:
        assert threat.confidence == expected_max[threat.ip_address], (
            f"IP {threat.ip_address}: expected confidence {expected_max[threat.ip_address]}, "
            f"got {threat.confidence}"
        )


@settings(max_examples=200)
@given(entries=threat_ip_list_with_duplicates())
def test_deduplicated_tags_are_merged(entries: list[ThreatIP]):
    """Property 3: Tags are merged — all unique tags from all entries for the same IP are present.

    For any list of ThreatIP entries with duplicate IPs, after deduplication the tags
    for each IP SHALL contain all unique tags from all input entries for that IP.

    **Validates: Requirements 3.3**
    """
    aggregator = _create_aggregator()
    result = aggregator.deduplicate(entries)

    # Build expected merged tags per IP from input
    expected_tags: dict[str, set[str]] = {}
    for entry in entries:
        if entry.ip_address not in expected_tags:
            expected_tags[entry.ip_address] = set()
        expected_tags[entry.ip_address].update(entry.tags)

    # Verify each result entry has all expected tags
    for threat in result:
        result_tags = set(threat.tags)
        expected = expected_tags[threat.ip_address]
        assert expected.issubset(result_tags), (
            f"IP {threat.ip_address}: missing tags {expected - result_tags}. "
            f"Expected all of {expected}, got {result_tags}"
        )


@settings(max_examples=100)
@given(entries=st.lists(threat_ip_strategy, min_size=1, max_size=20))
def test_single_entries_pass_through_unchanged(entries: list[ThreatIP]):
    """Property 3: Single (non-duplicate) entries pass through with same confidence.

    For any list of ThreatIP entries where each IP is unique, the deduplication
    function SHALL return all entries unchanged.

    **Validates: Requirements 3.3**
    """
    # Make IPs unique by overriding with index-based IPs
    unique_entries = []
    for i, entry in enumerate(entries):
        unique_entries.append(
            ThreatIP(
                ip_address=f"10.0.{i // 256}.{i % 256 + 1}",
                source=entry.source,
                confidence=entry.confidence,
                tags=list(entry.tags),
                first_seen=entry.first_seen,
            )
        )

    aggregator = _create_aggregator()
    result = aggregator.deduplicate(unique_entries)

    # Same number of results as inputs (all unique)
    assert len(result) == len(unique_entries), (
        f"Expected {len(unique_entries)} results, got {len(result)}"
    )

    # Each entry should have its original confidence preserved
    result_lookup = {t.ip_address: t for t in result}
    for entry in unique_entries:
        assert entry.ip_address in result_lookup
        assert result_lookup[entry.ip_address].confidence == entry.confidence


def test_empty_input_returns_empty_output():
    """Property 3: Empty input returns empty output.

    The deduplication function SHALL return an empty list when given an empty input.

    **Validates: Requirements 3.3**
    """
    aggregator = _create_aggregator()
    result = aggregator.deduplicate([])
    assert result == [], f"Expected empty list, got {result}"
