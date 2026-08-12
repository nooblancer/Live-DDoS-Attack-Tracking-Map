"""Property-based tests for source failure resilience.

Tests that the ThreatAggregator's refresh_all() method handles individual source
failures gracefully — when any subset of sources fail, the aggregator continues
processing results from remaining sources without raising unhandled exceptions.
"""

# Feature: v2-attack-tracking-overhaul, Property 4: Source Failure Resilience

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from models.schemas import ThreatIP
from services.threat_aggregator import ThreatAggregator


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# The 4 source fetch method names on ThreatAggregator
ALL_SOURCES = ["fetch_abuseipdb", "fetch_firehol", "fetch_feodo", "fetch_emerging_threats"]

# Strategy to pick a non-empty subset of sources to fail (1 to 3 out of 4)
failing_sources_strategy = st.lists(
    st.sampled_from(ALL_SOURCES),
    min_size=1,
    max_size=3,
    unique=True,
)

# Strategy to pick a non-empty subset of sources to fail (1 to 4, including all failing)
failing_sources_all_strategy = st.lists(
    st.sampled_from(ALL_SOURCES),
    min_size=1,
    max_size=4,
    unique=True,
)

# Strategy for the type of exception that a failing source raises
exception_strategy = st.sampled_from([
    RuntimeError("Connection timeout"),
    TimeoutError("Request timed out after 30s"),
    ConnectionError("Failed to connect to remote host"),
    OSError("Network is unreachable"),
    ValueError("Unexpected response format"),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_threat_ips(source_name: str, count: int = 3) -> list[ThreatIP]:
    """Create a small list of valid ThreatIP entries for a given source."""
    source_key = source_name.replace("fetch_", "")
    if source_key == "abuseipdb":
        source_key = "abuseipdb"
    elif source_key == "firehol":
        source_key = "firehol"
    elif source_key == "feodo":
        source_key = "feodo"
    elif source_key == "emerging_threats":
        source_key = "emerging_threats"

    return [
        ThreatIP(
            ip_address=f"10.{ALL_SOURCES.index(source_name)}.0.{i + 1}",
            source=source_key,
            confidence=0.8,
            tags=["test_tag"],
            first_seen=None,
        )
        for i in range(count)
    ]


def _create_aggregator() -> ThreatAggregator:
    """Create a ThreatAggregator with mock dependencies.

    The geo_service, db, and event_bus are mocked so refresh_all() can
    run without real infrastructure.
    """
    mock_geo = AsyncMock()
    mock_geo.geolocate_batch = AsyncMock(return_value=[])

    mock_db = AsyncMock()
    mock_db.get_all_attacks = AsyncMock(return_value=[])
    mock_db.upsert_attack = AsyncMock()

    mock_event_bus = AsyncMock()
    mock_event_bus.publish = AsyncMock()

    return ThreatAggregator(
        geo_service=mock_geo,
        db=mock_db,
        event_bus=mock_event_bus,
        stats=None,
    )


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    failing_sources=failing_sources_strategy,
    exception=exception_strategy,
)
@pytest.mark.asyncio
async def test_partial_source_failure_does_not_raise(
    failing_sources: list[str],
    exception: Exception,
):
    """Property 4: refresh_all() completes without exception when 1-3 sources fail.

    For any subset of threat sources (1 to 3 out of 4) that return errors, the
    Threat Aggregator SHALL still produce valid results from the remaining
    non-failing sources without raising an unhandled exception.

    **Validates: Requirements 3.6**
    """
    aggregator = _create_aggregator()

    # Determine which sources succeed
    non_failing_sources = [s for s in ALL_SOURCES if s not in failing_sources]

    # Patch failing sources to raise exceptions
    patches = {}
    for source in failing_sources:
        patches[source] = AsyncMock(side_effect=exception)

    # Patch non-failing sources to return valid ThreatIP entries
    for source in non_failing_sources:
        patches[source] = AsyncMock(return_value=_make_sample_threat_ips(source))

    with patch.multiple(aggregator, **patches):
        # Should NOT raise any exception
        await aggregator.refresh_all()


@settings(max_examples=100, deadline=None)
@given(
    failing_sources=failing_sources_strategy,
    exception=exception_strategy,
)
@pytest.mark.asyncio
async def test_non_failing_sources_are_processed(
    failing_sources: list[str],
    exception: Exception,
):
    """Property 4: Results from non-failing sources are still processed.

    When some sources fail, the aggregator still collects and processes IPs
    from the remaining working sources.

    **Validates: Requirements 3.6**
    """
    aggregator = _create_aggregator()

    non_failing_sources = [s for s in ALL_SOURCES if s not in failing_sources]

    # Patch sources
    patches = {}
    for source in failing_sources:
        patches[source] = AsyncMock(side_effect=exception)

    expected_ips: list[ThreatIP] = []
    for source in non_failing_sources:
        ips = _make_sample_threat_ips(source)
        expected_ips.extend(ips)
        patches[source] = AsyncMock(return_value=ips)

    # Mock geo to return results for the collected IPs so we can verify processing
    geo_results = [
        {"ip": ip.ip_address, "lat": 40.0, "lon": -74.0, "country": "US", "city": "NYC", "isp": "Test"}
        for ip in expected_ips
    ]
    aggregator._geo_service.geolocate_batch = AsyncMock(return_value=geo_results)

    with patch.multiple(aggregator, **patches):
        await aggregator.refresh_all()

    # If there are non-failing sources, geolocate_batch should have been called
    if non_failing_sources:
        aggregator._geo_service.geolocate_batch.assert_called_once()
        # Verify the IPs passed to geolocate_batch match expected
        call_args = aggregator._geo_service.geolocate_batch.call_args[0][0]
        expected_ip_set = {ip.ip_address for ip in expected_ips}
        actual_ip_set = set(call_args)
        assert actual_ip_set == expected_ip_set, (
            f"Expected IPs {expected_ip_set} to be geolocated, got {actual_ip_set}"
        )


@settings(max_examples=100, deadline=None)
@given(exception=exception_strategy)
@pytest.mark.asyncio
async def test_all_sources_failing_does_not_raise(exception: Exception):
    """Property 4: When ALL sources fail, refresh_all() still does not raise.

    The aggregator handles all sources failing gracefully — logs a warning and
    returns without crashing.

    **Validates: Requirements 3.6**
    """
    aggregator = _create_aggregator()

    # All 4 sources fail
    patches = {source: AsyncMock(side_effect=exception) for source in ALL_SOURCES}

    with patch.multiple(aggregator, **patches):
        # Should NOT raise any exception even when all sources fail
        await aggregator.refresh_all()

    # geolocate_batch should NOT have been called (no IPs collected)
    aggregator._geo_service.geolocate_batch.assert_not_called()
