"""Property-based tests for top attackers sorted and bounded."""

# Feature: v2-attack-tracking-overhaul, Property 15: Top Attackers Sorted and Bounded

import asyncio
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from models.schemas import ClassificationResult
from services.database import DatabaseService
from services.stats_accumulator import StatsAccumulator


# --- Strategies ---

# Generate a list of (ip_address, count) pairs representing attacker records.
# Each IP is unique and count is a positive integer.
ip_octet = st.integers(min_value=1, max_value=254)

ip_strategy = st.tuples(ip_octet, ip_octet, ip_octet, ip_octet).map(
    lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}"
)

# Strategy for generating a list of unique IPs with random attack counts.
# We generate between 0 and 50 unique IPs, each with 1-100 attacks.
attacker_data_strategy = st.lists(
    st.tuples(ip_strategy, st.integers(min_value=1, max_value=100)),
    min_size=0,
    max_size=50,
    unique_by=lambda x: x[0],
)

# Attack types from the 12 valid types (non-benign to ensure they count as attacks)
attack_types = st.sampled_from([
    "SYN Flood", "UDP Flood", "DNS Amplification", "HTTP Flood",
    "LDAP", "NTP", "MSSQL", "NetBIOS", "SSDP", "TFTP", "UDPLag", "WebDDoS",
])


# --- Helpers ---


def _create_stats_accumulator() -> StatsAccumulator:
    """Create a StatsAccumulator with a mocked database service."""
    mock_db = MagicMock(spec=DatabaseService)
    mock_db._conn = None  # Disable DB operations
    return StatsAccumulator(db=mock_db)


async def _populate_accumulator(
    accumulator: StatsAccumulator,
    attacker_data: list[tuple[str, int]],
) -> None:
    """Populate the accumulator by recording predictions for each IP.

    Each (ip, count) pair triggers `count` calls to record_prediction for that IP.
    """
    for ip, count in attacker_data:
        for _ in range(count):
            result = ClassificationResult(
                attack_type="SYN Flood",
                confidence=0.95,
                classified_in_ms=1.0,
                top_features=["feature_a", "feature_b", "feature_c"],
            )
            await accumulator.record_prediction(result, source_ip=ip, country="US")


# --- Property Tests ---


@settings(max_examples=100)
@given(attacker_data=attacker_data_strategy)
def test_top_attackers_returns_at_most_20_entries(
    attacker_data: list[tuple[str, int]],
):
    """Property 15: get_top_attackers() returns at most 20 entries.

    For any number of IP addresses with varying counts, the result SHALL
    contain at most 20 entries regardless of total input size.

    **Validates: Requirements 14.4**
    """
    accumulator = _create_stats_accumulator()
    asyncio.run(_populate_accumulator(accumulator, attacker_data))

    result = accumulator.get_top_attackers()

    assert len(result) <= 20, (
        f"Expected at most 20 entries, got {len(result)} "
        f"(input had {len(attacker_data)} unique IPs)"
    )


@settings(max_examples=100)
@given(attacker_data=attacker_data_strategy)
def test_top_attackers_sorted_descending_by_attack_count(
    attacker_data: list[tuple[str, int]],
):
    """Property 15: Returned list is sorted in strictly descending order by attack_count.

    For any set of attacker records, the top-attackers function SHALL return a list
    sorted in descending order by attack_count.

    **Validates: Requirements 14.4**
    """
    accumulator = _create_stats_accumulator()
    asyncio.run(_populate_accumulator(accumulator, attacker_data))

    result = accumulator.get_top_attackers()

    if len(result) > 1:
        for i in range(len(result) - 1):
            assert result[i].attack_count >= result[i + 1].attack_count, (
                f"Result not sorted descending: entry {i} has count "
                f"{result[i].attack_count} but entry {i+1} has count "
                f"{result[i+1].attack_count}"
            )


@settings(max_examples=100)
@given(attacker_data=attacker_data_strategy)
def test_top_attackers_excluded_entries_have_lower_or_equal_count(
    attacker_data: list[tuple[str, int]],
):
    """Property 15: Every excluded entry has attack_count ≤ the minimum included entry.

    For any set of attacker records, each entry's attack_count in the result is
    greater than or equal to any entry not included in the result.

    **Validates: Requirements 14.4**
    """
    accumulator = _create_stats_accumulator()
    asyncio.run(_populate_accumulator(accumulator, attacker_data))

    result = accumulator.get_top_attackers()

    if not result:
        return  # Nothing to check if no results

    # Get the minimum attack_count in the returned results
    min_included_count = min(entry.attack_count for entry in result)

    # Build a set of IPs that are in the result (they are masked, so we check
    # against the internal state of the accumulator instead)
    included_count = len(result)
    all_counts = sorted(
        [data["count"] for data in accumulator._attacker_counts.values()],
        reverse=True,
    )

    # All excluded entries (those beyond the top N) should have counts <= min_included
    excluded_counts = all_counts[included_count:]
    for exc_count in excluded_counts:
        assert exc_count <= min_included_count, (
            f"Excluded entry with count {exc_count} is greater than "
            f"minimum included count {min_included_count}"
        )


@settings(max_examples=100)
@given(attacker_data=attacker_data_strategy)
def test_top_attackers_ranks_start_from_1_and_increment(
    attacker_data: list[tuple[str, int]],
):
    """Property 15: All returned entries have rank starting from 1 and incrementing.

    The rank field SHALL start at 1 for the first entry and increment by 1 for
    each subsequent entry.

    **Validates: Requirements 14.4**
    """
    accumulator = _create_stats_accumulator()
    asyncio.run(_populate_accumulator(accumulator, attacker_data))

    result = accumulator.get_top_attackers()

    for i, entry in enumerate(result):
        expected_rank = i + 1
        assert entry.rank == expected_rank, (
            f"Expected rank {expected_rank} at position {i}, got {entry.rank}"
        )


@settings(max_examples=100)
@given(attacker_data=attacker_data_strategy)
def test_top_attackers_ip_addresses_are_masked(
    attacker_data: list[tuple[str, int]],
):
    """Property 15: IP addresses in the result are masked (contain "x.x").

    Each returned entry's ip_address SHALL be partially masked with the last
    two octets replaced by "x.x".

    **Validates: Requirements 14.4**
    """
    accumulator = _create_stats_accumulator()
    asyncio.run(_populate_accumulator(accumulator, attacker_data))

    result = accumulator.get_top_attackers()

    for entry in result:
        assert "x.x" in entry.ip_address, (
            f"IP '{entry.ip_address}' is not masked (expected 'x.x' in last two octets)"
        )
        # Verify the masked IP has the format "A.B.x.x"
        parts = entry.ip_address.split(".")
        assert len(parts) == 4, (
            f"Masked IP '{entry.ip_address}' does not have 4 parts"
        )
        assert parts[2] == "x" and parts[3] == "x", (
            f"Masked IP '{entry.ip_address}' last two octets are not 'x.x'"
        )
