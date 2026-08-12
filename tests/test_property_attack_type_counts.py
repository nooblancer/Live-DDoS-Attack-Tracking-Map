"""Property-based tests for attack type counts sum to total."""

# Feature: v2-attack-tracking-overhaul, Property 16: Attack Type Counts Sum to Total

import pytest
from collections import Counter
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from models.schemas import AttackType, ClassificationResult
from services.database import DatabaseService
from services.stats_accumulator import StatsAccumulator


# --- Strategies ---

# The 12 valid attack type strings (NOT "Benign" or "UNKNOWN")
VALID_ATTACK_TYPES = [at.value for at in AttackType]

# Strategy for a single valid attack type
attack_type_strategy = st.sampled_from(VALID_ATTACK_TYPES)

# Strategy for a source IP address
source_ip_strategy = st.from_regex(
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", fullmatch=True
).filter(lambda ip: all(0 <= int(part) <= 255 for part in ip.split(".")))

# Strategy for a list of (attack_type, source_ip) pairs
attack_type_ip_pairs_strategy = st.lists(
    st.tuples(attack_type_strategy, source_ip_strategy),
    min_size=1,
    max_size=50,
)


def _make_classification_result(attack_type: str) -> ClassificationResult:
    """Create a ClassificationResult with the given attack type."""
    return ClassificationResult(
        attack_type=attack_type,
        confidence=0.85,
        classified_in_ms=1.2,
        top_features=["feature_a", "feature_b", "feature_c"],
    )


def _make_stats_accumulator() -> StatsAccumulator:
    """Create a StatsAccumulator with a mocked DatabaseService."""
    mock_db = MagicMock(spec=DatabaseService)
    mock_db._conn = None
    return StatsAccumulator(db=mock_db)


# --- Property Tests ---


@pytest.mark.asyncio
@settings(max_examples=200)
@given(pairs=attack_type_ip_pairs_strategy)
async def test_attack_type_counts_sum_equals_total_recorded(
    pairs: list[tuple[str, str]],
):
    """Property 16: Sum of all per-type counts equals the number of results recorded.

    For any sequence of N classification results with various attack types,
    the sum of all per-type counts in the attack-types response SHALL equal N.

    **Validates: Requirements 14.5**
    """
    accumulator = _make_stats_accumulator()

    # Record all predictions
    for attack_type, source_ip in pairs:
        result = _make_classification_result(attack_type)
        await accumulator.record_prediction(result, source_ip)

    # Verify: sum of get_attack_types().values() equals total count
    attack_types = accumulator.get_attack_types()
    total_from_types = sum(attack_types.values())

    assert total_from_types == len(pairs), (
        f"Sum of attack type counts ({total_from_types}) does not equal "
        f"number of results recorded ({len(pairs)}). "
        f"Attack types: {attack_types}"
    )


@pytest.mark.asyncio
@settings(max_examples=200)
@given(pairs=attack_type_ip_pairs_strategy)
async def test_individual_attack_type_counts_match_input_frequency(
    pairs: list[tuple[str, str]],
):
    """Property 16: Each individual attack type count matches input frequency.

    For any sequence of classification results, each individual attack type count
    SHALL match the number of times that type appeared in the input sequence.

    **Validates: Requirements 14.5**
    """
    accumulator = _make_stats_accumulator()

    # Record all predictions
    for attack_type, source_ip in pairs:
        result = _make_classification_result(attack_type)
        await accumulator.record_prediction(result, source_ip)

    # Count expected frequencies from input
    expected_counts = Counter(attack_type for attack_type, _ in pairs)

    # Verify: each type's count matches expected
    attack_types = accumulator.get_attack_types()

    for attack_type, expected_count in expected_counts.items():
        actual_count = attack_types.get(attack_type, 0)
        assert actual_count == expected_count, (
            f"Attack type '{attack_type}': expected {expected_count}, "
            f"got {actual_count}. Full breakdown: {attack_types}"
        )

    # Verify no extra types appeared
    for attack_type in attack_types:
        assert attack_type in expected_counts, (
            f"Unexpected attack type '{attack_type}' in results "
            f"(not in input data). Full breakdown: {attack_types}"
        )
