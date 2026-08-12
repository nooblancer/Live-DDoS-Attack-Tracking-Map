"""Property-based tests for timeline bucketing correctness."""

# Feature: v2-attack-tracking-overhaul, Property 14: Timeline Bucketing Correctness

from datetime import datetime, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from services.timeline_bucketing import bucket_events


# --- Strategies ---

# Strategy: a base datetime within a reasonable range
_base_datetime_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
)


def _timestamps_within_window(base: datetime, window_minutes: int = 60):
    """Strategy that generates a list of timestamps within window_minutes of base."""
    return st.lists(
        st.integers(min_value=0, max_value=(window_minutes * 60) - 1).map(
            lambda offset_secs: base + timedelta(seconds=offset_secs)
        ),
        min_size=0,
        max_size=200,
    )


# Composite strategy: generate a base datetime and then timestamps within the window
@st.composite
def timestamps_in_window(draw, window_minutes: int = 60):
    """Draw a list of timestamps all within a 60-minute window."""
    base = draw(_base_datetime_strategy)
    # Ensure base has no sub-second precision for predictability
    base = base.replace(microsecond=0)
    ts_list = draw(_timestamps_within_window(base, window_minutes))
    return ts_list


# --- Property Tests ---


@settings(max_examples=200)
@given(timestamps=timestamps_in_window())
def test_at_most_60_buckets_for_60_minute_window(timestamps: list[datetime]):
    """Property 14: At most 60 buckets for a 60-minute window.

    For any set of timestamped events within a 60-minute window, the bucketing
    function SHALL produce at most 60 buckets (one per minute).

    **Validates: Requirements 11.3**
    """
    result = bucket_events(timestamps, window_minutes=60)

    assert len(result) <= 60, (
        f"Expected at most 60 buckets for a 60-minute window, "
        f"got {len(result)} buckets. Keys: {list(result.keys())}"
    )


@settings(max_examples=200)
@given(timestamps=timestamps_in_window())
def test_sum_of_bucket_counts_equals_total_events(timestamps: list[datetime]):
    """Property 14: Sum of all bucket counts equals total number of input events.

    Each event is counted in exactly one bucket, so the sum of all bucket counts
    SHALL equal the total number of input events.

    **Validates: Requirements 11.3**
    """
    result = bucket_events(timestamps, window_minutes=60)
    total_in_buckets = sum(result.values())

    assert total_in_buckets == len(timestamps), (
        f"Sum of bucket counts ({total_in_buckets}) does not equal "
        f"total number of input events ({len(timestamps)}). "
        f"Buckets: {result}"
    )


@settings(max_examples=200)
@given(timestamps=timestamps_in_window())
def test_events_with_same_minute_land_in_same_bucket(timestamps: list[datetime]):
    """Property 14: Events with the same minute timestamp land in the same bucket.

    Two events that share the same minute (after truncating seconds/microseconds)
    SHALL be counted in the same bucket.

    **Validates: Requirements 11.3**
    """
    result = bucket_events(timestamps, window_minutes=60)

    # Verify by manually grouping and checking counts match
    expected: dict[str, int] = {}
    for ts in timestamps:
        minute_key = ts.replace(second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")
        expected[minute_key] = expected.get(minute_key, 0) + 1

    for key, count in expected.items():
        assert key in result, (
            f"Expected bucket key '{key}' missing from result. "
            f"Result keys: {list(result.keys())}"
        )
        assert result[key] == count, (
            f"Bucket '{key}': expected count {count}, got {result[key]}"
        )


@settings(max_examples=200)
@given(timestamps=timestamps_in_window())
def test_buckets_are_sorted_chronologically(timestamps: list[datetime]):
    """Property 14: Buckets are sorted chronologically.

    The bucketing function SHALL return buckets in chronological order
    (keys sorted lexicographically, which corresponds to time order for ISO strings).

    **Validates: Requirements 11.3**
    """
    result = bucket_events(timestamps, window_minutes=60)
    keys = list(result.keys())

    assert keys == sorted(keys), (
        f"Bucket keys are not in chronological order. "
        f"Got: {keys}, Expected: {sorted(keys)}"
    )
