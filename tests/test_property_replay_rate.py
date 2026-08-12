"""Property-based tests for replay rate proportional to speed multiplier."""

# Feature: v2-attack-tracking-overhaul, Property 17: Replay Rate Proportional to Speed Multiplier

from hypothesis import given, settings
from hypothesis import strategies as st


# --- Pure function under test ---


def compute_delay(
    original_delta_ms: float,
    speed_multiplier: int,
    max_events_per_second: float = 50.0,
) -> float:
    """Compute the inter-event delay for the replay engine.

    The delay is the original timestamp delta divided by the speed multiplier,
    but floored at 1/max_events_per_second to prevent overwhelming the frontend.

    Args:
        original_delta_ms: Time between consecutive flows in milliseconds.
        speed_multiplier: Replay speed factor (1, 10, 100, or 1000).
        max_events_per_second: Maximum event emission rate (default 50).

    Returns:
        Delay in seconds.
    """
    # Convert ms to seconds and divide by speed
    raw_delay = (original_delta_ms / 1000.0) / speed_multiplier
    # Apply minimum delay floor
    min_delay = 1.0 / max_events_per_second
    return max(raw_delay, min_delay)


# --- Strategies ---

# Speed multipliers as specified in the requirements
speed_multiplier_strategy = st.sampled_from([1, 10, 100, 1000])

# Positive deltas in milliseconds (realistic range: 1ms to 10 seconds)
delta_ms_strategy = st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False)

# Max events per second (reasonable range)
max_eps_strategy = st.floats(min_value=10.0, max_value=200.0, allow_nan=False, allow_infinity=False)


# --- Property Tests ---


@settings(max_examples=200)
@given(
    speed_multiplier=speed_multiplier_strategy,
    delta_ms=delta_ms_strategy,
)
def test_computed_delay_equals_delta_over_speed_or_floor(
    speed_multiplier: int,
    delta_ms: float,
):
    """Property 17: Computed delay equals D/S or the min_delay floor, whichever is larger.

    For any speed multiplier S in {1, 10, 100, 1000} and any positive delta D,
    the computed delay equals D/S (or the min_delay floor, whichever is larger).

    **Validates: Requirements 1.1, 1.2**
    """
    max_eps = 50.0
    min_delay = 1.0 / max_eps
    expected_raw_delay = (delta_ms / 1000.0) / speed_multiplier
    expected_delay = max(expected_raw_delay, min_delay)

    actual_delay = compute_delay(delta_ms, speed_multiplier, max_eps)

    assert actual_delay == expected_delay, (
        f"Expected delay={expected_delay:.6f}s for delta={delta_ms}ms, "
        f"speed={speed_multiplier}x, but got {actual_delay:.6f}s"
    )


@settings(max_examples=200)
@given(
    speed_multiplier=speed_multiplier_strategy,
    delta_ms=delta_ms_strategy,
    max_eps=max_eps_strategy,
)
def test_delay_always_at_least_min_delay(
    speed_multiplier: int,
    delta_ms: float,
    max_eps: float,
):
    """Property 17: The delay is always >= min_delay (1/max_events_per_second).

    For any speed multiplier and any positive delta, the computed delay SHALL
    never be less than 1/max_events_per_second.

    **Validates: Requirements 1.1, 1.2**
    """
    min_delay = 1.0 / max_eps
    actual_delay = compute_delay(delta_ms, speed_multiplier, max_eps)

    assert actual_delay >= min_delay - 1e-12, (
        f"Delay {actual_delay:.8f}s is below min_delay {min_delay:.8f}s "
        f"(delta={delta_ms}ms, speed={speed_multiplier}x, max_eps={max_eps})"
    )


@settings(max_examples=200)
@given(
    delta_ms=delta_ms_strategy,
)
def test_higher_speed_produces_shorter_or_equal_delay(
    delta_ms: float,
):
    """Property 17: Higher speed multipliers produce shorter delays for the same delta.

    For any positive delta D, if S1 < S2 then delay(S1) >= delay(S2).
    The delay is monotonically non-increasing with speed multiplier.

    **Validates: Requirements 1.1, 1.2**
    """
    speeds = [1, 10, 100, 1000]
    max_eps = 50.0

    delays = [compute_delay(delta_ms, s, max_eps) for s in speeds]

    for i in range(len(speeds) - 1):
        assert delays[i] >= delays[i + 1], (
            f"Delay at {speeds[i]}x ({delays[i]:.6f}s) is less than "
            f"delay at {speeds[i+1]}x ({delays[i+1]:.6f}s) for delta={delta_ms}ms"
        )


@settings(max_examples=200)
@given(
    delta_ms=st.floats(min_value=200.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
)
def test_proportional_relationship_when_not_floored(
    delta_ms: float,
):
    """Property 17: delay(S=10) ≈ 10 * delay(S=100) when neither is floored.

    For deltas large enough that neither speed is floored by min_delay,
    the relationship is proportional: delay(S=10) = 10 * delay(S=100).

    **Validates: Requirements 1.1, 1.2**
    """
    max_eps = 50.0
    min_delay = 1.0 / max_eps

    delay_10 = compute_delay(delta_ms, 10, max_eps)
    delay_100 = compute_delay(delta_ms, 100, max_eps)

    # Only test proportionality when neither is floored
    raw_delay_10 = (delta_ms / 1000.0) / 10
    raw_delay_100 = (delta_ms / 1000.0) / 100

    if raw_delay_10 > min_delay and raw_delay_100 > min_delay:
        # Both are above floor, so proportional relationship holds exactly
        assert abs(delay_10 - 10 * delay_100) < 1e-10, (
            f"Proportional relationship violated: delay(10x)={delay_10:.8f}s "
            f"!= 10 * delay(100x)={10 * delay_100:.8f}s for delta={delta_ms}ms"
        )
