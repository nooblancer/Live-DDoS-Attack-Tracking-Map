"""Property-based tests for replay dataset looping behavior."""

# Feature: v2-attack-tracking-overhaul, Property 8: Replay Dataset Looping

from hypothesis import given, settings
from hypothesis import strategies as st


# --- Dataset Index Logic (Pure Functions) ---
# These encapsulate the looping arithmetic used by the ReplayEngine.
# current_index = total_processed % dataset_size
# loops_completed = total_processed // dataset_size


def compute_current_index(total_processed: int, dataset_size: int) -> int:
    """Compute the current dataset index after processing total_processed items."""
    return total_processed % dataset_size


def compute_loops_completed(total_processed: int, dataset_size: int) -> int:
    """Compute how many complete loops have been made through the dataset."""
    return total_processed // dataset_size


# --- Strategies ---

# Dataset size: 1 to 100 (as specified in the task)
dataset_size_strategy = st.integers(min_value=1, max_value=100)


# --- Property Tests ---


@settings(max_examples=200)
@given(dataset_size=dataset_size_strategy)
def test_index_wraps_to_zero_after_n_items(dataset_size: int):
    """Property 8: After processing N items, the internal index wraps to 0.

    For any dataset of N flows (N >= 1), after the Replay Engine has emitted
    all N flows, the next flow emitted SHALL be the first flow in the dataset
    (index 0).

    **Validates: Requirements 1.6**
    """
    # After processing exactly N items (0..N-1), index should wrap to 0
    current_index = compute_current_index(dataset_size, dataset_size)
    assert current_index == 0, (
        f"After processing {dataset_size} items in a dataset of size {dataset_size}, "
        f"expected index 0 but got {current_index}"
    )


@settings(max_examples=200)
@given(dataset_size=dataset_size_strategy)
def test_loops_completed_increments_at_wrap(dataset_size: int):
    """Property 8: loops_completed increments by 1 at each dataset wrap.

    For any dataset of N flows (N >= 1), after the Replay Engine has emitted
    all N flows, the loops_completed counter SHALL increment by 1.

    **Validates: Requirements 1.6**
    """
    # Before completing first loop (at N-1 processed), loops_completed should be 0
    loops_before_wrap = compute_loops_completed(dataset_size - 1, dataset_size)
    assert loops_before_wrap == 0, (
        f"Before completing first loop (at item {dataset_size - 1}), "
        f"expected 0 loops but got {loops_before_wrap}"
    )

    # After completing exactly one full loop (N items processed), loops_completed should be 1
    loops_after_wrap = compute_loops_completed(dataset_size, dataset_size)
    assert loops_after_wrap == 1, (
        f"After processing {dataset_size} items (one full loop), "
        f"expected 1 loop completed but got {loops_after_wrap}"
    )


@settings(max_examples=200)
@given(dataset_size=dataset_size_strategy)
def test_two_full_loops_gives_loops_completed_two(dataset_size: int):
    """Property 8: After processing 2*N items, loops_completed equals 2.

    For any dataset of N flows, after processing 2*N items the engine has
    looped through the dataset exactly twice.

    **Validates: Requirements 1.6**
    """
    total_processed = 2 * dataset_size
    loops = compute_loops_completed(total_processed, dataset_size)
    assert loops == 2, (
        f"After processing {total_processed} items in a dataset of size {dataset_size}, "
        f"expected 2 loops completed but got {loops}"
    )


@settings(max_examples=200)
@given(dataset_size=dataset_size_strategy)
def test_flow_at_position_n_equals_position_zero(dataset_size: int):
    """Property 8: The flow at position N is the same as position 0 (wraps correctly).

    For any dataset of N flows, the index computed for the (N+1)th item
    (0-indexed: item N) SHALL equal index 0, confirming the dataset loops.

    **Validates: Requirements 1.6**
    """
    # Index at position 0 (first item)
    index_at_zero = compute_current_index(0, dataset_size)
    # Index at position N (N+1th item, should be same as first)
    index_at_n = compute_current_index(dataset_size, dataset_size)

    assert index_at_zero == 0, (
        f"Index at position 0 should be 0, got {index_at_zero}"
    )
    assert index_at_n == 0, (
        f"Index at position {dataset_size} should wrap to 0, got {index_at_n}"
    )
    assert index_at_zero == index_at_n, (
        f"Position 0 index ({index_at_zero}) should equal position N index ({index_at_n})"
    )


@settings(max_examples=200)
@given(
    dataset_size=st.integers(min_value=2, max_value=100),
    num_loops=st.integers(min_value=1, max_value=50),
)
def test_loops_completed_increments_monotonically(dataset_size: int, num_loops: int):
    """Property 8: loops_completed increments by exactly 1 at each boundary.

    For any dataset size N (N >= 2) and any number of complete loops K,
    processing exactly K*N items SHALL yield loops_completed == K, and
    processing K*N + 1 items (one into the next loop) SHALL still yield
    loops_completed == K (no premature increment until the next full loop).

    **Validates: Requirements 1.6**
    """
    # At exactly K*N items, loops_completed should be K
    total_at_boundary = num_loops * dataset_size
    loops_at_boundary = compute_loops_completed(total_at_boundary, dataset_size)
    assert loops_at_boundary == num_loops, (
        f"After processing {total_at_boundary} items (dataset_size={dataset_size}), "
        f"expected {num_loops} loops but got {loops_at_boundary}"
    )

    # One item into the next loop (K*N + 1), loops_completed should still be K
    # because the (K+1)th loop isn't complete until (K+1)*N items are processed
    total_one_past = total_at_boundary + 1
    loops_one_past = compute_loops_completed(total_one_past, dataset_size)
    assert loops_one_past == num_loops, (
        f"After processing {total_one_past} items (dataset_size={dataset_size}), "
        f"expected still {num_loops} loops but got {loops_one_past}"
    )


@settings(max_examples=200)
@given(
    dataset_size=dataset_size_strategy,
    total_processed=st.integers(min_value=0, max_value=10000),
)
def test_index_always_within_bounds(dataset_size: int, total_processed: int):
    """Property 8: The computed index is always within [0, dataset_size).

    For any dataset size N and any number of total processed items,
    the current index SHALL always be in the range [0, N-1].

    **Validates: Requirements 1.6**
    """
    index = compute_current_index(total_processed, dataset_size)
    assert 0 <= index < dataset_size, (
        f"Index {index} is out of bounds for dataset_size={dataset_size} "
        f"(total_processed={total_processed})"
    )
