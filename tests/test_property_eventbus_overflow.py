"""Property-based tests for EventBus overflow drops oldest behavior."""

# Feature: v2-attack-tracking-overhaul, Property 6: EventBus Overflow Drops Oldest

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from services.event_bus import EventBus


# --- Strategies ---

# Queue capacity: small values to keep tests fast but meaningful
capacity_strategy = st.integers(min_value=5, max_value=50)

# Number of events to publish, always greater than capacity.
# We generate a ratio > 1 and multiply by capacity later.
overflow_ratio_strategy = st.integers(min_value=2, max_value=10)


# --- Helpers ---


def _make_event(index: int) -> dict:
    """Create a simple numbered event dict."""
    return {"id": index, "type": "test_event", "data": f"event_{index}"}


async def _publish_n_events(bus: EventBus, n: int) -> None:
    """Publish n events to the bus sequentially."""
    for i in range(n):
        await bus.publish(_make_event(i))


def _drain_queue(queue: asyncio.Queue) -> list[dict]:
    """Drain all items from a queue into a list (non-blocking)."""
    items = []
    while not queue.empty():
        try:
            items.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items


# --- Property Tests ---


@settings(max_examples=100)
@given(
    capacity=capacity_strategy,
    overflow_ratio=overflow_ratio_strategy,
)
def test_overflow_queue_contains_exactly_capacity_items(
    capacity: int,
    overflow_ratio: int,
):
    """Property 6: After N > C publishes without consumption, queue contains exactly C items.

    For any subscriber queue of capacity C, when N > C events are published
    without the subscriber consuming, the queue SHALL contain exactly C events.

    **Validates: Requirements 4.4**
    """
    n = capacity * overflow_ratio  # N > C guaranteed since ratio >= 2

    bus = EventBus(maxsize=capacity)
    queue = bus.subscribe()

    asyncio.run(_publish_n_events(bus, n))

    assert queue.qsize() == capacity, (
        f"Expected queue to contain exactly {capacity} items after "
        f"publishing {n} events, but got {queue.qsize()}"
    )


@settings(max_examples=100)
@given(
    capacity=capacity_strategy,
    overflow_ratio=overflow_ratio_strategy,
)
def test_overflow_queue_contains_most_recent_events(
    capacity: int,
    overflow_ratio: int,
):
    """Property 6: Queue contains the most recently published C events.

    For any subscriber queue of capacity C, when N > C events are published
    without the subscriber consuming, the queue SHALL contain exactly C events
    corresponding to the most recently published C events.

    **Validates: Requirements 4.4**
    """
    n = capacity * overflow_ratio  # N > C guaranteed

    bus = EventBus(maxsize=capacity)
    queue = bus.subscribe()

    asyncio.run(_publish_n_events(bus, n))

    # Drain the queue and verify contents
    items = _drain_queue(queue)

    assert len(items) == capacity, (
        f"Expected {capacity} items in queue, got {len(items)}"
    )

    # The items should be the last C events published (indices n-C to n-1)
    expected_ids = list(range(n - capacity, n))
    actual_ids = [item["id"] for item in items]

    assert actual_ids == expected_ids, (
        f"Queue does not contain the most recent {capacity} events. "
        f"Expected event IDs {expected_ids}, got {actual_ids}"
    )


@settings(max_examples=100)
@given(
    capacity=capacity_strategy,
    overflow_ratio=overflow_ratio_strategy,
)
def test_overflow_publish_never_blocks(
    capacity: int,
    overflow_ratio: int,
):
    """Property 6: The publish operation completes without blocking.

    For any subscriber queue of capacity C, when N > C events are published
    without the subscriber consuming, the publish operation SHALL never block
    (completes within a reasonable timeout).

    **Validates: Requirements 4.4**
    """
    n = capacity * overflow_ratio  # N > C guaranteed

    bus = EventBus(maxsize=capacity)
    _queue = bus.subscribe()

    async def publish_with_timeout():
        """Publish all events with a timeout to detect blocking."""
        try:
            await asyncio.wait_for(_publish_n_events(bus, n), timeout=5.0)
            return True
        except asyncio.TimeoutError:
            return False

    did_complete = asyncio.run(publish_with_timeout())

    assert did_complete, (
        f"Publish blocked when queue was full! Published {n} events to "
        f"a queue of capacity {capacity} and it did not complete in 5 seconds."
    )
