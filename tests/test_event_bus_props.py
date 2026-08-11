# Feature: ddos-attack-tracking-map, Property 15: EventBus fanout delivers to all subscribers and disconnect is isolated
"""Property-based tests for EventBus fanout delivery and disconnect isolation.

Validates: Requirements 9.2, 9.4
"""

import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from services.event_bus import EventBus

# --- Strategies ---

# Generate a random number of subscribers (1-10)
subscriber_count_st = st.integers(min_value=1, max_value=10)

# Generate random event dicts with arbitrary string keys and values
event_st = st.fixed_dictionaries(
    {
        "ip_address": st.text(min_size=1, max_size=15),
        "latitude": st.floats(min_value=-90, max_value=90, allow_nan=False),
        "longitude": st.floats(min_value=-180, max_value=180, allow_nan=False),
    },
    optional={
        "country": st.text(min_size=1, max_size=30),
        "city": st.text(min_size=1, max_size=30),
    },
)


# --- Property 15 Part 1: Fanout delivers to all subscribers ---


@settings(max_examples=100, deadline=None)
@given(n_subscribers=subscriber_count_st, event=event_st)
def test_fanout_delivers_to_all_subscribers(
    n_subscribers: int, event: dict
) -> None:
    """For any set of N subscribers and any published event, all N subscribers
    shall receive the event.

    **Validates: Requirements 9.2**
    """

    async def _run():
        bus = EventBus()

        # Subscribe N clients
        queues = [bus.subscribe() for _ in range(n_subscribers)]

        # Publish the event
        await bus.publish(event)

        # All N subscribers must have received exactly the event
        for i, queue in enumerate(queues):
            assert not queue.empty(), f"Subscriber {i} queue is empty after publish"
            received = queue.get_nowait()
            assert received == event, (
                f"Subscriber {i} received {received!r}, expected {event!r}"
            )
            # No extra events in queue
            assert queue.empty(), f"Subscriber {i} has extra events in queue"

    asyncio.run(_run())


# --- Property 15 Part 2: Disconnect isolation ---


@settings(max_examples=100, deadline=None)
@given(n_subscribers=st.integers(min_value=2, max_value=10), event=event_st)
def test_disconnect_isolation(n_subscribers: int, event: dict) -> None:
    """After one subscriber disconnects, the remaining N-1 subscribers shall
    continue receiving subsequent events.

    **Validates: Requirements 9.4**
    """

    async def _run():
        bus = EventBus()

        # Subscribe N clients
        queues = [bus.subscribe() for _ in range(n_subscribers)]

        # Disconnect the first subscriber
        disconnected_queue = queues[0]
        bus.unsubscribe(disconnected_queue)

        # Publish an event after disconnection
        await bus.publish(event)

        # The disconnected subscriber should NOT receive the event
        assert disconnected_queue.empty(), (
            "Disconnected subscriber still received an event"
        )

        # All remaining N-1 subscribers must receive the event
        for i, queue in enumerate(queues[1:], start=1):
            assert not queue.empty(), (
                f"Remaining subscriber {i} queue is empty after publish"
            )
            received = queue.get_nowait()
            assert received == event, (
                f"Remaining subscriber {i} received {received!r}, expected {event!r}"
            )
            # No extra events
            assert queue.empty(), (
                f"Remaining subscriber {i} has extra events in queue"
            )

    asyncio.run(_run())
