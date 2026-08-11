"""Unit tests for EventBus pub/sub system."""

import asyncio

import pytest

from services.event_bus import EventBus


@pytest.fixture
def bus() -> EventBus:
    """Create a fresh EventBus instance."""
    return EventBus()


class TestSubscribe:
    """Tests for subscribe()."""

    def test_subscribe_returns_queue(self, bus: EventBus):
        queue = bus.subscribe()
        assert isinstance(queue, asyncio.Queue)

    def test_subscribe_adds_to_subscribers(self, bus: EventBus):
        bus.subscribe()
        bus.subscribe()
        assert len(bus._subscribers) == 2

    def test_each_subscriber_gets_own_queue(self, bus: EventBus):
        q1 = bus.subscribe()
        q2 = bus.subscribe()
        assert q1 is not q2


class TestUnsubscribe:
    """Tests for unsubscribe()."""

    def test_unsubscribe_removes_queue(self, bus: EventBus):
        queue = bus.subscribe()
        bus.unsubscribe(queue)
        assert len(bus._subscribers) == 0

    def test_unsubscribe_unknown_queue_does_not_raise(self, bus: EventBus):
        unknown_queue = asyncio.Queue()
        # Should not raise
        bus.unsubscribe(unknown_queue)

    def test_unsubscribe_only_removes_target(self, bus: EventBus):
        q1 = bus.subscribe()
        q2 = bus.subscribe()
        bus.unsubscribe(q1)
        assert bus._subscribers == [q2]


class TestPublish:
    """Tests for publish()."""

    @pytest.mark.asyncio
    async def test_publish_delivers_to_single_subscriber(self, bus: EventBus):
        queue = bus.subscribe()
        event = {"ip_address": "1.2.3.4", "latitude": 10.0, "longitude": 20.0}
        await bus.publish(event)
        received = queue.get_nowait()
        assert received == event

    @pytest.mark.asyncio
    async def test_publish_delivers_to_all_subscribers(self, bus: EventBus):
        q1 = bus.subscribe()
        q2 = bus.subscribe()
        q3 = bus.subscribe()
        event = {"type": "attack", "ip": "5.6.7.8"}
        await bus.publish(event)
        assert q1.get_nowait() == event
        assert q2.get_nowait() == event
        assert q3.get_nowait() == event

    @pytest.mark.asyncio
    async def test_publish_to_empty_subscribers_is_noop(self, bus: EventBus):
        # Should not raise even with no subscribers
        await bus.publish({"test": True})

    @pytest.mark.asyncio
    async def test_publish_drops_oldest_when_queue_full(self):
        bus = EventBus(maxsize=2)
        queue = bus.subscribe()
        await bus.publish({"seq": 1})
        await bus.publish({"seq": 2})
        # Queue is now full
        await bus.publish({"seq": 3})
        # Oldest (seq=1) should be dropped
        first = queue.get_nowait()
        second = queue.get_nowait()
        assert first == {"seq": 2}
        assert second == {"seq": 3}

    @pytest.mark.asyncio
    async def test_unsubscribed_client_does_not_receive_events(self, bus: EventBus):
        q1 = bus.subscribe()
        q2 = bus.subscribe()
        bus.unsubscribe(q1)
        await bus.publish({"event": "data"})
        assert q1.empty()
        assert q2.get_nowait() == {"event": "data"}
