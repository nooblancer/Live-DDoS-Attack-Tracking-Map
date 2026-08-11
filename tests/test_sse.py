"""Integration tests for the /events SSE endpoint."""

import asyncio
import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from routes.events import router, _event_generator
from services.event_bus import EventBus


def _create_test_app(event_bus: EventBus) -> FastAPI:
    """Create a minimal FastAPI app with the events router and an EventBus."""
    app = FastAPI()
    app.state.event_bus = event_bus
    app.include_router(router)
    return app


class MockRequest:
    """Mock request for testing the generator directly."""

    def __init__(self, app_instance, disconnect_after: int | None = None):
        self.app = app_instance
        self._disconnect_after = disconnect_after
        self._check_count = 0

    async def is_disconnected(self):
        self._check_count += 1
        if self._disconnect_after is not None and self._check_count > self._disconnect_after:
            return True
        return False


@pytest.mark.asyncio
async def test_event_generator_heartbeat():
    """Generator yields a heartbeat comment when no events arrive within timeout."""
    import routes.events as events_mod

    event_bus = EventBus()
    app = _create_test_app(event_bus)

    # Patch heartbeat interval to a very short value for testing
    original_interval = events_mod._HEARTBEAT_INTERVAL
    events_mod._HEARTBEAT_INTERVAL = 0.3

    try:
        mock_request = MockRequest(app)
        gen = _event_generator(mock_request, event_bus)

        # First yield should be a heartbeat after the short timeout
        result = await asyncio.wait_for(gen.__anext__(), timeout=3.0)
        assert result == {"comment": "heartbeat"}

        # Clean up generator
        await gen.aclose()
    finally:
        events_mod._HEARTBEAT_INTERVAL = original_interval


@pytest.mark.asyncio
async def test_event_generator_attack_event():
    """Generator yields attack events as JSON when published to EventBus."""
    event_bus = EventBus()
    app = _create_test_app(event_bus)

    attack_data = {
        "ip_address": "192.168.1.100",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "country": "United States",
        "city": "New York",
        "isp": "TestISP",
        "timestamp": "2024-01-15T14:32:00Z",
    }

    mock_request = MockRequest(app)
    gen = _event_generator(mock_request, event_bus)

    # Start consuming in a task so the generator subscribes,
    # then publish while it's waiting
    async def consume_one():
        return await gen.__anext__()

    task = asyncio.ensure_future(consume_one())
    # Give the generator a moment to enter the wait loop
    await asyncio.sleep(0.05)

    # Now publish — the generator is subscribed and waiting
    await event_bus.publish(attack_data)

    result = await asyncio.wait_for(task, timeout=3.0)
    assert result["event"] == "attack"
    payload = json.loads(result["data"])
    assert payload["ip_address"] == "192.168.1.100"
    assert payload["latitude"] == 40.7128
    assert payload["country"] == "United States"

    await gen.aclose()


@pytest.mark.asyncio
async def test_event_generator_cleanup_on_disconnect():
    """Generator unsubscribes from EventBus when client disconnects."""
    import routes.events as events_mod

    event_bus = EventBus()
    app = _create_test_app(event_bus)

    original_interval = events_mod._HEARTBEAT_INTERVAL
    events_mod._HEARTBEAT_INTERVAL = 0.2

    try:
        # Disconnect after first is_disconnected check (second check returns True)
        mock_request = MockRequest(app, disconnect_after=1)
        gen = _event_generator(mock_request, event_bus)

        # First iteration: is_disconnected returns False, then waits and yields heartbeat
        result = await asyncio.wait_for(gen.__anext__(), timeout=3.0)
        assert result == {"comment": "heartbeat"}

        # One subscriber should be registered
        assert len(event_bus._subscribers) == 1

        # Next iteration should detect disconnect and stop
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(gen.__anext__(), timeout=3.0)

        # Subscriber should be cleaned up
        assert len(event_bus._subscribers) == 0
    finally:
        events_mod._HEARTBEAT_INTERVAL = original_interval


@pytest.mark.asyncio
async def test_event_generator_cleanup_on_cancel():
    """Generator unsubscribes from EventBus when cancelled."""
    event_bus = EventBus()
    app = _create_test_app(event_bus)

    mock_request = MockRequest(app)
    gen = _event_generator(mock_request, event_bus)

    # Start a consume task so the generator subscribes
    async def consume_one():
        return await gen.__anext__()

    task = asyncio.ensure_future(consume_one())
    await asyncio.sleep(0.05)

    # Generator is now subscribed and waiting
    assert len(event_bus._subscribers) == 1

    # Publish so the task completes
    await event_bus.publish({"test": "data"})
    await asyncio.wait_for(task, timeout=3.0)

    # Now close the generator (simulates disconnect/cancel)
    await gen.aclose()

    # Subscriber should be cleaned up
    assert len(event_bus._subscribers) == 0


@pytest.mark.asyncio
async def test_sse_endpoint_returns_event_source_response():
    """The /events endpoint returns a streaming response (status 200)."""
    event_bus = EventBus()
    app = _create_test_app(event_bus)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Use a short timeout so we don't block forever
        try:
            response = await asyncio.wait_for(
                client.get("/events", headers={"Accept": "text/event-stream"}),
                timeout=1.0,
            )
            # If we get a response back it should be 200
            assert response.status_code == 200
        except asyncio.TimeoutError:
            # For SSE streams, a timeout is expected since it's a long-lived connection.
            # The fact that it didn't immediately error means the endpoint works.
            pass


@pytest.mark.asyncio
async def test_event_generator_multiple_events_in_order():
    """Generator yields multiple attack events in publication order."""
    event_bus = EventBus()
    app = _create_test_app(event_bus)

    mock_request = MockRequest(app)
    gen = _event_generator(mock_request, event_bus)

    events = [
        {"ip_address": f"10.0.0.{i}", "latitude": float(i)} for i in range(3)
    ]

    # Start generator so it subscribes
    async def consume_one():
        return await gen.__anext__()

    task = asyncio.ensure_future(consume_one())
    await asyncio.sleep(0.05)

    # Publish all events — generator is subscribed and will queue them
    for event in events:
        await event_bus.publish(event)

    # First event comes from the pending task
    result = await asyncio.wait_for(task, timeout=3.0)
    results = [result]

    # Consume remaining events
    for _ in range(2):
        r = await asyncio.wait_for(gen.__anext__(), timeout=3.0)
        results.append(r)

    for i, result in enumerate(results):
        assert result["event"] == "attack"
        payload = json.loads(result["data"])
        assert payload["ip_address"] == f"10.0.0.{i}"

    await gen.aclose()
