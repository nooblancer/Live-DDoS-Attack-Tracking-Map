"""SSE events route for real-time attack event streaming."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from services.event_bus import EventBus

logger = logging.getLogger(__name__)

router = APIRouter()

# Heartbeat interval in seconds
_HEARTBEAT_INTERVAL = 30


async def _event_generator(
    request: Request, event_bus: EventBus
) -> AsyncGenerator[dict, None]:
    """Async generator that yields SSE events from the EventBus.

    Subscribes to the bus, waits for events with a 30s timeout,
    and yields heartbeat comments when idle. Cleans up on disconnect.
    """
    queue = event_bus.subscribe()
    logger.debug("SSE client connected, subscribed to event bus.")
    try:
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            try:
                # Use asyncio.wait with a timeout to avoid CancelledError
                # propagation issues with asyncio.wait_for in generators
                get_task = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait(
                    {get_task}, timeout=_HEARTBEAT_INTERVAL
                )
                if get_task in done:
                    event = get_task.result()
                    # Yield attack event as JSON
                    yield {
                        "event": "attack",
                        "data": json.dumps(event),
                    }
                else:
                    # Timeout — cancel pending get and send heartbeat
                    get_task.cancel()
                    try:
                        await get_task
                    except asyncio.CancelledError:
                        pass
                    yield {"comment": "heartbeat"}
            except asyncio.CancelledError:
                raise
    except asyncio.CancelledError:
        logger.debug("SSE client disconnected (CancelledError).")
    finally:
        event_bus.unsubscribe(queue)
        logger.debug("SSE client unsubscribed and cleaned up.")


@router.get("/events")
async def events_endpoint(request: Request) -> EventSourceResponse:
    """GET /events — SSE stream.

    Sends heartbeat comment every 30s.
    Broadcasts new attack events as JSON.
    Cleans up on client disconnect.
    """
    # EventBus is attached to app state by main.py at startup
    event_bus: EventBus = request.app.state.event_bus
    return EventSourceResponse(_event_generator(request, event_bus))
