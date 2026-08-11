"""In-process async pub/sub for broadcasting attack events to SSE clients."""

import asyncio
import logging

logger = logging.getLogger(__name__)

# Default max size for subscriber queues. When full, oldest event is dropped.
_DEFAULT_QUEUE_MAXSIZE = 128


class EventBus:
    """In-process async pub/sub for broadcasting attack events to SSE clients."""

    def __init__(self, maxsize: int = _DEFAULT_QUEUE_MAXSIZE):
        self._subscribers: list[asyncio.Queue] = []
        self._maxsize = maxsize

    def subscribe(self) -> asyncio.Queue:
        """Register a new subscriber, return their queue."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._subscribers.append(queue)
        logger.debug("New subscriber added. Total subscribers: %d", len(self._subscribers))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove subscriber and clean up."""
        try:
            self._subscribers.remove(queue)
            logger.debug("Subscriber removed. Total subscribers: %d", len(self._subscribers))
        except ValueError:
            # Queue was already removed or never registered
            pass

    async def publish(self, event: dict) -> None:
        """Push event to all subscriber queues.

        If a subscriber's queue is full, the oldest event is dropped to make
        room for the new one, and a warning is logged.
        """
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest event and push the new one
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass
                logger.warning(
                    "Subscriber queue full — dropped oldest event. "
                    "Queue size: %d",
                    self._maxsize,
                )
