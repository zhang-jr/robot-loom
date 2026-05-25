"""Async event bus for intra-harness communication."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from robot_harness.bus.events import HarnessEvent
from robot_harness.observability.tracer import tracer

_Handler = Callable[[HarnessEvent], Coroutine[Any, Any, None]]


class EventBus:
    """Simple pub-sub bus.  Handlers are called sequentially per event."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[HarnessEvent] = asyncio.Queue()
        self._handlers: list[_Handler] = []
        self._running = False

    def subscribe(self, handler: _Handler) -> None:
        self._handlers.append(handler)

    async def publish(self, event: HarnessEvent) -> None:
        await self._queue.put(event)

    async def run(self) -> None:
        """Drain the queue and dispatch to all handlers.  Run as a background task."""
        self._running = True
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                continue
            for handler in self._handlers:
                try:
                    await handler(event)
                except Exception as exc:  # noqa: BLE001
                    tracer.event(
                        "bus.handler_error",
                        event_type=event.event_type,
                        handler=handler.__qualname__,
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
            self._queue.task_done()

    def stop(self) -> None:
        self._running = False


bus: EventBus = EventBus()
