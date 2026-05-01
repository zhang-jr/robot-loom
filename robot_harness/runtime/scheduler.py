"""Heartbeat scheduler — periodic health-check loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from robot_harness.observability.tracer import tracer

_Tick = Callable[[], Coroutine[Any, Any, None]]


class HeartbeatScheduler:
    """Calls registered tick functions at a fixed interval.

    Usage::

        scheduler = HeartbeatScheduler(interval_s=5.0)
        scheduler.register(my_health_check)
        task = asyncio.create_task(scheduler.run())
    """

    def __init__(self, interval_s: float = 5.0) -> None:
        self._interval = interval_s
        self._ticks: list[_Tick] = []
        self._running = False

    def register(self, tick: _Tick) -> None:
        self._ticks.append(tick)

    async def run(self) -> None:
        self._running = True
        while self._running:
            await asyncio.sleep(self._interval)
            for tick in self._ticks:
                try:
                    await tick()
                except Exception as exc:  # noqa: BLE001
                    tracer.event("scheduler.tick_error", error=str(exc))

    def stop(self) -> None:
        self._running = False
