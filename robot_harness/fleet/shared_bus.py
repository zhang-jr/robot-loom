"""FleetBus — real-time shared state across robots (ADR-005 / ADR-028).

ADR-005 boundary, made concrete here:

    • FleetBus  = the *real-time* state of the fleet: what each robot is doing
                  right now, its last-known status. Volatile, in-process, never
                  the source of truth for history. This is how one robot's
                  AgentLoop learns "what is robot-2 doing" without a shared Brain
                  context (ADR-028).
    • EpisodicMemory = the *historical* record: what happened, for grounding and
                  replay. Durable, queried on demand (ADR-024).

A status written here is overwritten by the next status from the same robot; if
you need to remember that it happened, write an episode. The bus does not
persist and does not deduplicate history.

Asyncio-safe, in-process. fleet_size=1 keeps exactly one publisher/subscriber
and one status entry — no special-casing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from robot_harness.observability.tracer import tracer


def _now() -> datetime:
    return datetime.now(tz=UTC)


class FleetEvent(BaseModel):
    """A real-time fleet signal (status change, lease change, hand-off, …)."""

    event_type: str
    robot_id: str
    timestamp: datetime = Field(default_factory=_now)
    payload: dict[str, Any] = Field(default_factory=dict)


_Subscriber = Callable[[FleetEvent], Coroutine[Any, Any, None]]


class FleetBus:
    """Pub-sub for FleetEvents plus a last-known-status snapshot per robot.

    The snapshot is the bus's real-time value: any robot's loop can read the
    current status of its peers synchronously, while event subscribers react to
    changes asynchronously.
    """

    def __init__(self) -> None:
        self._subscribers: list[_Subscriber] = []
        # robot_id -> the most recent FleetEvent from that robot.
        self._latest: dict[str, FleetEvent] = {}
        self._lock = asyncio.Lock()

    def subscribe(self, handler: _Subscriber) -> None:
        """Register a coroutine handler invoked on every published event."""
        self._subscribers.append(handler)

    async def publish(self, event: FleetEvent) -> None:
        """Record *event* as the robot's latest status and fan out to subscribers.

        A subscriber raising does not stop delivery to the others — the failure
        is traced, not swallowed silently into the publisher (ADR anti-pattern:
        no try/except that hides errors; here it is surfaced via the tracer and
        delivery continues, matching the internal EventBus contract).
        """
        async with self._lock:
            self._latest[event.robot_id] = event
        for handler in self._subscribers:
            try:
                await handler(event)
            except Exception as exc:  # noqa: BLE001
                tracer.event(
                    "fleet.bus_handler_error",
                    event_type=event.event_type,
                    robot_id=event.robot_id,
                    handler=getattr(handler, "__qualname__", repr(handler)),
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

    def latest(self, robot_id: str) -> FleetEvent | None:
        """Return the last-known event from *robot_id*, or None if never seen."""
        return self._latest.get(robot_id)

    def snapshot(self) -> dict[str, FleetEvent]:
        """Return a shallow copy of the current per-robot status map."""
        return dict(self._latest)
