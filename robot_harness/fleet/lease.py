"""RobotLease and LeaseManager — exclusive per-robot leasing (ADR-008 / ADR-028).

A lease is the primitive that prevents two tasks from driving the same robot at
once. Each leased robot maps to exactly one AgentLoop (ADR-028 N-loop), so the
lease holder is effectively "the loop currently owning this robot".

The manager is in-process and asyncio-safe. For ``fleet_size=1`` it is a single
dict entry behind one lock — the fast path ADR-008 promised costs no more than a
dict lookup, with no special-casing for the single-robot fleet.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from robot_harness.errors import RobotLockConflictError
from robot_harness.observability.tracer import tracer


def _now() -> datetime:
    return datetime.now(tz=UTC)


class RobotLease(BaseModel):
    """An exclusive, time-bounded claim on one robot.

    ``holder`` is an opaque owner id (typically a task_id or AgentLoop id). The
    lease is serializable for tracing; its expiry deadline is tracked by the
    :class:`LeaseManager` on a monotonic clock, not on this wall-clock model, so
    NTP steps cannot extend or shorten a live lease.
    """

    lease_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    robot_id: str
    holder: str
    lease_s: float
    acquired_at: datetime = Field(default_factory=_now)


class LeaseManager:
    """In-process, asyncio-safe registry of one active lease per robot.

    Acquiring a robot already held by a *different* holder raises
    ``RobotLockConflictError`` unless the existing lease has expired (then it is
    silently taken over). Re-acquiring as the *same* holder renews the lease and
    is idempotent — a held loop re-asserting its claim never conflicts with
    itself.
    """

    def __init__(self) -> None:
        self._leases: dict[str, RobotLease] = {}
        # robot_id -> monotonic deadline; kept separate from the serializable
        # RobotLease so expiry is immune to wall-clock changes.
        self._deadlines: dict[str, float] = {}
        self._lock = asyncio.Lock()

    def _is_expired(self, robot_id: str) -> bool:
        deadline = self._deadlines.get(robot_id)
        return deadline is not None and time.monotonic() >= deadline

    async def acquire(self, robot_id: str, lease_s: float, *, holder: str) -> RobotLease:
        """Acquire (or renew) an exclusive lease on *robot_id*.

        Raises:
            RobotLockConflictError: the robot is currently leased to a different,
                non-expired holder.
        """
        async with self._lock:
            existing = self._leases.get(robot_id)
            if existing is not None and not self._is_expired(robot_id):
                if existing.holder != holder:
                    raise RobotLockConflictError(
                        f"Robot '{robot_id}' is leased to '{existing.holder}', "
                        f"requested by '{holder}'",
                        robot_id=robot_id,
                    )
                # Same holder → renew in place.
            lease = RobotLease(robot_id=robot_id, holder=holder, lease_s=lease_s)
            self._leases[robot_id] = lease
            self._deadlines[robot_id] = time.monotonic() + lease_s
            tracer.event(
                "fleet.lease_acquired",
                robot_id=robot_id,
                holder=holder,
                lease_id=lease.lease_id,
                lease_s=lease_s,
            )
            return lease

    async def release(self, lease: RobotLease) -> None:
        """Release *lease*. A stale lease (already replaced/expired) is a no-op.

        Releasing is intentionally lenient: a loop that times out and a takeover
        that already reclaimed the robot must not deadlock each other.
        """
        async with self._lock:
            current = self._leases.get(lease.robot_id)
            if current is None or current.lease_id != lease.lease_id:
                tracer.event(
                    "fleet.lease_release_stale",
                    robot_id=lease.robot_id,
                    lease_id=lease.lease_id,
                )
                return
            del self._leases[lease.robot_id]
            self._deadlines.pop(lease.robot_id, None)
            tracer.event(
                "fleet.lease_released",
                robot_id=lease.robot_id,
                holder=lease.holder,
                lease_id=lease.lease_id,
            )

    def is_leased(self, robot_id: str) -> bool:
        """True if *robot_id* has a live (non-expired) lease."""
        return robot_id in self._leases and not self._is_expired(robot_id)

    def get(self, robot_id: str) -> RobotLease | None:
        """Return the live lease on *robot_id*, or None if unleased/expired."""
        if self.is_leased(robot_id):
            return self._leases[robot_id]
        return None

    def held_by(self, robot_id: str) -> str | None:
        """Return the holder of a live lease on *robot_id*, or None."""
        lease = self.get(robot_id)
        return lease.holder if lease is not None else None
