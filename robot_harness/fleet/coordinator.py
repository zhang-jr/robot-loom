"""FleetCoordinator — lease + routing + real-time broadcast (ADR-008 / ADR-028).

The coordinator is the harness-side owner of "who drives which robot". It holds
NO Brain and runs NO control loop (ADR-028): each leased robot gets its own
AgentLoop elsewhere, and the coordinator only hands out exclusive leases, routes
subtasks to compatible bodies, and broadcasts real-time fleet state.

    assign(subtask) ─┐
                     ├─ FleetRouter.route ── pick a compatible, free robot
                     └─ LeaseManager.acquire ── claim it exclusively
                            │
                            └─ FleetBus.publish ── announce the assignment

``fleet_size=1`` is the degenerate case (ADR-008): one robot, one lease entry,
one subscriber — no single-robot fast path is special-cased.
"""

from __future__ import annotations

from robot_harness.embodiment.base import RobotType
from robot_harness.fleet.lease import LeaseManager, RobotLease
from robot_harness.fleet.routing import FleetRouter, RobotAssignment
from robot_harness.fleet.shared_bus import FleetBus, FleetEvent
from robot_harness.observability.tracer import tracer
from robot_harness.skill.base import Subtask


class FleetCoordinator:
    """Coordinate exclusive robot access, subtask routing, and fleet broadcast.

    Args:
        capabilities: robot_id → robot_type for the whole fleet. Drives routing
            and bounds the leasable set.
        lease_manager: optional shared LeaseManager (one per harness instance).
        bus:           optional shared FleetBus.
    """

    def __init__(
        self,
        capabilities: dict[str, RobotType],
        *,
        lease_manager: LeaseManager | None = None,
        bus: FleetBus | None = None,
    ) -> None:
        self._capabilities = dict(capabilities)
        self._router = FleetRouter(capabilities)
        self._leases = lease_manager or LeaseManager()
        self._bus = bus or FleetBus()

    @property
    def fleet_size(self) -> int:
        return len(self._capabilities)

    @property
    def bus(self) -> FleetBus:
        return self._bus

    async def acquire(self, robot_id: str, lease_s: float, *, holder: str) -> RobotLease:
        """Lease *robot_id* exclusively to *holder* (raises on conflict)."""
        return await self._leases.acquire(robot_id, lease_s, holder=holder)

    async def release(self, lease: RobotLease) -> None:
        """Release a held lease (stale releases are no-ops)."""
        await self._leases.release(lease)

    def lease_for(self, robot_id: str) -> RobotLease | None:
        """Return the live lease on *robot_id*, or None (one robot = one lease)."""
        return self._leases.get(robot_id)

    async def assign(self, subtask: Subtask, *, lease_s: float, holder: str) -> RobotAssignment:
        """Route *subtask* to a compatible free robot and lease it to *holder*.

        Busy = currently leased; the router never picks a leased robot, so a
        routed assignment is always immediately leasable. Returns the
        assignment; the caller stamps ``assignment.robot_id`` onto the subtask
        and spins up that robot's AgentLoop (ADR-028).

        Raises:
            HeterogeneousMismatchError / FleetCapacityExceededError: from routing.
            RobotLockConflictError: only on a race where the routed robot was
                leased between routing and acquire; the caller may retry.
        """
        busy = {rid for rid in self._capabilities if self._leases.is_leased(rid)}
        assignment = self._router.route(subtask, busy=busy)
        await self._leases.acquire(assignment.robot_id, lease_s, holder=holder)
        await self.broadcast(
            FleetEvent(
                event_type="subtask.assigned",
                robot_id=assignment.robot_id,
                payload={"subtask_id": subtask.subtask_id, "holder": holder},
            )
        )
        tracer.event(
            "fleet.subtask_assigned",
            subtask_id=subtask.subtask_id,
            robot_id=assignment.robot_id,
            holder=holder,
        )
        return assignment

    async def broadcast(self, event: FleetEvent) -> None:
        """Publish a real-time fleet event to all subscribers (ADR-005)."""
        await self._bus.publish(event)
