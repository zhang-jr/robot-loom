"""Fleet coordination tests — lease / routing / bus / coordinator (ADR-008 / ADR-028).

Boundaries exercised at fleet_size = 1 / 3 / 5: single-robot is the degenerate
case of the same code path, never a separate fast path.
"""

from __future__ import annotations

import asyncio

import pytest

from robot_harness.errors import (
    FleetCapacityExceededError,
    HeterogeneousMismatchError,
    RobotLockConflictError,
)
from robot_harness.fleet.coordinator import FleetCoordinator
from robot_harness.fleet.lease import LeaseManager
from robot_harness.fleet.routing import FleetRouter
from robot_harness.fleet.shared_bus import FleetBus, FleetEvent
from robot_harness.skill.base import Subtask


def _subtask(sid: str, types: list[str] | None = None) -> Subtask:
    params = {"required_robot_types": types} if types is not None else {}
    return Subtask(subtask_id=sid, description=sid, robot_id="", parameters=params)


# ---------------------------------------------------------------------------
# LeaseManager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lease_acquire_and_release_single_robot() -> None:
    mgr = LeaseManager()
    lease = await mgr.acquire("robot-0", 10.0, holder="task-A")
    assert mgr.is_leased("robot-0")
    assert mgr.held_by("robot-0") == "task-A"
    await mgr.release(lease)
    assert not mgr.is_leased("robot-0")


@pytest.mark.asyncio
async def test_lease_conflict_on_different_holder() -> None:
    mgr = LeaseManager()
    await mgr.acquire("robot-0", 10.0, holder="task-A")
    with pytest.raises(RobotLockConflictError):
        await mgr.acquire("robot-0", 10.0, holder="task-B")


@pytest.mark.asyncio
async def test_lease_renew_same_holder_is_idempotent() -> None:
    mgr = LeaseManager()
    first = await mgr.acquire("robot-0", 10.0, holder="task-A")
    second = await mgr.acquire("robot-0", 10.0, holder="task-A")
    assert mgr.is_leased("robot-0")
    # Renewal replaces the lease; the old handle is now stale but release is lenient.
    assert second.lease_id != first.lease_id
    await mgr.release(first)  # stale → no-op, must not drop the live lease
    assert mgr.is_leased("robot-0")
    await mgr.release(second)
    assert not mgr.is_leased("robot-0")


@pytest.mark.asyncio
async def test_lease_expiry_allows_takeover() -> None:
    mgr = LeaseManager()
    await mgr.acquire("robot-0", 0.01, holder="task-A")
    await asyncio.sleep(0.02)
    assert not mgr.is_leased("robot-0")  # expired
    # A different holder can now take it without conflict.
    await mgr.acquire("robot-0", 10.0, holder="task-B")
    assert mgr.held_by("robot-0") == "task-B"


@pytest.mark.asyncio
async def test_lease_independent_across_fleet_of_5() -> None:
    mgr = LeaseManager()
    ids = [f"robot-{i}" for i in range(5)]
    await asyncio.gather(*(mgr.acquire(rid, 10.0, holder=f"t-{rid}") for rid in ids))
    assert all(mgr.is_leased(rid) for rid in ids)
    # Releasing one does not touch the others.
    await mgr.release(mgr.get("robot-2"))  # type: ignore[arg-type]
    assert not mgr.is_leased("robot-2")
    assert all(mgr.is_leased(rid) for rid in ids if rid != "robot-2")


# ---------------------------------------------------------------------------
# FleetRouter
# ---------------------------------------------------------------------------


def test_route_by_type_compatibility() -> None:
    router = FleetRouter({"arm-0": "arm", "arm-1": "arm", "mobile-0": "mobile"})
    assignment = router.route(_subtask("patrol", ["mobile"]))
    assert assignment.robot_id == "mobile-0"
    assert assignment.robot_type == "mobile"


def test_route_any_type_when_unconstrained() -> None:
    router = FleetRouter({"arm-1": "arm", "arm-0": "arm"})
    # No required types → any robot, picked deterministically (lowest id).
    assert router.route(_subtask("x")).robot_id == "arm-0"


def test_route_skips_busy_robots() -> None:
    router = FleetRouter({"arm-0": "arm", "arm-1": "arm"})
    assignment = router.route(_subtask("x", ["arm"]), busy={"arm-0"})
    assert assignment.robot_id == "arm-1"


def test_route_capacity_exceeded_when_all_busy() -> None:
    router = FleetRouter({"arm-0": "arm"})
    with pytest.raises(FleetCapacityExceededError):
        router.route(_subtask("x", ["arm"]), busy={"arm-0"})


def test_route_heterogeneous_mismatch() -> None:
    router = FleetRouter({"arm-0": "arm"})
    with pytest.raises(HeterogeneousMismatchError):
        router.route(_subtask("x", ["humanoid"]))


# ---------------------------------------------------------------------------
# FleetBus
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bus_publish_updates_snapshot_and_notifies() -> None:
    bus = FleetBus()
    seen: list[FleetEvent] = []

    async def handler(event: FleetEvent) -> None:
        seen.append(event)

    bus.subscribe(handler)
    await bus.publish(FleetEvent(event_type="status", robot_id="robot-0", payload={"s": "busy"}))
    await bus.publish(FleetEvent(event_type="status", robot_id="robot-0", payload={"s": "idle"}))

    assert len(seen) == 2
    # Real-time state: latest overwrites prior (ADR-005, no history here).
    assert bus.latest("robot-0").payload == {"s": "idle"}  # type: ignore[union-attr]
    assert set(bus.snapshot()) == {"robot-0"}


@pytest.mark.asyncio
async def test_bus_handler_error_does_not_stop_delivery() -> None:
    bus = FleetBus()
    delivered: list[str] = []

    async def bad(event: FleetEvent) -> None:
        raise RuntimeError("boom")

    async def good(event: FleetEvent) -> None:
        delivered.append(event.robot_id)

    bus.subscribe(bad)
    bus.subscribe(good)
    await bus.publish(FleetEvent(event_type="status", robot_id="robot-0"))
    assert delivered == ["robot-0"]  # good still ran despite bad raising


# ---------------------------------------------------------------------------
# FleetCoordinator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinator_fleet_size_one() -> None:
    coord = FleetCoordinator({"robot-0": "arm"})
    assert coord.fleet_size == 1
    assignment = await coord.assign(_subtask("t1", ["arm"]), lease_s=10.0, holder="t1")
    assert assignment.robot_id == "robot-0"
    assert coord.lease_for("robot-0") is not None


@pytest.mark.asyncio
async def test_coordinator_assigns_distinct_robots_in_fleet_of_3() -> None:
    coord = FleetCoordinator({"arm-0": "arm", "arm-1": "arm", "mobile-0": "mobile"})
    a1 = await coord.assign(_subtask("s1", ["arm"]), lease_s=10.0, holder="s1")
    a2 = await coord.assign(_subtask("s2", ["arm"]), lease_s=10.0, holder="s2")
    # Two arm subtasks must land on the two distinct arms, not the same one.
    assert {a1.robot_id, a2.robot_id} == {"arm-0", "arm-1"}
    # A third arm subtask has no free arm → capacity exceeded.
    with pytest.raises(FleetCapacityExceededError):
        await coord.assign(_subtask("s3", ["arm"]), lease_s=10.0, holder="s3")


@pytest.mark.asyncio
async def test_coordinator_release_frees_robot_for_reassignment() -> None:
    coord = FleetCoordinator({"arm-0": "arm"})
    await coord.assign(_subtask("s1", ["arm"]), lease_s=10.0, holder="s1")
    lease = coord.lease_for("arm-0")
    assert lease is not None
    await coord.release(lease)
    # Now reassignable.
    a2 = await coord.assign(_subtask("s2", ["arm"]), lease_s=10.0, holder="s2")
    assert a2.robot_id == "arm-0"


@pytest.mark.asyncio
async def test_coordinator_assign_broadcasts_on_bus() -> None:
    bus = FleetBus()
    events: list[FleetEvent] = []

    async def handler(event: FleetEvent) -> None:
        events.append(event)

    bus.subscribe(handler)
    coord = FleetCoordinator({"arm-0": "arm"}, bus=bus)
    await coord.assign(_subtask("s1", ["arm"]), lease_s=10.0, holder="s1")
    assert any(e.event_type == "subtask.assigned" and e.robot_id == "arm-0" for e in events)
