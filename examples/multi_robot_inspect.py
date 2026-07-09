"""Multi-robot inspection — N-loop fleet coordination (ADR-008 / ADR-028).

Demonstrates the fleet execution topology decided in ADR-028:

  • One FleetCoordinator hands out exclusive leases and routes inspection
    subtasks to type-compatible robots.
  • Each leased robot runs its OWN AgentLoop concurrently — independent Brain
    session, independent conversation. The coordinator holds no Brain.
  • Robots announce status on the FleetBus (real-time state, ADR-005); peers
    read it without sharing a Brain context.

fleet_size=1 would be the same code with a one-robot fleet — no special path.

Run:  python examples/multi_robot_inspect.py
"""

from __future__ import annotations

import asyncio

from robot_harness.brain.base import BrainDecision, Task
from robot_harness.config.schema import EmbodimentBackendConfig, HarnessConfig
from robot_harness.fleet.coordinator import FleetCoordinator
from robot_harness.fleet.shared_bus import FleetEvent
from robot_harness.runtime.agent_loop import AgentLoop
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.skill.base import Subtask

# ---------------------------------------------------------------------------
# A fleet of 3 heterogeneous robots: 2 arms (manipulation) + 1 mobile (patrol).
# ---------------------------------------------------------------------------

FLEET: dict[str, str] = {
    "arm-0": "arm",
    "arm-1": "arm",
    "mobile-0": "mobile",
}

# Inspection subtasks, each declaring which robot types can carry it.
SUBTASKS = [
    Subtask(
        subtask_id="inspect-shelf-A",
        description="Inspect shelf A for missing items",
        robot_id="",  # unbound — the router assigns it
        parameters={"required_robot_types": ["arm"]},
    ),
    Subtask(
        subtask_id="inspect-shelf-B",
        description="Inspect shelf B for missing items",
        robot_id="",
        parameters={"required_robot_types": ["arm"]},
    ),
    Subtask(
        subtask_id="patrol-aisle-3",
        description="Patrol aisle 3 and report obstacles",
        robot_id="",
        parameters={"required_robot_types": ["mobile"]},
    ),
]


class InspectMockBrain:
    """Trivial Brain that finishes an inspection subtask in one turn."""

    def __init__(self, subtask: Subtask) -> None:
        self._subtask = subtask

    async def decide(self, messages: list, tools: list, **_: object) -> BrainDecision:  # type: ignore[type-arg]
        return BrainDecision(
            decision_type="plan",
            plan=f"Inspection '{self._subtask.subtask_id}' complete.",
            message=f"{self._subtask.description}: OK",
        )

    @property
    def supports_streaming(self) -> bool:
        return False


async def _run_one(
    coord: FleetCoordinator,
    ctx: HarnessContext,
    subtask: Subtask,
) -> tuple[str, str]:
    """Route → lease → run a dedicated AgentLoop → release. One robot, one loop."""
    assignment = await coord.assign(subtask, lease_s=30.0, holder=subtask.subtask_id)
    robot_id = assignment.robot_id
    lease = coord.lease_for(robot_id)
    assert lease is not None  # just acquired in assign()
    try:
        await coord.broadcast(
            FleetEvent(event_type="status", robot_id=robot_id, payload={"state": "inspecting"})
        )
        brain = InspectMockBrain(subtask)
        loop = AgentLoop(brain=brain, ctx=ctx, max_turns=5)
        task = Task(
            task_id=subtask.subtask_id,
            description=subtask.description,
            robot_id=robot_id,
        )
        result = await loop.run(task)
        await coord.broadcast(
            FleetEvent(event_type="status", robot_id=robot_id, payload={"state": "idle"})
        )
        return robot_id, result.outcome
    finally:
        await coord.release(lease)


async def main() -> None:
    cfg = HarnessConfig(
        fleet_size=len(FLEET),
        robot_ids=list(FLEET),
        embodiments={
            rid: EmbodimentBackendConfig(backend="mock", robot_type=rtype)  # type: ignore[arg-type]
            for rid, rtype in FLEET.items()
        },
    )
    ctx = HarnessContext.build(cfg)
    capabilities = {rid: a.robot_type for rid, a in ctx.embodiment_adapters.items()}
    coord = FleetCoordinator(capabilities)

    # Log every fleet event as it happens (real-time state, ADR-005).
    async def _log(event: FleetEvent) -> None:
        print(f"  [bus] {event.robot_id}: {event.event_type} {event.payload}")

    coord.bus.subscribe(_log)

    print(f"Fleet of {coord.fleet_size}: {capabilities}")
    print("Running inspection subtasks on independent per-robot AgentLoops...\n")

    # All subtasks dispatched concurrently — N independent loops (ADR-028).
    results = await asyncio.gather(*(_run_one(coord, ctx, st) for st in SUBTASKS))

    print("\n=== Results ===")
    for robot_id, outcome in results:
        print(f"  {robot_id}: {outcome}")
    print(f"\nFinal fleet snapshot: { {r: e.payload for r, e in coord.bus.snapshot().items()} }")


if __name__ == "__main__":
    asyncio.run(main())
