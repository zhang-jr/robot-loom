"""FleetRouter — map a Subtask to a concrete robot (ADR-008 / ADR-028).

Routing is capability-first: a subtask declares which robot *types* can carry it
(via ``embodiment_compat`` semantics), the router picks an available robot of a
compatible type. ``robot_id`` on the resulting assignment is what stamps the
otherwise-unbound subtask onto a body — the router does not mutate the subtask.

The router is pure policy over a capability map + a busy set; it owns no leases
and no Brain. The FleetCoordinator feeds it liveness (which robots are leased)
and acts on the assignment.
"""

from __future__ import annotations

from pydantic import BaseModel

from robot_harness.embodiment.base import RobotType
from robot_harness.errors import FleetCapacityExceededError, HeterogeneousMismatchError
from robot_harness.observability.tracer import tracer
from robot_harness.skill.base import Subtask

# Subtask carries its compatible robot types under this parameter key; absent or
# empty means "any type is acceptable" (a homogeneous fleet never needs to set it).
REQUIRED_TYPES_KEY = "required_robot_types"


class RobotAssignment(BaseModel):
    """Result of routing one subtask to one robot."""

    subtask_id: str
    robot_id: str
    robot_type: RobotType
    reason: str = ""


def required_types(subtask: Subtask) -> list[str]:
    """Read the compatible robot types a subtask declares (possibly empty)."""
    raw = subtask.parameters.get(REQUIRED_TYPES_KEY, [])
    return [str(t) for t in raw] if isinstance(raw, list) else []


class FleetRouter:
    """Assign subtasks to robots by type compatibility, then availability.

    Args:
        capabilities: robot_id → robot_type for every robot in the fleet. Built
            once from the embodiment catalog at wiring time so the router stays
            decoupled from concrete adapters.
    """

    def __init__(self, capabilities: dict[str, RobotType]) -> None:
        self._capabilities = dict(capabilities)

    def route(self, subtask: Subtask, *, busy: set[str] | None = None) -> RobotAssignment:
        """Pick a robot for *subtask*.

        Selection order: filter the fleet to types compatible with the subtask,
        then to robots not in *busy*, then pick deterministically (lowest
        robot_id) so assignment is reproducible.

        Raises:
            HeterogeneousMismatchError: no robot of any compatible type exists in
                the fleet at all (a configuration / capability problem).
            FleetCapacityExceededError: compatible robots exist but all are busy
                (a transient contention problem — caller may retry).
        """
        busy = busy or set()
        wanted = required_types(subtask)

        compatible = [
            rid for rid, rtype in self._capabilities.items() if not wanted or rtype in wanted
        ]
        if not compatible:
            raise HeterogeneousMismatchError(
                f"No robot of type {wanted} in fleet {sorted(self._capabilities)}",
                subtask_id=subtask.subtask_id,
            )

        free = sorted(rid for rid in compatible if rid not in busy)
        if not free:
            raise FleetCapacityExceededError(
                f"All {len(compatible)} compatible robot(s) busy for subtask "
                f"'{subtask.subtask_id}'",
                subtask_id=subtask.subtask_id,
            )

        chosen = free[0]
        assignment = RobotAssignment(
            subtask_id=subtask.subtask_id,
            robot_id=chosen,
            robot_type=self._capabilities[chosen],
            reason=f"compatible={wanted or 'any'} free={free}",
        )
        tracer.event(
            "fleet.subtask_routed",
            subtask_id=subtask.subtask_id,
            robot_id=chosen,
            robot_type=self._capabilities[chosen],
        )
        return assignment
