"""Taught-motion catalog — named on-robot motions discovered from ``/health``.

A *taught motion* is a joint-waypoint sequence that was demonstrated on a
specific robot and stored **on that robot's agent_server** under a name
(``place_on_tray``, ``stow_left_bin``, …). The harness never authors one and
never keeps the numbers in workspace prose: it discovers the catalog live, the
same way it discovers ``available_verbs``.

Two consumers, deliberately asymmetric — this split is the whole point:

* **The Brain sees only names + descriptions.** ``run_taught_motion`` takes one
  ``motion`` string, so a placement runs as a single tool call and the model
  never transcribes joint angles. Keeping metric assets out of the planning
  vocabulary is the same altitude rule that keeps grasp-pose estimation out of
  it.
* **The SafetyEnvelope sees every joint value.** The catalog carries the points
  precisely so ``RunTaughtMotionTool.to_safety_commands`` can validate the whole
  trajectory against ``safety.joint_limits_rad`` BEFORE the first point is
  dispatched. Sending a bare name with no points would leave the pre-dispatch
  gate checking nothing while still recording an audit entry — a false "passed"
  is worse than an honest skip.

Those are not in tension: the altitude rule governs the *planning vocabulary*,
the envelope is a machine-side gate whose inputs never enter a prompt.

The catalog is fleet-wide but keyed per robot (bodies are taught separately);
the Brain-facing name enum is the fleet UNION, mirroring the verb gate — the
tool spec is shared across robots with ``robot_id`` as an argument, so pruning
to one robot's names would hide vocabulary another robot legitimately supports.
The addressed robot's agent_server rejects a name it does not know.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TaughtPoint(BaseModel):
    """One waypoint of a taught motion: a full joint target plus optional gripper."""

    joints: list[float] = Field(min_length=1)
    gripper: float | None = None


class TaughtMotion(BaseModel):
    """A named joint-waypoint sequence taught on one robot.

    ``description`` is what the Brain reads to choose the motion, so it must say
    what the motion accomplishes ("release the held object onto the tray") rather
    than restate the joint count.
    """

    name: str
    description: str = ""
    points: list[TaughtPoint] = Field(default_factory=list)


class TaughtMotionCatalog:
    """Fleet-wide ``robot_id → {motion name → TaughtMotion}``, refreshed from ``/health``.

    Mutable and shared by reference: ``HarnessContext`` refreshes it at task
    start, ``RunTaughtMotionTool`` reads it for both its name enum and its
    safety commands. Passing the same object to both is what lets a re-taught
    motion take effect without rebuilding the tool registry.

    Empty is the honest default — an agent_server that advertises no taught
    motions leaves ``run_taught_motion`` out of the Brain's vocabulary entirely
    (the verb gate prunes it), rather than exposing a tool with no valid input.
    """

    def __init__(self) -> None:
        self._by_robot: dict[str, dict[str, TaughtMotion]] = {}

    def replace(self, robot_id: str, motions: list[TaughtMotion]) -> None:
        """Swap in one robot's freshly discovered motions (last write wins)."""
        self._by_robot[robot_id] = {m.name: m for m in motions}

    def forget(self, robot_id: str) -> None:
        """Drop a robot's motions — its ``/health`` no longer advertises any."""
        self._by_robot.pop(robot_id, None)

    def get(self, robot_id: str, name: str) -> TaughtMotion | None:
        """The named motion as taught on *robot_id*, or None if it has no such motion."""
        return self._by_robot.get(robot_id, {}).get(name)

    def names(self) -> list[str]:
        """Sorted fleet union of motion names — the Brain-facing enum."""
        return sorted({name for motions in self._by_robot.values() for name in motions})

    def describe(self) -> list[str]:
        """``"name — description"`` lines for the tool schema, fleet-wide.

        A name taught on several robots is listed once; the first non-empty
        description wins (bodies performing the same motion describe it the same
        way, and a disagreement is a teaching bug worth surfacing as-is rather
        than silently concatenating).
        """
        seen: dict[str, str] = {}
        for motions in self._by_robot.values():
            for name, motion in motions.items():
                if not seen.get(name):
                    seen[name] = motion.description
        return [f"{name} — {desc}" if desc else name for name, desc in sorted(seen.items())]

    def is_empty(self) -> bool:
        return not any(self._by_robot.values())


def parse_taught_motions(raw: Any) -> list[TaughtMotion]:
    """Parse the ``/health.taught_motions`` payload, dropping malformed entries.

    Discovery must never take the fleet down: a backend that advertises a
    half-broken catalog loses the bad entries (and, if all of them are bad, the
    verb) instead of raising into the planning path. Entries missing ``points``
    are dropped too — a motion the envelope cannot check is one the harness will
    not offer.
    """
    if not isinstance(raw, list):
        return []
    motions: list[TaughtMotion] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            motion = TaughtMotion.model_validate(entry)
        except ValueError:
            continue
        if motion.name and motion.points:
            motions.append(motion)
    return motions


__all__ = [
    "TaughtMotion",
    "TaughtMotionCatalog",
    "TaughtPoint",
    "parse_taught_motions",
]
