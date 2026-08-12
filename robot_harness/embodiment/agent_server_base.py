"""Shared base for agent_server-backed EmbodimentAdapters (real hardware + sim).

A per-robot ``agent_server`` (real hardware) and a simulator ``agent_server`` speak
the *same* HTTP/JSON contract and both absorb all morphology- and hardware-specific
control behind it (trajectory planning, joint servo, IK, e-stop reflex). So the
harness-side adapter is a single **morphology-agnostic** wire client parameterised
by ``robot_type`` — not a per-morphology subclass. The body-specific action-space
mapping lives in the agent_server, never here.

``AgentServerAdapter`` is the common parent of both backend families. It lives at
the embodiment root (next to ``base.py``) — NOT inside ``real/`` or ``sim/`` —
because both families extend it; putting it under one family would make the other
depend on it (e.g. "sim depends on real"), which is semantically wrong.

    embodiment/
      base.py                 # EmbodimentAdapter Protocol + models + SupportsVerbs
      agent_server_base.py    # AgentServerAdapter  ← this file (shared parent)
      real/                   # real-hardware backends  → RealAgentServerAdapter
      sim/                    # simulator backends      → SimEmbodimentAdapter (+ lifecycle)

Concrete backends differ only by ① the transport client they construct and ②, for
sim, an extra lifecycle surface (reset / load_scene / sim_time).
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol

from pydantic import ValidationError

from robot_harness.embodiment.base import (
    DispatchHandle,
    EmbodimentCommand,
    Frame,
    RobotState,
    RobotType,
    SafetyVerdict,
)
from robot_harness.errors import RobotOfflineError


class AgentServerClient(Protocol):
    """The wire surface ``AgentServerAdapter`` delegates to.

    Both the real-robot HTTP client and the simulator HTTP client satisfy this;
    the simulator client additionally exposes reset / load_scene / sim_time.
    """

    async def get_state(self) -> dict[str, Any]: ...
    async def get_camera_frame(self, camera: str) -> dict[str, Any]: ...
    async def dispatch(self, cmd: dict[str, Any]) -> dict[str, Any]: ...
    async def safety_check(self, cmd: dict[str, Any]) -> dict[str, Any]: ...
    async def call_verb(self, verb: str, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def abort(self, payload: dict[str, Any] | None = None) -> dict[str, Any]: ...
    async def health(self) -> dict[str, Any]: ...
    async def aclose(self) -> None: ...


class AgentServerAdapter:
    """Base EmbodimentAdapter backed by an external agent_server wire client.

    Satisfies the ``EmbodimentAdapter`` Protocol AND ``SupportsVerbs`` via
    structural typing. Subclasses build the concrete transport client and pass it
    up; this base owns the EmbodimentCommand ⇄ wire translation only.
    """

    def __init__(
        self,
        robot_id: str,
        *,
        robot_type: RobotType,
        dof: int,
        client: AgentServerClient,
    ) -> None:
        self.robot_id = robot_id
        self.robot_type: RobotType = robot_type
        self._dof = dof
        self._client = client

    # -- EmbodimentAdapter Protocol ----------------------------------------

    async def get_camera_frame(self, camera: str) -> Frame:
        raw = await self._client.get_camera_frame(camera)
        try:
            return Frame(**raw)
        except (ValidationError, TypeError) as exc:
            raise self._malformed(f"camera frame ('{camera}')", exc) from exc

    async def get_state(self) -> RobotState:
        raw = await self._client.get_state()
        try:
            return self._to_state(raw)
        except (ValidationError, TypeError, ValueError) as exc:
            raise self._malformed("robot state", exc) from exc

    async def dispatch(self, cmd: EmbodimentCommand) -> DispatchHandle:
        raw = await self._client.dispatch(cmd.model_dump())
        return DispatchHandle(
            robot_id=self.robot_id,
            action_id=raw.get("action_id", str(uuid.uuid4())),
            estimated_duration_s=raw.get("estimated_duration_s", 1.0),
        )

    async def safety_check(self, cmd: EmbodimentCommand) -> SafetyVerdict:
        raw = await self._client.safety_check(cmd.model_dump())
        passed = raw.get("passed")
        if not isinstance(passed, bool):
            # Fail closed: this check is part of the mandatory pre-dispatch gate
            # (SafetyEnvelope pass 2), so a response without a boolean verdict is
            # a refusal, never a default pass.
            return SafetyVerdict(
                passed=False,
                reason=f"malformed safety_check response: {raw!r}",
                violated_rules=["malformed safety_check response"],
            )
        return SafetyVerdict(
            passed=passed,
            reason=str(raw.get("reason", "")),
            violated_rules=[str(r) for r in (raw.get("violated_rules") or [])],
        )

    # -- SupportsVerbs capability ------------------------------------------

    async def call_verb(self, verb: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run an on-robot mid-loop verb. Returns the raw CompletionVerdict dict.

        The robot_sdk verb tool validates the shape; this is pure transport so
        the verb path is identical across real hardware and simulator backends.
        """
        return await self._client.call_verb(verb, payload)

    async def available_verbs(self) -> list[str] | None:
        """Verbs this robot's agent_server *currently* advertises (``/health``).

        Reflects live backend reachability, not a fixed capability list (ADR-019):
        a quadruped whose arm bridge is down drops its manipulation verbs from
        ``/health.available_verbs`` while it's down, so the harness can prune those
        verbs from the Brain's planning vocabulary instead of letting the Brain
        plan a verb that fails at call time.

        Returns bare verb names (``reactive_grasp`` / ``locomote_to`` / … —
        WITHOUT the ``robot_sdk.`` tool prefix). Three distinct outcomes:

        * ``None``  — the backend does not advertise a verb set (key absent or
          malformed payload; mock / older agent_server): capability UNKNOWN,
          callers must not prune anything.
        * ``[]``    — the backend explicitly advertises zero verbs right now
          (every capability bridge down): prune everything.
        * ``[...]`` — the current allowlist.

        Raises :class:`RobotOfflineError` when ``/health`` is unreachable or not
        speaking the contract (the wire client translates transport/decode errors).
        """
        health = await self._client.health()
        verbs = health.get("available_verbs")
        if not isinstance(verbs, list):
            # Absent, null, or a non-list (e.g. a bare string) — the backend is
            # not advertising a usable verb set; report "unknown", never a
            # char-split or a crash.
            return None
        return [str(v) for v in verbs]

    async def taught_motions(self) -> list[dict[str, Any]]:
        """Motions this body has been taught, from ``/health.taught_motions``.

        Same liveness semantics as :meth:`available_verbs` — read fresh, because
        a re-taught tray or a reloaded catalog changes the answer without a
        restart. Returns raw dicts (``{name, description, points}``); the
        robot_sdk layer parses and validates them, so a backend that advertises
        a malformed entry loses that entry, not the fleet.

        An absent or non-list key means "this backend has no taught motions" —
        the empty list. Unlike ``available_verbs`` there is no third "unknown"
        state to preserve: nothing is pruned on the strength of this answer, so
        absent and empty lead to the same place (no motion is offered).
        """
        health = await self._client.health()
        motions = health.get("taught_motions")
        if not isinstance(motions, list):
            return []
        return [m for m in motions if isinstance(m, dict)]

    async def abort(self, trace_id: str = "") -> dict[str, Any]:
        """Stop the in-flight action / verb on the robot (agent_server ``/abort``).

        Robot-level, not verb-level: cancelling an in-flight verb and cancelling a
        low-level dispatch go through the same ``/abort``. The verb tool's
        ``cancel()`` routes here so a mid-loop verb actually unwinds on the robot
        instead of only flipping a local flag.
        """
        return await self._client.abort({"trace_id": trace_id})

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- helpers -----------------------------------------------------------

    def _malformed(self, what: str, exc: Exception) -> RobotOfflineError:
        """A payload that cannot be parsed is "not speaking the contract" — the
        same typed failure as unreachable (wire-client precedent), never a raw
        ValidationError escaping into the planning loop (ISS-034)."""
        return RobotOfflineError(
            f"agent_server returned malformed {what}: {exc}",
            robot_id=self.robot_id,
            module_name="embodiment.agent_server_base",
        )

    def _to_state(self, raw: dict[str, Any]) -> RobotState:
        return RobotState(
            robot_id=str(raw.get("robot_id", self.robot_id)),
            joint_positions=list(raw.get("joint_positions", [0.0] * self._dof)),
            joint_velocities=list(raw.get("joint_velocities", [0.0] * self._dof)),
            end_effector_pose=dict(raw.get("end_effector_pose", {})),
            gripper_state=float(raw.get("gripper_state", 0.0)),
        )
