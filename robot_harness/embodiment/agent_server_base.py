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

from robot_harness.embodiment.base import (
    DispatchHandle,
    EmbodimentCommand,
    Frame,
    RobotState,
    RobotType,
    SafetyVerdict,
)


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
        return Frame(**raw)

    async def get_state(self) -> RobotState:
        raw = await self._client.get_state()
        return self._to_state(raw)

    async def dispatch(self, cmd: EmbodimentCommand) -> DispatchHandle:
        raw = await self._client.dispatch(cmd.model_dump())
        return DispatchHandle(
            robot_id=self.robot_id,
            action_id=raw.get("action_id", str(uuid.uuid4())),
            estimated_duration_s=raw.get("estimated_duration_s", 1.0),
        )

    async def safety_check(self, cmd: EmbodimentCommand) -> SafetyVerdict:
        raw = await self._client.safety_check(cmd.model_dump())
        return SafetyVerdict(
            passed=raw.get("passed", True),
            reason=raw.get("reason", ""),
            violated_rules=raw.get("violated_rules", []),
        )

    # -- SupportsVerbs capability ------------------------------------------

    async def call_verb(self, verb: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run an on-robot mid-loop verb. Returns the raw CompletionVerdict dict.

        The robot_sdk verb tool validates the shape; this is pure transport so
        the verb path is identical across real hardware and simulator backends.
        """
        return await self._client.call_verb(verb, payload)

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- helpers -----------------------------------------------------------

    def _to_state(self, raw: dict[str, Any]) -> RobotState:
        return RobotState(
            robot_id=str(raw.get("robot_id", self.robot_id)),
            joint_positions=list(raw.get("joint_positions", [0.0] * self._dof)),
            joint_velocities=list(raw.get("joint_velocities", [0.0] * self._dof)),
            end_effector_pose=dict(raw.get("end_effector_pose", {})),
            gripper_state=float(raw.get("gripper_state", 0.0)),
        )
