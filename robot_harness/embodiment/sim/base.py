"""SimEmbodimentAdapter — shared base for simulator embodiment backends.

A simulator backend implements the embodiment-agnostic ``EmbodimentAdapter``
Protocol (ADR-009) by delegating to a ``SimAgentServerClient`` that talks to an
external sim agent_server. On top of the standard contract it exposes a small
sim-lifecycle surface (``reset`` / ``load_scene`` / ``sim_time``) that real
hardware does not have; this is deliberately kept off the ``EmbodimentAdapter``
Protocol so the harness business layer stays sim-unaware.

Per-engine subclasses (``MujocoSimRobot`` / ``IsaacLabSimRobot``) only set the
engine label and default port — the wire contract is identical, so the rest is
shared here to avoid per-adapter duplication.
"""

from __future__ import annotations

import uuid
from typing import Any

from robot_harness.embodiment.base import (
    DispatchHandle,
    EmbodimentCommand,
    Frame,
    RobotState,
    RobotType,
    SafetyVerdict,
)
from robot_harness.embodiment.interface.sim import SimAgentServerClient


class SimEmbodimentAdapter:
    """Base EmbodimentAdapter backed by an external simulator agent_server.

    Satisfies the ``EmbodimentAdapter`` Protocol via structural typing. The
    physics step and any in-sim control loop run inside the sim process, not
    here (ADR-019); this class only transports high-level intent and samples
    state / frames.
    """

    sim_engine: str = "sim"
    default_url: str = "http://localhost:8800"

    def __init__(
        self,
        robot_id: str,
        *,
        robot_type: RobotType = "arm",
        server_url: str | None = None,
        dof: int = 6,
        scene: str = "",
        timeout_s: float = 5.0,
        client: SimAgentServerClient | None = None,
    ) -> None:
        self.robot_id = robot_id
        self.robot_type: RobotType = robot_type
        self._dof = dof
        self._scene = scene
        self._client = client or SimAgentServerClient(
            server_url or self.default_url,
            robot_id,
            sim_engine=self.sim_engine,
            timeout_s=timeout_s,
        )

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

    # -- sim-only lifecycle (not part of EmbodimentAdapter Protocol) -------

    async def reset(self) -> RobotState:
        """Reset the simulator to its initial state and return the new state."""
        raw = await self._client.reset()
        return self._to_state(raw)

    async def load_scene(self, scene: str) -> None:
        """Load a scene (MJCF / USD path or inline description) into the sim."""
        await self._client.load_scene(scene)
        self._scene = scene

    async def sim_time(self) -> float:
        """Return the current simulation clock in seconds."""
        raw = await self._client.get_sim_time()
        return float(raw.get("sim_time", 0.0))

    async def call_verb(self, verb: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Invoke an on-robot mid-loop verb on the sim agent_server (ADR-019).

        Returns the raw CompletionVerdict-shaped dict; the robot_sdk verb tool
        validates it. This is how a sim runs ``move_to_pose`` / reactive verbs
        internally while the harness only sees the completion event.
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
