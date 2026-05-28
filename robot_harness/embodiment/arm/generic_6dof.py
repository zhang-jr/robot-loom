"""Generic6DofArm — mock EmbodimentAdapter for a 6/7-DOF robot arm.

Currently an in-process stub that satisfies the EmbodimentAdapter Protocol.
TODO: delegate to HttpAgentServerClient once a real per-robot agent_server is
running.
"""

from __future__ import annotations

import uuid

from robot_harness.embodiment.base import (
    DispatchHandle,
    EmbodimentCommand,
    Frame,
    RobotState,
    RobotType,
    SafetyVerdict,
)
from robot_harness.embodiment.interface.http import HttpAgentServerClient


class Generic6DofArm:
    """Mock EmbodimentAdapter for a 6/7-DOF robot arm.

    Satisfies EmbodimentAdapter Protocol via structural typing.
    """

    robot_type: RobotType = "arm"

    def __init__(
        self,
        robot_id: str,
        dof: int = 6,
        server_url: str = "http://localhost:8765",
    ) -> None:
        self.robot_id = robot_id
        self._dof = dof
        self._client = HttpAgentServerClient(server_url, robot_id)

    async def get_camera_frame(self, camera: str) -> Frame:
        raw = await self._client.get_camera_frame(camera)
        return Frame(**raw)

    async def get_state(self) -> RobotState:
        raw = await self._client.get_state()
        return RobotState(
            robot_id=raw["robot_id"],
            joint_positions=raw.get("joint_positions", [0.0] * self._dof),
            joint_velocities=raw.get("joint_velocities", [0.0] * self._dof),
            end_effector_pose=raw.get("end_effector_pose", {}),
            gripper_state=raw.get("gripper_state", 0.0),
        )

    async def dispatch(self, cmd: EmbodimentCommand) -> DispatchHandle:
        raw = await self._client.dispatch(cmd.model_dump())
        return DispatchHandle(
            robot_id=self.robot_id,
            action_id=raw.get("action_id", str(uuid.uuid4())),
            estimated_duration_s=raw.get("estimated_duration_s", 1.5),
        )

    async def safety_check(self, cmd: EmbodimentCommand) -> SafetyVerdict:
        raw = await self._client.safety_check(cmd.model_dump())
        return SafetyVerdict(
            passed=raw.get("passed", True),
            reason=raw.get("reason", ""),
            violated_rules=raw.get("violated_rules", []),
        )
