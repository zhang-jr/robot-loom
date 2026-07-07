"""Built-in PlaceSkill — move to target via on-robot verb → release gripper."""

from __future__ import annotations

from typing import Any

from robot_harness.skill.base import SkillManifest, SkillResult, Subtask
from robot_harness.skill.safety_class import SafetyClass
from robot_harness.tools.base import ToolContext, ToolRegistry
from robot_harness.tools.robot_sdk.verbs import ROBOT_SDK_MOVE_TO_POSE


class PlaceSkill:
    """Place the held object at a target location.

    Tool call sequence:
    1. robot_sdk.move_to_pose     — on-robot verb handles trajectory planning
    2. robot_sdk.execute_action   — open gripper (hand_grasp)
    """

    manifest = SkillManifest(
        name="place",
        version="0.3.0",
        description="Place a held object at the specified target location",
        embodiment_compat=["arm", "humanoid"],
        safety_class=SafetyClass.HIGH,
        required_tools=[ROBOT_SDK_MOVE_TO_POSE, "robot_sdk.execute_action"],
        tags=["manipulation", "place"],
        data_schema={
            "type": "object",
            "properties": {
                "robot_id": {"type": "string", "description": "Robot that places the object."},
                "description": {"type": "string"},
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_pose": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 6,
                            "maxItems": 6,
                            "description": "Target pose [x, y, z, rx, ry, rz] in meters/radians.",
                        }
                    },
                    "required": ["target_pose"],
                },
            },
            "required": ["robot_id", "parameters"],
        },
    )

    async def execute(
        self,
        subtask: Subtask,
        tools: ToolRegistry,
        ctx: Any,
    ) -> SkillResult:
        robot_id = subtask.robot_id
        # A motion skill must never fall back to a made-up pose: a missing
        # parameter is a failed subtask the Brain can correct, not a default move.
        target_pose: list[float] | None = subtask.parameters.get("target_pose")
        if not target_pose:
            return _fail(subtask, "missing required parameter 'target_pose' [x, y, z, rx, ry, rz]")

        tool_ctx = ToolContext.create(robot_id, subtask_id=subtask.subtask_id)

        move = await tools.get(ROBOT_SDK_MOVE_TO_POSE).invoke(
            {
                "robot_id": robot_id,
                "target_pose": target_pose,
            },
            tool_ctx,
        )
        if not move.success:
            return _fail(subtask, move.error or "move to target failed")

        release = await tools.get("robot_sdk.execute_action").invoke(
            {
                "robot_id": robot_id,
                "command_type": "hand_grasp",
                "values": [0.0],
            },
            tool_ctx,
        )

        if release.success:
            return SkillResult(
                skill_name="place",
                skill_version=self.manifest.version,
                subtask_id=subtask.subtask_id,
                success=True,
                outcome="success",
                artifacts={"target_pose": target_pose},
            )
        return _fail(subtask, release.error or "gripper release failed")

    async def rollback(self, ctx: Any) -> None:
        pass


def _fail(subtask: Subtask, message: str) -> SkillResult:
    return SkillResult(
        skill_name="place",
        skill_version=PlaceSkill.manifest.version,
        subtask_id=subtask.subtask_id,
        success=False,
        message=message,
    )
