"""Built-in PlaceSkill — move to target pose → release gripper."""

from __future__ import annotations

from typing import Any

from robot_harness.skill.base import SkillManifest, SkillResult, Subtask
from robot_harness.skill.safety_class import SafetyClass
from robot_harness.tools.base import ToolContext, ToolRegistry


class PlaceSkill:
    """Place the held object at a target location.

    Tool call sequence:
    1. robot_sdk.execute_action  — move to target position (cartesian)
    2. robot_sdk.execute_action  — open gripper (hand_grasp with open=True)
    """

    manifest = SkillManifest(
        name="place",
        version="0.1.0",
        description="Place a held object at the specified target location",
        embodiment_compat=["arm", "humanoid"],
        safety_class=SafetyClass.HIGH,
        required_tools=["robot_sdk.execute_action"],
        tags=["manipulation", "place"],
    )

    async def can_handle(self, subtask: Subtask, ctx: Any) -> bool:
        kw = subtask.description.lower()
        return "place" in kw or "put" in kw or "release" in kw or "drop" in kw

    async def execute(
        self,
        subtask: Subtask,
        tools: ToolRegistry,
        ctx: Any,
    ) -> SkillResult:
        robot_id = subtask.robot_id
        target_pose: list[float] = subtask.parameters.get(
            "target_pose", [0.5, 0.2, 0.1, 0.0, 0.0, 0.0]
        )

        tool_ctx = ToolContext.create(robot_id, subtask_id=subtask.subtask_id)

        move = await tools.get("robot_sdk.execute_action").invoke(
            {
                "robot_id": robot_id,
                "command_type": "cartesian",
                "values": target_pose,
                "gripper_close": True,
            },
            tool_ctx,
        )
        if not move.success:
            return _fail(subtask, move.error or "move to target failed")

        release = await tools.get("robot_sdk.execute_action").invoke(
            {
                "robot_id": robot_id,
                "command_type": "hand_grasp",
                "values": [0.0],  # 0.0 = fully open
                "gripper_close": False,
            },
            tool_ctx,
        )

        if release.success:
            return SkillResult(
                skill_name="place",
                skill_version="0.1.0",
                subtask_id=subtask.subtask_id,
                success=True,
                outcome="success",
                artifacts={"target_pose": target_pose},
            )
        return _fail(subtask, release.error or "gripper release failed")

    async def rollback(self, ctx: Any) -> None:
        # Phase 2: re-close gripper if release happened
        pass


def _fail(subtask: Subtask, message: str) -> SkillResult:
    return SkillResult(
        skill_name="place",
        skill_version="0.1.0",
        subtask_id=subtask.subtask_id,
        success=False,
        outcome="failure",
        message=message,
    )
