"""Built-in NavigateToSkill — move robot base to a target position."""

from __future__ import annotations

from typing import Any

from robot_harness.skill.base import SkillManifest, SkillResult, Subtask
from robot_harness.skill.safety_class import SafetyClass
from robot_harness.tools.base import ToolContext, ToolRegistry
from robot_harness.tools.robot_sdk.verbs import ROBOT_SDK_LOCOMOTE_TO


class NavigateToSkill:
    """Navigate the robot base to a named or coordinate target.

    Hands a goal pose to the on-robot locomotion verb and awaits a
    CompletionVerdict; local planning and obstacle avoidance run on-robot.
    """

    manifest = SkillManifest(
        name="navigate_to",
        version="0.3.0",
        description="Navigate the robot to a target location",
        embodiment_compat=["mobile", "humanoid", "quadruped"],
        safety_class=SafetyClass.MEDIUM,
        required_tools=[ROBOT_SDK_LOCOMOTE_TO],
        tags=["navigation", "locomotion"],
        data_schema={
            "type": "object",
            "properties": {
                "robot_id": {"type": "string", "description": "Robot to navigate."},
                "description": {"type": "string"},
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_pose": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 3,
                            "description": "Map-frame goal [x, y] or [x, y, yaw].",
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
        # A motion skill must never fall back to a made-up goal: a missing
        # parameter is a failed subtask the Brain can correct, not a default move.
        target_pose: list[float] | None = subtask.parameters.get("target_pose")
        if not target_pose:
            return SkillResult(
                skill_name="navigate_to",
                skill_version=self.manifest.version,
                subtask_id=subtask.subtask_id,
                success=False,
                message="missing required parameter 'target_pose' [x, y] or [x, y, yaw]",
            )

        tool_ctx = ToolContext.create(robot_id, subtask_id=subtask.subtask_id)

        nav = await tools.get(ROBOT_SDK_LOCOMOTE_TO).invoke(
            {
                "robot_id": robot_id,
                "target_pose": target_pose,
            },
            tool_ctx,
        )

        if nav.success:
            return SkillResult(
                skill_name="navigate_to",
                skill_version=self.manifest.version,
                subtask_id=subtask.subtask_id,
                success=True,
                outcome="success",
                artifacts={"reached_pose": target_pose},
            )
        return SkillResult(
            skill_name="navigate_to",
            skill_version=self.manifest.version,
            subtask_id=subtask.subtask_id,
            success=False,
            outcome="failure",
            message=nav.error or "navigation failed",
        )

    async def rollback(self, ctx: Any) -> None:
        pass
