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
        version="0.2.0",
        description="Navigate the robot to a target location",
        embodiment_compat=["mobile", "humanoid", "quadruped"],
        safety_class=SafetyClass.MEDIUM,
        required_tools=[ROBOT_SDK_LOCOMOTE_TO],
        tags=["navigation", "locomotion"],
    )

    async def can_handle(self, subtask: Subtask, ctx: Any) -> bool:
        kw = subtask.description.lower()
        return "navigate" in kw or "go to" in kw or "move to" in kw or "walk to" in kw

    async def execute(
        self,
        subtask: Subtask,
        tools: ToolRegistry,
        ctx: Any,
    ) -> SkillResult:
        robot_id = subtask.robot_id
        target_pose: list[float] = subtask.parameters.get("target_pose", [1.0, 0.0, 0.0])

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
                skill_version="0.2.0",
                subtask_id=subtask.subtask_id,
                success=True,
                outcome="success",
                artifacts={"reached_pose": target_pose},
            )
        return SkillResult(
            skill_name="navigate_to",
            skill_version="0.2.0",
            subtask_id=subtask.subtask_id,
            success=False,
            outcome="failure",
            message=nav.error or "navigation failed",
        )

    async def rollback(self, ctx: Any) -> None:
        pass
