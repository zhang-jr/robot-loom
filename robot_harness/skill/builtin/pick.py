"""Built-in PickSkill — perceive → estimate grasp → execute pick."""

from __future__ import annotations

from typing import Any

from robot_harness.skill.base import SkillManifest, SkillResult, Subtask
from robot_harness.skill.safety_class import SafetyClass
from robot_harness.tools.base import ToolContext, ToolRegistry


class PickSkill:
    """Pick an object from the workspace.

    Tool call sequence:
    1. perception.detect_objects  — find the target object
    2. grasp.estimate_pose        — compute grasp pose
    3. robot_sdk.execute_action   — dispatch the pick motion
    """

    manifest = SkillManifest(
        name="pick",
        version="0.1.0",
        description="Pick a named object from the workspace",
        embodiment_compat=["arm", "humanoid"],
        safety_class=SafetyClass.HIGH,
        required_tools=[
            "perception.detect_objects",
            "grasp.estimate_pose",
            "robot_sdk.execute_action",
        ],
        tags=["manipulation", "pick"],
    )

    async def can_handle(self, subtask: Subtask, ctx: Any) -> bool:
        kw = subtask.description.lower()
        return "pick" in kw or "grasp" in kw or "grab" in kw

    async def execute(
        self,
        subtask: Subtask,
        tools: ToolRegistry,
        ctx: Any,
    ) -> SkillResult:
        robot_id = subtask.robot_id
        object_name = subtask.parameters.get("object_name", "object")

        tool_ctx = ToolContext.create(robot_id, subtask_id=subtask.subtask_id)

        detect = await tools.get("perception.detect_objects").invoke(
            {"image_source": "wrist_camera", "query": object_name},
            tool_ctx,
        )
        if not detect.success:
            return _fail(subtask, detect.error or "detection failed")

        grasp = await tools.get("grasp.estimate_pose").invoke(
            {"detections": detect.output or {}, "object_name": object_name},
            tool_ctx,
        )
        if not grasp.success:
            return _fail(subtask, grasp.error or "grasp estimation failed")

        pose: list[float] = (grasp.output or {}).get("pose", [0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
        act = await tools.get("robot_sdk.execute_action").invoke(
            {
                "robot_id": robot_id,
                "command_type": "cartesian",
                "values": pose,
                "gripper_close": True,
            },
            tool_ctx,
        )

        if act.success:
            return SkillResult(
                skill_name="pick",
                skill_version="0.1.0",
                subtask_id=subtask.subtask_id,
                success=True,
                outcome="success",
                artifacts={"picked_object": object_name, "pose": pose},
            )
        return _fail(subtask, act.error or "action dispatch failed")

    async def rollback(self, ctx: Any) -> None:
        # Phase 2: open gripper + return to home pose
        pass


def _fail(subtask: Subtask, message: str) -> SkillResult:
    return SkillResult(
        skill_name="pick",
        skill_version="0.1.0",
        subtask_id=subtask.subtask_id,
        success=False,
        outcome="failure",
        message=message,
    )
