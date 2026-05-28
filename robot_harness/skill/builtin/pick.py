"""Built-in PickSkill — hand off intent to the on-robot reactive grasp verb."""

from __future__ import annotations

from typing import Any

from robot_harness.memory.base import MemoryEntry
from robot_harness.skill.base import SkillManifest, SkillResult, Subtask
from robot_harness.skill.safety_class import SafetyClass
from robot_harness.tools.base import ToolContext, ToolRegistry
from robot_harness.tools.robot_sdk.verbs import ROBOT_SDK_REACTIVE_GRASP


class PickSkill:
    """Pick a named object from the workspace.

    Per ADR-019, the on-robot agent_server owns the 5-30 Hz perception-action
    loop. This skill just hands the high-level intent (a phrase) to
    ``robot_sdk.reactive_grasp`` and awaits a single CompletionVerdict — it
    does NOT run harness-side detection or compute bounding boxes.
    """

    manifest = SkillManifest(
        name="pick",
        version="0.3.0",
        description="Pick a named object from the workspace",
        embodiment_compat=["arm", "humanoid"],
        safety_class=SafetyClass.HIGH,
        required_tools=[ROBOT_SDK_REACTIVE_GRASP],
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

        grasp = await tools.get(ROBOT_SDK_REACTIVE_GRASP).invoke(
            {
                "robot_id": robot_id,
                "target_hint": {"kind": "phrase", "phrase": object_name},
            },
            tool_ctx,
        )

        if grasp.success:
            result = SkillResult(
                skill_name="pick",
                skill_version="0.3.0",
                subtask_id=subtask.subtask_id,
                success=True,
                outcome="success",
                artifacts={"picked_object": object_name, "grasp_result": grasp.output or {}},
            )
        else:
            result = _fail(subtask, grasp.error or "reactive grasp failed")

        await _write_episode(ctx, subtask, result, object_name)
        return result

    async def rollback(self, ctx: Any) -> None:
        pass


def _fail(subtask: Subtask, message: str) -> SkillResult:
    return SkillResult(
        skill_name="pick",
        skill_version="0.3.0",
        subtask_id=subtask.subtask_id,
        success=False,
        message=message,
    )


async def _write_episode(ctx: Any, subtask: Subtask, result: SkillResult, object_name: str) -> None:
    try:
        await ctx.memory.write(
            MemoryEntry(
                memory_type="episodic",
                robot_id=subtask.robot_id,
                content={
                    "skill": "pick",
                    "object": object_name,
                    "outcome": result.outcome,
                    "subtask_id": subtask.subtask_id,
                },
                tags=["skill:pick", result.outcome, subtask.robot_id],
            )
        )
    except Exception:  # noqa: BLE001
        pass
