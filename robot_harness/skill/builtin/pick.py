"""Built-in PickSkill — hand off intent to the on-robot reactive grasp verb."""

from __future__ import annotations

from typing import Any

from robot_harness.memory.base import MemoryEntry
from robot_harness.observability.tracer import tracer
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
        version="0.4.0",
        description="Pick a named object from the workspace",
        embodiment_compat=["arm", "humanoid"],
        safety_class=SafetyClass.HIGH,
        required_tools=[ROBOT_SDK_REACTIVE_GRASP],
        tags=["manipulation", "pick"],
        data_schema={
            "type": "object",
            "properties": {
                "robot_id": {"type": "string", "description": "Robot that picks the object."},
                "description": {"type": "string"},
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": (
                                "Name/phrase of the object to grasp, e.g. 'red mug'. "
                                "Grounded by the on-robot perception stack."
                            ),
                        }
                    },
                    "required": ["object_name"],
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
        # An actuation-target parameter must never have a default: a fallback
        # phrase like "object" grounds to an arbitrary scene item and the robot
        # grasps whatever matched. Missing target is a failed subtask the Brain
        # can correct, never a guess.
        object_name = subtask.parameters.get("object_name")
        if not object_name:
            return _fail(subtask, "missing required parameter 'object_name' (grasp target phrase)")
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
                skill_version=self.manifest.version,
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
        skill_version=PickSkill.manifest.version,
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
    except Exception as exc:  # noqa: BLE001
        # Episode write is best-effort: failure must not fail the skill, but it
        # must be observable.
        tracer.event(
            "episode_write_failed",
            robot_id=subtask.robot_id,
            subtask_id=subtask.subtask_id,
            skill="pick",
            error=str(exc),
        )
