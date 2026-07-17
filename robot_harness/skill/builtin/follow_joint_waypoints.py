"""Built-in FollowJointWaypointsSkill — sequential joint-space waypoint motion.

Executes a sparse sequence of taught joint waypoints (e.g. a placement table
from the robot's workspace docs) by issuing one ``robot_sdk.move_joints`` verb
per waypoint: send a point, await its CompletionVerdict, and only advance when
the verdict is ``success``. On any non-success verdict the arm HALTS at the
current waypoint and the skill reports which point failed — it never skips
ahead. Fine interpolation between waypoints is the on-robot bridge's job; the
harness only sequences the semantic points.

Each waypoint may carry a ``gripper`` value so gripper transitions (open to
release, close to hold) ride on the same motion call — there is no separate
gripper dispatch. Every waypoint passes the SafetyEnvelope joint-limit check
before dispatch via the gated registry.
"""

from __future__ import annotations

from typing import Any

from robot_harness.skill.base import SkillManifest, SkillResult, Subtask
from robot_harness.skill.safety_class import SafetyClass
from robot_harness.tools.base import ToolContext, ToolRegistry
from robot_harness.tools.robot_sdk.verbs import ROBOT_SDK_MOVE_JOINTS

_WAYPOINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "joints": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 1,
            "description": (
                "Target joint positions in radians, one per joint in the "
                "robot's joint order; length must match the arm's DOF."
            ),
        },
        "gripper": {
            "type": "number",
            "description": (
                "Gripper command to hold while moving to this waypoint "
                "(backend-specific value, see the robot's workspace docs). "
                "Omit to keep the backend default."
            ),
        },
    },
    "required": ["joints"],
}


class FollowJointWaypointsSkill:
    """Drive the arm through a sequence of joint waypoints, halting on failure.

    Tool call sequence: one ``robot_sdk.move_joints`` per waypoint, strictly
    sequential — waypoint N+1 is dispatched only after waypoint N's verdict is
    ``success``.
    """

    manifest = SkillManifest(
        name="follow_joint_waypoints",
        version="0.1.0",
        description=(
            "Move the arm through a sequence of joint-space waypoints (taught "
            "points, radians), one move_joints per point, confirming arrival "
            "before advancing; halts in place and reports on the first "
            "failed waypoint. Each waypoint may carry a gripper value, so "
            "placement sequences (descend, release, retreat) run as one call."
        ),
        embodiment_compat=["arm", "humanoid"],
        safety_class=SafetyClass.HIGH,
        required_tools=[ROBOT_SDK_MOVE_JOINTS],
        tags=["manipulation", "motion", "waypoints"],
        data_schema={
            "type": "object",
            "properties": {
                "robot_id": {"type": "string", "description": "Robot that runs the sequence."},
                "description": {"type": "string"},
                "parameters": {
                    "type": "object",
                    "properties": {
                        "waypoints": {
                            "type": "array",
                            "items": _WAYPOINT_SCHEMA,
                            "minItems": 1,
                            "description": (
                                "Ordered joint waypoints, executed strictly in sequence."
                            ),
                        },
                        "constraints": {
                            "type": "object",
                            "description": (
                                "Optional execution constraints (e.g. "
                                "timeout_s) applied to every waypoint's "
                                "move_joints call."
                            ),
                            "additionalProperties": True,
                        },
                    },
                    "required": ["waypoints"],
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
        # A motion skill must never invent a target: malformed waypoints are a
        # failed subtask the Brain can correct, detected before any tool call.
        waypoints = subtask.parameters.get("waypoints")
        if not waypoints or not isinstance(waypoints, list):
            return _fail(
                subtask,
                "missing required parameter 'waypoints' (non-empty list of {joints, gripper?})",
            )
        for i, wp in enumerate(waypoints):
            if not isinstance(wp, dict) or not wp.get("joints"):
                return _fail(
                    subtask,
                    f"waypoint {i + 1}/{len(waypoints)} has no 'joints' "
                    "(expected {joints: [radians...], gripper?: number})",
                )

        constraints = subtask.parameters.get("constraints")
        tool_ctx = ToolContext.create(robot_id, subtask_id=subtask.subtask_id)
        move_joints = tools.get(ROBOT_SDK_MOVE_JOINTS)

        completed = 0
        last_output: dict[str, Any] | None = None
        for i, wp in enumerate(waypoints):
            args: dict[str, Any] = {"robot_id": robot_id, "target_joints": list(wp["joints"])}
            if "gripper" in wp:
                args["gripper"] = wp["gripper"]
            if constraints:
                args["constraints"] = constraints

            res = await move_joints.invoke(args, tool_ctx)
            if not res.success:
                # Halt in place: remaining waypoints are NOT dispatched. The
                # arm stays where this verdict left it; the Brain replans from
                # the reported index + evidence.
                return SkillResult(
                    skill_name=self.manifest.name,
                    skill_version=self.manifest.version,
                    subtask_id=subtask.subtask_id,
                    success=False,
                    outcome="partial" if completed else "failure",
                    message=(
                        f"halted at waypoint {i + 1}/{len(waypoints)}: "
                        f"{res.error or 'move_joints failed'}"
                    ),
                    artifacts={
                        "waypoints_completed": completed,
                        "failed_waypoint_index": i,
                        "failed_verdict": res.output or {},
                    },
                )
            completed += 1
            last_output = res.output

        return SkillResult(
            skill_name=self.manifest.name,
            skill_version=self.manifest.version,
            subtask_id=subtask.subtask_id,
            success=True,
            outcome="success",
            artifacts={
                "waypoints_completed": completed,
                "final_verdict": last_output or {},
            },
        )

    async def rollback(self, ctx: Any) -> None:
        # Halting in place IS the failure semantics (per taught-waypoint
        # placement rules); there is no generically safe reverse motion.
        pass


def _fail(subtask: Subtask, message: str) -> SkillResult:
    return SkillResult(
        skill_name=FollowJointWaypointsSkill.manifest.name,
        skill_version=FollowJointWaypointsSkill.manifest.version,
        subtask_id=subtask.subtask_id,
        success=False,
        message=message,
    )
