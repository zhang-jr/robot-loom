"""On-robot verb tools - reactive-skill minimal subset.

These tools wrap the **high-level verb endpoints** exposed by a per-robot
``agent_server``. Each verb runs the mid-loop perception-action closed loop
(5-30 Hz) *on the robot itself*; the harness calls the verb with a high-level
intent and awaits a single :class:`CompletionVerdict` -- it never observes the
per-tick control state.

Minimal subset:

    | Verb              | Intent                              | Idempotent |
    | ----------------- | ----------------------------------- | ---------- |
    | reactive_grasp    | "grasp this thing, you figure out   | No         |
    |                   | the approach"                       |            |
    | visual_servo_to   | "drive end-effector to this pose    | No         |
    |                   | using visual feedback"              |            |
    | move_to_pose      | "go to this pose, no vision needed" | No         |
    | home              | "return to home configuration"      | Yes        |

The harness side stays thin: build a verb tool, register it with the
:class:`ToolRegistry`, and let Brain pick it up by name. All four tools share
the same :data:`COMPLETION_VERDICT_SCHEMA` output shape so downstream
ReplanPolicy can treat them uniformly.

Status: when constructed with an ``adapters`` map whose adapter implements
:class:`SupportsVerbs` (e.g. a sim agent_server), :meth:`_RobotSdkVerbTool._dispatch`
POSTs the verb to the agent_server's ``/verb/{name}`` endpoint and the on-robot
mid-loop runs there. Without a verb-capable adapter the tool returns a simulated
verdict (harness end-to-end tests). The reference sim implements ``move_to_pose``;
other verbs return a "not implemented" verdict against it until their on-robot
recipe lands.

Compare and contrast (do not confuse):
    * :class:`robot_harness.tools.robot_sdk.RobotSdkTool` (``execute_action``)
      is the **low-level dispatch** verb: harness already has a concrete
      joint/cartesian target. Use it when no mid-loop is needed.
    * The verbs in this module own the mid-loop *on the robot*. Use them
      whenever the action requires sensor feedback the harness shouldn't see
      tick-by-tick.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from robot_harness.embodiment.base import EmbodimentCommand, SupportsVerbs
from robot_harness.errors import ToolCancelledError
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema

# ---------------------------------------------------------------------------
# Tool name constants — import these instead of inlining literals
# ---------------------------------------------------------------------------

ROBOT_SDK_REACTIVE_GRASP = "robot_sdk.reactive_grasp"
ROBOT_SDK_VISUAL_SERVO_TO = "robot_sdk.visual_servo_to"
ROBOT_SDK_MOVE_TO_POSE = "robot_sdk.move_to_pose"
ROBOT_SDK_LOCOMOTE_TO = "robot_sdk.locomote_to"
ROBOT_SDK_HOME = "robot_sdk.home"

VERB_TOOL_NAMES: tuple[str, ...] = (
    ROBOT_SDK_REACTIVE_GRASP,
    ROBOT_SDK_VISUAL_SERVO_TO,
    ROBOT_SDK_MOVE_TO_POSE,
    ROBOT_SDK_LOCOMOTE_TO,
    ROBOT_SDK_HOME,
)


# ---------------------------------------------------------------------------
# Shared types — CompletionVerdict (ADR-019 §266)
# ---------------------------------------------------------------------------

CompletionOutcome = Literal["success", "partial", "failed"]


class CompletionVerdict(BaseModel):
    """Result of an on-robot verb execution.

    The verb-level dual to :class:`CriticVerdict`:

        * **CompletionVerdict** -- emitted by the on-robot agent_server,
          authoritative on internal state ("did the action finish?").
        * **CriticVerdict** -- emitted by an external supervisor, authoritative
          on task-level progress ("did the task actually advance?").

    ``ReplanPolicy`` consumes both and applies the joint-decision rule.

    ``extra="allow"`` so an agent_server may return verb-specific fields
    (``grasped_object_id``, ``final_grasp_pose``, ``residual_error_m``, …) on top
    of the shared core; they flow through to the tool output unchanged.
    """

    model_config = ConfigDict(extra="allow")

    outcome: CompletionOutcome
    evidence: str = ""
    robot_state_snapshot: dict[str, Any] = Field(default_factory=dict)
    duration_s: float = 0.0
    aborted_by: Literal["none", "cancel", "safety", "timeout", "hardware"] = "none"


COMPLETION_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": ["success", "partial", "failed"]},
        "evidence": {"type": "string"},
        "robot_state_snapshot": {"type": "object"},
        "duration_s": {"type": "number"},
        "aborted_by": {
            "type": "string",
            "enum": ["none", "cancel", "safety", "timeout", "hardware"],
        },
    },
    "required": ["outcome"],
}


# ---------------------------------------------------------------------------
# Schema fragments shared across verbs
# ---------------------------------------------------------------------------

_POSE6_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 6,
    "maxItems": 6,
    "description": "[x, y, z, rx, ry, rz] — meters and radians, robot base frame.",
}

_CONSTRAINTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "High-level execution constraints; on-robot runtime enforces.",
    "properties": {
        "max_velocity": {"type": "number", "description": "m/s or rad/s upper bound"},
        "max_force": {"type": "number", "description": "Newtons; for contact-rich verbs"},
        "timeout_s": {"type": "number", "default": 30.0},
        "stop_on_contact": {"type": "boolean", "default": False},
    },
    "additionalProperties": True,
}

_TARGET_HINT_SCHEMA: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "kind": {"const": "phrase"},
                "phrase": {
                    "type": "string",
                    "description": "Natural-language target (e.g. 'the red mug').",
                },
            },
            "required": ["kind", "phrase"],
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "bbox"},
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "[x1, y1, x2, y2] normalized to [0, 1].",
                },
                "camera": {"type": "string", "default": "wrist_camera"},
            },
            "required": ["kind", "bbox"],
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "object_id"},
                "object_id": {
                    "type": "string",
                    "description": "ObjectMemory id; on-robot runtime resolves via memory query.",
                },
            },
            "required": ["kind", "object_id"],
        },
    ],
    "description": (
        "Target description for the verb.  Three forms supported so the harness "
        "can produce hints from any of: language grounding, slow-loop detection, "
        "or memory lookup."
    ),
}


# ---------------------------------------------------------------------------
# Base implementation shared by all four verbs
# ---------------------------------------------------------------------------


class _RobotSdkVerbTool:
    """Common skeleton for on-robot verb tools.

    Concrete subclasses set ``name``, ``schema``, and override ``_simulate()``
    to produce a verb-specific mock :class:`CompletionVerdict`. The future
    real transport (HTTP/WS POST to the per-robot ``agent_server``) will be
    wired into :meth:`_dispatch` -- mock mode short-circuits to ``_simulate``.
    """

    name: ClassVar[str] = ""
    schema: ClassVar[ToolSchema]
    backend: ToolBackend = "native"

    # Verbs are *not* idempotent by default — calling reactive_grasp twice
    # produces two attempted grasps.  Subclasses (e.g. HomeTool) may override.
    _idempotent: ClassVar[bool] = False

    def __init__(self, adapters: dict[str, Any] | None = None) -> None:
        """Args:
        adapters: robot_id → EmbodimentAdapter. When the resolved adapter
            supports verbs (``SupportsVerbs``, e.g. a sim agent_server), the
            verb is POSTed to its ``/verb/{name}`` endpoint and the on-robot
            mid-loop runs there. Otherwise the tool returns a simulated verdict.
        """
        self._adapters = adapters or {}

    @property
    def is_idempotent(self) -> bool:
        return self._idempotent

    @property
    def is_cancellable(self) -> bool:
        return True

    @property
    def hardware_bound(self) -> bool:
        return True

    def to_safety_command(self, args: dict[str, Any], ctx: ToolContext) -> EmbodimentCommand | None:
        """Expose the verb's target for pre-dispatch validation.

        Default: no harness-checkable target, so the on-robot reflex is
        authoritative. Verbs that move to a known pose override this.
        """
        return None

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        t0 = time.monotonic()
        if ctx.is_cancelled:
            raise ToolCancelledError(
                f"Verb '{self.name}' cancelled before dispatch",
                tool_name=self.name,
            )
        verdict = await self._dispatch(args, ctx)
        latency = (time.monotonic() - t0) * 1000
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=verdict.outcome == "success",
            output=verdict.model_dump(),
            latency_ms=latency,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        """Signal the on-robot agent_server to abort and fall back to a safe pose.

        Today: sets the local cancel event only. Later this will additionally
        POST an ``/abort`` request to the on-robot agent_server so the in-robot
        mid/tight-loop can unwind to a safe pose before reporting the final
        :class:`CompletionVerdict` with ``aborted_by="cancel"``.
        """
        ctx.cancel()

    # ----- override hooks ---------------------------------------------------

    async def _dispatch(self, args: dict[str, Any], ctx: ToolContext) -> CompletionVerdict:
        """Dispatch to the agent_server's verb endpoint, or simulate if unwired.

        The verb name is the tool name without the ``robot_sdk.`` prefix.
        """
        robot_id = args.get("robot_id", ctx.robot_id)
        adapter = self._adapters.get(robot_id)
        if isinstance(adapter, SupportsVerbs):
            verb = self.name.split(".", 1)[1]
            resp = await adapter.call_verb(verb, args)
            return CompletionVerdict.model_validate(resp)
        return await self._simulate(args, ctx)

    async def _simulate(self, args: dict[str, Any], ctx: ToolContext) -> CompletionVerdict:
        """Subclasses produce a verb-specific mock verdict."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. reactive_grasp ----------------------------------------------------------
# ---------------------------------------------------------------------------


class ReactiveGraspTool(_RobotSdkVerbTool):
    """High-level grasp verb. The on-robot runtime owns the perception-action loop.

    Harness produces a ``target_hint`` (from perception or memory), supplies
    optional constraints, and awaits a :class:`CompletionVerdict`.  Per ADR-019
    the harness does *not* see per-tick state — visual servoing, grasp
    micro-adjustment, and contact-force fusion all happen on the robot.
    """

    name = ROBOT_SDK_REACTIVE_GRASP
    schema = ToolSchema(
        name=ROBOT_SDK_REACTIVE_GRASP,
        description=(
            "Grasp an object given a high-level target hint. The on-robot "
            "agent_server runs the 5-30 Hz perception-action loop (visual "
            "servo + grasp micro-adjustment + force fusion); the harness "
            "supplies intent and awaits a CompletionVerdict."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "robot_id": {"type": "string"},
                "target_hint": _TARGET_HINT_SCHEMA,
                "approach_pose": {
                    **_POSE6_SCHEMA,
                    "description": (
                        "Optional pre-grasp pose; if omitted, on-robot runtime "
                        "computes from current state + target_hint."
                    ),
                },
                "constraints": _CONSTRAINTS_SCHEMA,
            },
            "required": ["robot_id", "target_hint"],
        },
        output_schema={
            "type": "object",
            "properties": {
                **COMPLETION_VERDICT_SCHEMA["properties"],
                "grasped_object_id": {
                    "type": "string",
                    "description": (
                        "Identifier the on-robot runtime assigns to the grasped "
                        "object; harness can write it to ObjectMemory."
                    ),
                },
                "final_grasp_pose": _POSE6_SCHEMA,
            },
            "required": COMPLETION_VERDICT_SCHEMA["required"],
        },
    )

    def to_safety_command(self, args: dict[str, Any], ctx: ToolContext) -> EmbodimentCommand | None:
        pose = args.get("approach_pose")
        if not pose:
            return None
        return EmbodimentCommand(
            robot_id=args.get("robot_id", ctx.robot_id),
            command_type="cartesian",
            values=list(pose),
        )

    async def _simulate(self, args: dict[str, Any], ctx: ToolContext) -> CompletionVerdict:
        target = args.get("target_hint", {})
        return CompletionVerdict(
            outcome="success",
            evidence=f"mock reactive_grasp on hint={target!r}",
            duration_s=1.8,
            robot_state_snapshot={
                "robot_id": args.get("robot_id", ctx.robot_id),
                "gripper_state": 1.0,
            },
        )


# ---------------------------------------------------------------------------
# 2. visual_servo_to ---------------------------------------------------------
# ---------------------------------------------------------------------------


class VisualServoToTool(_RobotSdkVerbTool):
    """Drive the end-effector to a target pose using on-robot visual feedback.

    Use when the slow-loop pose estimate may drift and a closed-loop correction
    is required (e.g. tracking a moving target, or aligning to a fixture whose
    pose has measurement noise).  Differs from :class:`MoveToPoseTool` in that
    the on-robot runtime keeps a visual feedback loop running until tolerance
    is satisfied — the harness should not micromanage that loop.
    """

    name = ROBOT_SDK_VISUAL_SERVO_TO
    schema = ToolSchema(
        name=ROBOT_SDK_VISUAL_SERVO_TO,
        description=(
            "Drive the end-effector to a target pose with on-robot visual "
            "feedback. Loop runs at 5-30 Hz on the robot; harness only sees "
            "the final CompletionVerdict."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "robot_id": {"type": "string"},
                "target_pose": _POSE6_SCHEMA,
                "tolerance_m": {
                    "type": "number",
                    "default": 0.005,
                    "minimum": 0.0,
                    "description": "Translational tolerance in meters.",
                },
                "tolerance_rad": {
                    "type": "number",
                    "default": 0.02,
                    "minimum": 0.0,
                    "description": "Rotational tolerance in radians.",
                },
                "tracking_camera": {
                    "type": "string",
                    "default": "wrist_camera",
                },
                "max_duration_s": {"type": "number", "default": 10.0},
            },
            "required": ["robot_id", "target_pose"],
        },
        output_schema={
            "type": "object",
            "properties": {
                **COMPLETION_VERDICT_SCHEMA["properties"],
                "final_pose": _POSE6_SCHEMA,
                "residual_error_m": {"type": "number"},
                "residual_error_rad": {"type": "number"},
            },
            "required": COMPLETION_VERDICT_SCHEMA["required"],
        },
    )

    def to_safety_command(self, args: dict[str, Any], ctx: ToolContext) -> EmbodimentCommand | None:
        pose = args.get("target_pose")
        if not pose:
            return None
        return EmbodimentCommand(
            robot_id=args.get("robot_id", ctx.robot_id),
            command_type="cartesian",
            values=list(pose),
        )

    async def _simulate(self, args: dict[str, Any], ctx: ToolContext) -> CompletionVerdict:
        target = args.get("target_pose", [0.0] * 6)
        return CompletionVerdict(
            outcome="success",
            evidence=f"mock visual_servo_to converged on target={target}",
            duration_s=0.9,
            robot_state_snapshot={
                "robot_id": args.get("robot_id", ctx.robot_id),
                "final_pose": target,
            },
        )


# ---------------------------------------------------------------------------
# 3. move_to_pose ------------------------------------------------------------
# ---------------------------------------------------------------------------


class MoveToPoseTool(_RobotSdkVerbTool):
    """Plan-and-execute motion to a target pose, no visual feedback loop.

    Use when the target pose is trusted (e.g. a known stowage rack, a fiducial
    that's already been registered).  The on-robot runtime still owns
    trajectory planning, joint servo, and collision avoidance — this verb does
    NOT degrade to harness-side trajectory control.

    Contrast with :class:`VisualServoToTool` (closes a visual loop) and
    :class:`RobotSdkTool` (low-level ``execute_action`` for already-resolved
    targets the harness wants to send as-is).
    """

    name = ROBOT_SDK_MOVE_TO_POSE
    schema = ToolSchema(
        name=ROBOT_SDK_MOVE_TO_POSE,
        description=(
            "Plan and execute motion to a target end-effector pose.  On-robot "
            "runtime owns trajectory planning and joint servo; no visual "
            "feedback loop (use visual_servo_to for that)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "robot_id": {"type": "string"},
                "target_pose": _POSE6_SCHEMA,
                "frame": {
                    "type": "string",
                    "enum": ["base", "world", "tool"],
                    "default": "base",
                    "description": "Reference frame for target_pose.",
                },
                "constraints": _CONSTRAINTS_SCHEMA,
            },
            "required": ["robot_id", "target_pose"],
        },
        output_schema={
            "type": "object",
            "properties": {
                **COMPLETION_VERDICT_SCHEMA["properties"],
                "final_pose": _POSE6_SCHEMA,
                "planned_path_length_m": {"type": "number"},
            },
            "required": COMPLETION_VERDICT_SCHEMA["required"],
        },
    )

    def to_safety_command(self, args: dict[str, Any], ctx: ToolContext) -> EmbodimentCommand | None:
        pose = args.get("target_pose")
        if not pose:
            return None
        return EmbodimentCommand(
            robot_id=args.get("robot_id", ctx.robot_id),
            command_type="cartesian",
            values=list(pose),
        )

    async def _simulate(self, args: dict[str, Any], ctx: ToolContext) -> CompletionVerdict:
        target = args.get("target_pose", [0.0] * 6)
        return CompletionVerdict(
            outcome="success",
            evidence=f"mock move_to_pose reached target={target}",
            duration_s=1.2,
            robot_state_snapshot={
                "robot_id": args.get("robot_id", ctx.robot_id),
                "final_pose": target,
            },
        )


# ---------------------------------------------------------------------------
# 4. home --------------------------------------------------------------------
# ---------------------------------------------------------------------------


class HomeTool(_RobotSdkVerbTool):
    """Return the robot to its home / parked configuration.

    Idempotent: calling ``home`` twice in a row is well-defined (already home →
    no-op success).  Useful for skill rollback and end-of-task cleanup.
    """

    name = ROBOT_SDK_HOME
    _idempotent = True
    schema = ToolSchema(
        name=ROBOT_SDK_HOME,
        description=(
            "Return the robot to its home / parked configuration.  Idempotent: "
            "calling twice is safe; the on-robot runtime treats already-home "
            "as a no-op success."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "robot_id": {"type": "string"},
                "open_gripper": {"type": "boolean", "default": True},
            },
            "required": ["robot_id"],
        },
        output_schema={
            "type": "object",
            "properties": {**COMPLETION_VERDICT_SCHEMA["properties"]},
            "required": COMPLETION_VERDICT_SCHEMA["required"],
        },
    )

    async def _simulate(self, args: dict[str, Any], ctx: ToolContext) -> CompletionVerdict:
        return CompletionVerdict(
            outcome="success",
            evidence="mock home reached",
            duration_s=2.0,
            robot_state_snapshot={
                "robot_id": args.get("robot_id", ctx.robot_id),
                "is_home": True,
            },
        )


# ---------------------------------------------------------------------------
# 5. locomote_to -------------------------------------------------------------
# ---------------------------------------------------------------------------


class LocomoteToTool(_RobotSdkVerbTool):
    """Drive the robot base to a goal pose with on-robot obstacle avoidance.

    The on-robot agent_server owns local planning, obstacle avoidance, and the
    SLAM/odometry feedback loop. The harness supplies a goal pose and awaits a
    single CompletionVerdict; it does not stream velocity commands.
    """

    name = ROBOT_SDK_LOCOMOTE_TO
    schema = ToolSchema(
        name=ROBOT_SDK_LOCOMOTE_TO,
        description=(
            "Drive the robot base to a goal pose. The on-robot agent_server runs "
            "local planning and obstacle avoidance; the harness supplies the goal "
            "and awaits a CompletionVerdict."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "robot_id": {"type": "string"},
                "target_pose": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 3,
                    "description": "[x, y] or [x, y, yaw] goal in the map frame (meters, radians).",
                },
                "constraints": _CONSTRAINTS_SCHEMA,
            },
            "required": ["robot_id", "target_pose"],
        },
        output_schema={
            "type": "object",
            "properties": {
                **COMPLETION_VERDICT_SCHEMA["properties"],
                "final_pose": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Reached base pose in the map frame.",
                },
            },
            "required": COMPLETION_VERDICT_SCHEMA["required"],
        },
    )

    def to_safety_command(self, args: dict[str, Any], ctx: ToolContext) -> EmbodimentCommand | None:
        goal = args.get("target_pose")
        if not goal:
            return None
        return EmbodimentCommand(
            robot_id=args.get("robot_id", ctx.robot_id),
            command_type="locomotion",
            values=list(goal),
        )

    async def _simulate(self, args: dict[str, Any], ctx: ToolContext) -> CompletionVerdict:
        goal = args.get("target_pose", [0.0, 0.0])
        return CompletionVerdict(
            outcome="success",
            evidence=f"locomote_to reached goal={goal}",
            duration_s=4.0,
            robot_state_snapshot={
                "robot_id": args.get("robot_id", ctx.robot_id),
                "base_pose": goal,
            },
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_robot_sdk_verb_tools(
    adapters: dict[str, Any] | None = None,
) -> list[_RobotSdkVerbTool]:
    """Build the minimal-subset on-robot verb tools.

    Pass ``adapters`` (robot_id → EmbodimentAdapter) to dispatch verbs to a live
    agent_server (e.g. a sim's ``/verb/{name}``) when the adapter supports verbs;
    omit it for simulated verdicts (harness end-to-end tests).
    """
    return [
        ReactiveGraspTool(adapters),
        VisualServoToTool(adapters),
        MoveToPoseTool(adapters),
        LocomoteToTool(adapters),
        HomeTool(adapters),
    ]


__all__ = [
    "COMPLETION_VERDICT_SCHEMA",
    "ROBOT_SDK_HOME",
    "ROBOT_SDK_LOCOMOTE_TO",
    "ROBOT_SDK_MOVE_TO_POSE",
    "ROBOT_SDK_REACTIVE_GRASP",
    "ROBOT_SDK_VISUAL_SERVO_TO",
    "VERB_TOOL_NAMES",
    "CompletionOutcome",
    "CompletionVerdict",
    "HomeTool",
    "LocomoteToTool",
    "MoveToPoseTool",
    "ReactiveGraspTool",
    "VisualServoToTool",
    "build_robot_sdk_verb_tools",
]
