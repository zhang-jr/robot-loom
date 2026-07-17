"""On-robot verb tools - reactive-skill minimal subset.

These tools wrap the **high-level verb endpoints** exposed by a per-robot
``agent_server``. Each verb runs the mid-loop perception-action closed loop
(5-30 Hz) *on the robot itself*; the harness calls the verb with a high-level
intent and awaits a single :class:`CompletionVerdict` -- it never observes the
per-tick control state.

Minimal subset:

    | Verb              | Intent                                | Idempotent |
    | ----------------- | ------------------------------------- | ---------- |
    | reactive_grasp    | "grasp this thing, you figure out     | No         |
    |                   | the approach"                         |            |
    | visual_servo_to   | "drive end-effector to this pose      | No         |
    |                   | using visual feedback"                |            |
    | move_to_pose      | "go to this pose, no vision needed"   | No         |
    | move_joints       | "drive the joints to this exact       | No         |
    |                   | configuration, joint-space direct"    |            |
    | locomote_to       | "drive the base to this goal pose"    | No         |
    | home              | "return to home configuration"        | Yes        |

The harness side stays thin: build a verb tool, register it with the
:class:`ToolRegistry`, and let Brain pick it up by name. All six tools share
the same :data:`COMPLETION_VERDICT_SCHEMA` output shape so downstream
ReplanPolicy can treat them uniformly.

Status: when constructed with an ``adapters`` map (real, sim, or mock-client
agent_server — all implement :class:`SupportsVerbs`),
:meth:`_RobotSdkVerbTool._dispatch` POSTs the verb to the addressed robot's
``/verb/{name}`` endpoint and the on-robot mid-loop runs there. A robot_id not
in the wired fleet raises a typed ``RobotOfflineError`` — never a simulated
success; only a tool built with no adapters at all falls back to a
simulated verdict (harness plumbing tests). Which verbs an agent_server actually implements
is discovered live via ``/health.available_verbs`` (``SupportsVerbs.available_verbs``);
at planning time the harness excludes unadvertised verb tools from the Brain's
vocabulary (:func:`unavailable_verb_tool_names` → ``export_for_brain(exclude_names=…)``,
see ``HarnessContext.unavailable_tool_names``), and a verb that still reaches an
unavailable backend returns a ``failed`` CompletionVerdict rather than raising.

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
from collections.abc import Collection
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from robot_harness.embodiment.base import EmbodimentCommand, SupportsVerbs
from robot_harness.errors import HardwareNotReadyError, RobotOfflineError, ToolCancelledError
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema

# ---------------------------------------------------------------------------
# Tool name constants — import these instead of inlining literals
# ---------------------------------------------------------------------------

ROBOT_SDK_REACTIVE_GRASP = "robot_sdk.reactive_grasp"
ROBOT_SDK_VISUAL_SERVO_TO = "robot_sdk.visual_servo_to"
ROBOT_SDK_MOVE_TO_POSE = "robot_sdk.move_to_pose"
ROBOT_SDK_MOVE_JOINTS = "robot_sdk.move_joints"
ROBOT_SDK_LOCOMOTE_TO = "robot_sdk.locomote_to"
ROBOT_SDK_HOME = "robot_sdk.home"

VERB_TOOL_NAMES: tuple[str, ...] = (
    ROBOT_SDK_REACTIVE_GRASP,
    ROBOT_SDK_VISUAL_SERVO_TO,
    ROBOT_SDK_MOVE_TO_POSE,
    ROBOT_SDK_MOVE_JOINTS,
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

_JOINTS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 1,
    "description": (
        "Target joint positions in radians, one per joint in the robot's joint "
        "order; length must match the arm's DOF."
    ),
}

_GRIPPER_SCHEMA: dict[str, Any] = {
    "type": "number",
    "description": (
        "Gripper command to hold during the motion. Value semantics are "
        "backend-specific — consult the robot's workspace docs for its "
        "open/close values. Omit to keep the backend default; pose/joint "
        "motions never actuate the gripper on their own."
    ),
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
    to produce a verb-specific mock :class:`CompletionVerdict` for the
    no-fleet-wired case; with a fleet wired, :meth:`_dispatch` POSTs the verb
    to the addressed robot's ``agent_server`` and never simulates.
    """

    name: ClassVar[str] = ""
    schema: ClassVar[ToolSchema]
    backend: ToolBackend = "native"

    # Verbs are *not* idempotent by default — calling reactive_grasp twice
    # produces two attempted grasps.  Subclasses (e.g. HomeTool) may override.
    _idempotent: ClassVar[bool] = False

    def __init__(self, adapters: dict[str, Any] | None = None) -> None:
        """Args:
        adapters: robot_id → EmbodimentAdapter. With adapters wired, the verb
            is POSTed to the addressed robot's ``/verb/{name}`` endpoint and
            the on-robot mid-loop runs there; a robot_id not in the map raises
            instead of simulating. Only a tool built with no adapters at all
            returns simulated verdicts (plumbing tests).
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
        success = verdict.outcome == "success"
        # A non-success verdict must surface WHY as the error string — evidence
        # and aborted_by are what the Brain (and skills reading ``.error``)
        # replan on (ISS-032); the full verdict still rides in ``output``.
        error = None
        if not success:
            error = (
                f"{self.name} outcome={verdict.outcome} "
                f"(aborted_by={verdict.aborted_by}): "
                f"{verdict.evidence or 'no evidence reported'}"
            )
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=success,
            output=verdict.model_dump(),
            error=error,
            error_type=None if success else "VerbFailed",
            latency_ms=latency,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        """Signal the on-robot agent_server to abort and fall back to a safe pose.

        Sets the local cancel event first (so it holds even if the abort round-trip
        fails), then POSTs ``/abort`` to the robot's agent_server via the adapter so
        the in-robot mid/tight-loop unwinds to a safe pose and the in-flight verb
        reports ``aborted_by="cancel"``. When no verb-capable adapter is wired (mock
        / offline), only the local event is set — there is nothing on-robot to abort.
        """
        ctx.cancel()
        adapter = self._adapters.get(ctx.robot_id)
        abort = getattr(adapter, "abort", None)
        if abort is not None:
            await abort(trace_id=ctx.trace_id)

    # ----- override hooks ---------------------------------------------------

    async def _dispatch(self, args: dict[str, Any], ctx: ToolContext) -> CompletionVerdict:
        """Dispatch to the addressed robot's agent_server verb endpoint.

        The verb name is the tool name without the ``robot_sdk.`` prefix.

        With a fleet wired, a robot_id outside it is a typed failure the Brain
        can correct — never a fabricated ``success`` verdict for a robot that
        does not exist. ``_simulate`` is reachable only when the tool was
        built with no fleet at all (standalone / harness plumbing tests).
        """
        robot_id = args.get("robot_id", ctx.robot_id)
        verb = self.name.split(".", 1)[1]
        if not self._adapters:
            return await self._simulate(args, ctx)
        adapter = self._adapters.get(robot_id)
        if adapter is None:
            raise RobotOfflineError(
                f"unknown robot_id '{robot_id}' for verb '{verb}' — "
                f"wired fleet: {sorted(self._adapters)}",
                robot_id=robot_id,
                tool_name=self.name,
                module_name="tools.robot_sdk.verbs",
            )
        if not isinstance(adapter, SupportsVerbs):
            raise HardwareNotReadyError(
                f"adapter for robot '{robot_id}' does not implement on-robot verbs "
                f"(SupportsVerbs) — cannot run '{verb}'",
                robot_id=robot_id,
                tool_name=self.name,
                module_name="tools.robot_sdk.verbs",
            )
        resp = await adapter.call_verb(verb, args)
        try:
            return CompletionVerdict.model_validate(resp)
        except ValidationError as exc:
            # A response that is not a CompletionVerdict is "not speaking
            # the contract" — the same typed failure as unreachable (the
            # wire clients set the precedent), never a raw ValidationError
            # escaping into the loop (ISS-034).
            raise RobotOfflineError(
                f"agent_server returned a malformed CompletionVerdict for verb '{verb}': {exc}",
                robot_id=robot_id,
                tool_name=self.name,
                module_name="tools.robot_sdk.verbs",
            ) from exc

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

    Contrast with :class:`VisualServoToTool` (closes a visual loop),
    :class:`MoveJointsTool` (joint-space direct drive for already-known joint
    configurations), and :class:`RobotSdkTool` (low-level ``execute_action``
    for already-resolved targets the harness wants to send as-is).
    """

    name = ROBOT_SDK_MOVE_TO_POSE
    schema = ToolSchema(
        name=ROBOT_SDK_MOVE_TO_POSE,
        description=(
            "Plan and execute motion to a target end-effector pose.  On-robot "
            "runtime owns trajectory planning and joint servo; no visual "
            "feedback loop (use visual_servo_to for that). If the target is "
            "a known joint configuration rather than a pose, prefer "
            "move_joints — joint-space drive tracks more reliably."
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
                "gripper": _GRIPPER_SCHEMA,
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
# 4. move_joints -------------------------------------------------------------
# ---------------------------------------------------------------------------


class MoveJointsTool(_RobotSdkVerbTool):
    """Drive the joints directly to a target configuration, joint-space direct.

    Real-hardware feedback: joint-space direct drive tracks more reliably than
    Cartesian end-pose tracking, so prefer this verb whenever the target joint
    configuration is already *known* — taught waypoints and named
    configurations recorded in the robot's workspace docs. Targets computed at
    runtime (perception, grasp poses, memory) only exist as Cartesian poses
    and must go through :class:`MoveToPoseTool` instead: the harness does no
    IK, and the on-robot runtime only accepts joint targets as-is here.

    The full joint target is harness-checkable, so ``to_safety_command``
    exposes it as a ``joint`` command — SafetyEnvelope validates every joint
    against ``safety.joint_limits_rad`` pre-dispatch (honest skip when no
    limits are configured).
    """

    name = ROBOT_SDK_MOVE_JOINTS
    schema = ToolSchema(
        name=ROBOT_SDK_MOVE_JOINTS,
        description=(
            "Drive the arm joints directly to a target joint configuration "
            "(radians); the on-robot runtime interpolates in joint space. "
            "More reliable than Cartesian end-pose tracking — prefer it over "
            "move_to_pose whenever the target joint configuration is already "
            "known (e.g. a taught waypoint). Targets computed at runtime as "
            "Cartesian poses must use move_to_pose instead; the harness does "
            "no IK."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "robot_id": {"type": "string"},
                "target_joints": _JOINTS_SCHEMA,
                "gripper": _GRIPPER_SCHEMA,
                "constraints": _CONSTRAINTS_SCHEMA,
            },
            "required": ["robot_id", "target_joints"],
        },
        output_schema={
            "type": "object",
            "properties": {
                **COMPLETION_VERDICT_SCHEMA["properties"],
                "final_joints": _JOINTS_SCHEMA,
                "residual_error_rad": {
                    "type": "number",
                    "description": "Worst-joint residual to the target, radians.",
                },
            },
            "required": COMPLETION_VERDICT_SCHEMA["required"],
        },
    )

    def to_safety_command(self, args: dict[str, Any], ctx: ToolContext) -> EmbodimentCommand | None:
        joints = args.get("target_joints")
        if not joints:
            return None
        return EmbodimentCommand(
            robot_id=args.get("robot_id", ctx.robot_id),
            command_type="joint",
            values=list(joints),
        )

    async def _simulate(self, args: dict[str, Any], ctx: ToolContext) -> CompletionVerdict:
        target = args.get("target_joints", [0.0] * 6)
        return CompletionVerdict(
            outcome="success",
            evidence=f"mock move_joints reached target={target}",
            duration_s=1.0,
            robot_state_snapshot={
                "robot_id": args.get("robot_id", ctx.robot_id),
                "final_joints": target,
            },
        )


# ---------------------------------------------------------------------------
# 5. home --------------------------------------------------------------------
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
# 6. locomote_to -------------------------------------------------------------
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
        # SafetyEnvelope checks the goal against the map-frame geofence
        # (``safety.map_bounds_m``). With no geofence configured the envelope
        # honestly skips (audit outcome "skipped", never a false "passed") and the
        # on-robot nav stack (local planning + obstacle avoidance) stays
        # authoritative either way — the geofence only bounds the goal position.
        pose = args.get("target_pose")
        if not pose:
            return None
        return EmbodimentCommand(
            robot_id=args.get("robot_id", ctx.robot_id),
            command_type="locomotion",
            values=list(pose),
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


def unavailable_verb_tool_names(available_verbs: Collection[str] | None) -> frozenset[str]:
    """Verb TOOL names to exclude from the Brain's planning vocabulary.

    ``available_verbs`` is the fleet's live allowlist of bare verb names (from
    ``/health.available_verbs``, see ``AgentServerAdapter.available_verbs``):

    * ``None`` — availability unknown (some backend doesn't advertise): exclude
      nothing; the call-time ``SupportsVerbs`` / simulated-verdict path stays
      authoritative. This never removes capability that works today.
    * otherwise — every verb tool whose bare verb is absent is excluded, so the
      Brain never plans a verb no robot can currently run. An empty collection
      (all bridges explicitly down) excludes all verb tools.

    Returns prefixed TOOL names (``robot_sdk.<verb>``) ready to feed to
    ``ToolRegistry.export_for_brain(exclude_names=...)`` and
    ``SkillRegistry.export_for_brain(unavailable_tools=...)`` — the filtering
    happens on tool names BEFORE spec serialization, so no caller needs to know
    any Brain profile's wire shape.
    """
    if available_verbs is None:
        return frozenset()
    allowed = set(available_verbs)
    return frozenset(
        tool_name for tool_name in VERB_TOOL_NAMES if tool_name.split(".", 1)[1] not in allowed
    )


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
        MoveJointsTool(adapters),
        LocomoteToTool(adapters),
        HomeTool(adapters),
    ]


__all__ = [
    "COMPLETION_VERDICT_SCHEMA",
    "ROBOT_SDK_HOME",
    "ROBOT_SDK_LOCOMOTE_TO",
    "ROBOT_SDK_MOVE_JOINTS",
    "ROBOT_SDK_MOVE_TO_POSE",
    "ROBOT_SDK_REACTIVE_GRASP",
    "ROBOT_SDK_VISUAL_SERVO_TO",
    "VERB_TOOL_NAMES",
    "CompletionOutcome",
    "CompletionVerdict",
    "HomeTool",
    "LocomoteToTool",
    "MoveJointsTool",
    "MoveToPoseTool",
    "ReactiveGraspTool",
    "VisualServoToTool",
    "build_robot_sdk_verb_tools",
    "unavailable_verb_tool_names",
]
