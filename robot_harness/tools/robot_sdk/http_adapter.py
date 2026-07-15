"""RobotSdkTool — harness-side dispatch tool for a per-robot HTTP agent_server.

This is the low-level "execute a concrete command" flavor (see package docstring
and ADR-019). It hands a fully-resolved EmbodimentCommand to the on-robot
agent_server and returns a handle; it does NOT drive control loops or
visual-servoing loops. Reactive verbs that need a tight perception-action loop
will live as separate tools in this package and call distinct verb endpoints
on the same agent_server.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from robot_harness.embodiment.base import EmbodimentAdapter, EmbodimentCommand
from robot_harness.errors import RobotOfflineError
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class RobotSdkTool:
    """Low-level per-robot action dispatch tool.

    Boundary (ADR-019):
      • This tool transports a HIGH-LEVEL command and returns a handle.
      • The on-robot agent_server (or a sim agent_server) is responsible for
        trajectory planning, joint servo, hand-eye calibration, IMU/force
        fusion, and e-stop reflex.
      • The harness only awaits a completion event; it never observes per-tick
        state through this tool.
      • Visual servoing or "approach until X" loops must NOT be implemented by
        repeatedly invoking this tool — wrap them as a reactive verb on the
        agent_server instead (separate tool in this package).

    Backend selection:
      • Constructed with an ``adapters`` map (robot_id → EmbodimentAdapter) it
        dispatches *for real* — to a real robot or a sim agent_server — then
        samples ``get_state()`` so the loop sees the post-step state. This is
        the path that actually advances a sim ``env.step()`` (ADR-021). A
        robot_id not in the map raises instead of mocking.
      • Constructed without it, it returns a mock handle (offline / unit tests).
    """

    name = "robot_sdk.execute_action"
    backend: ToolBackend = "native"
    # Low-level override dispatch: hidden from the Brain's planning vocabulary so
    # it does not hand-assemble open-loop control. Reactive verbs are the default
    # act surface; this stays invocable for skills / explicit override paths.
    brain_visible = False
    schema = ToolSchema(
        name="robot_sdk.execute_action",
        description=(
            "Dispatch an action command to the robot's control layer. "
            "Command types: joint | cartesian | delta | locomotion | hand_grasp."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "robot_id": {"type": "string"},
                "command_type": {
                    "type": "string",
                    "enum": ["joint", "cartesian", "delta", "locomotion", "hand_grasp"],
                },
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Joint positions / cartesian pose / locomotion target",
                },
                "gripper_close": {
                    "type": "boolean",
                    "default": False,
                },
            },
            "required": ["robot_id", "command_type", "values"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "action_id": {"type": "string"},
                "estimated_duration_s": {"type": "number"},
                "robot_id": {"type": "string"},
                "state": {
                    "type": "object",
                    "description": "Post-dispatch RobotState sample (live backend only).",
                },
            },
        },
    )

    def __init__(self, adapters: dict[str, EmbodimentAdapter] | None = None) -> None:
        """Args:
        adapters: robot_id → EmbodimentAdapter. When provided, ``invoke``
            dispatches to the real/sim backend and samples state; an unknown
            robot_id raises instead of mocking. When omitted, ``invoke``
            returns a mock handle (offline / tests).
        """
        self._adapters = adapters or {}

    @property
    def is_idempotent(self) -> bool:
        return False

    @property
    def is_cancellable(self) -> bool:
        return True

    @property
    def hardware_bound(self) -> bool:
        return True

    def to_safety_command(self, args: dict[str, Any], ctx: ToolContext) -> EmbodimentCommand:
        """Map the dispatched command directly onto the safety command."""
        return EmbodimentCommand(
            robot_id=args.get("robot_id", ctx.robot_id),
            command_type=args.get("command_type", "joint"),
            values=args.get("values", []),
            extra={
                k: v for k, v in args.items() if k not in ("robot_id", "command_type", "values")
            },
        )

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        t0 = time.monotonic()
        robot_id = args.get("robot_id", ctx.robot_id)

        if not self._adapters:
            # No fleet wired at all — mock dispatch (offline / unit tests).
            latency = (time.monotonic() - t0) * 1000
            return ToolResult(
                tool_name=self.name,
                trace_id=ctx.trace_id,
                success=True,
                output={
                    "action_id": str(uuid.uuid4()),
                    "estimated_duration_s": 1.5,
                    "robot_id": robot_id,
                },
                latency_ms=latency,
            )

        adapter = self._adapters.get(robot_id)
        if adapter is None:
            # A fleet IS wired: an unknown robot_id must be a typed failure the
            # Brain can correct — never a fabricated success handle.
            raise RobotOfflineError(
                f"unknown robot_id '{robot_id}' for execute_action — "
                f"wired fleet: {sorted(self._adapters)}",
                robot_id=robot_id,
                tool_name=self.name,
                module_name="tools.robot_sdk.http_adapter",
            )

        # Live dispatch: send the high-level command, then sample the resulting
        # state so the AgentLoop sees the post-step world (SafetyEnvelope has
        # already run in AgentLoop before invoke — ADR-007).
        cmd = self.to_safety_command(args, ctx)
        handle = await adapter.dispatch(cmd)
        state = await adapter.get_state()
        latency = (time.monotonic() - t0) * 1000
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={
                "action_id": handle.action_id,
                "estimated_duration_s": handle.estimated_duration_s,
                "robot_id": robot_id,
                "state": state.model_dump(),
            },
            latency_ms=latency,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        """Abort the dispatched action on the robot, not just the local flag.

        ``dispatch`` returns while the command is still executing on-robot, so a
        cancel (e.g. the AgentLoop's sibling-cancel after a safety violation in
        a parallel call) must reach the robot's ``/abort`` — same route the verb
        tools take (ISS-035). Local event first so it holds even if the abort
        round-trip fails; no adapter (mock/offline) means nothing to abort.
        """
        ctx.cancel()
        adapter = self._adapters.get(ctx.robot_id)
        abort = getattr(adapter, "abort", None)
        if abort is not None:
            await abort(trace_id=ctx.trace_id)
