"""RobotSdkTool — mock adapter for a per-robot HTTP agent_server."""

from __future__ import annotations

import time
import uuid
from typing import Any

from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class RobotSdkTool:
    """Mock per-robot action dispatch tool.

    Phase 1: simulates command dispatch without contacting any hardware.
    Phase 2: will POST EmbodimentCommand to a per-robot HTTP/WebSocket agent_server.
    """

    name = "robot_sdk.execute_action"
    backend: ToolBackend = "native"
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
            },
        },
    )

    @property
    def is_idempotent(self) -> bool:
        return False

    @property
    def is_cancellable(self) -> bool:
        return True

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        t0 = time.monotonic()
        action_id = str(uuid.uuid4())
        latency = (time.monotonic() - t0) * 1000
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={
                "action_id": action_id,
                "estimated_duration_s": 1.5,
                "robot_id": args.get("robot_id", ctx.robot_id),
            },
            latency_ms=latency,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        ctx.cancel()
