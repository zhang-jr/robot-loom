"""VlaServingTool — mock adapter for an external VLA serving runtime."""

from __future__ import annotations

import time
from typing import Any

from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class VlaServingTool:
    """Mock VLA (Vision-Language-Action) inference tool.

    Phase 1: returns a synthetic joint-space action without contacting any server.
    Phase 2: will POST observation + instruction to an external VLA serving runtime
             (e.g. OpenVLA, RoboFlamingo, π0) over HTTP/WebSocket.
    """

    name = "vla.infer_action"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="vla.infer_action",
        description=(
            "Query a VLA model to infer the next robot action given "
            "an image observation and a language instruction."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "image_source": {"type": "string"},
                "instruction": {"type": "string"},
                "robot_id": {"type": "string"},
                "num_actions": {
                    "type": "integer",
                    "default": 1,
                    "description": "Number of action steps to predict",
                },
            },
            "required": ["image_source", "instruction", "robot_id"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "description": (
                        "List of predicted 7D actions: "
                        "[dx, dy, dz, drx, dry, drz, gripper]"
                    ),
                    "items": {
                        "type": "array",
                        "items": [
                            {"type": "number"},
                            {"type": "number"},
                            {"type": "number"},
                            {"type": "number"},
                            {"type": "number"},
                            {"type": "number"},
                            {
                                "type": "number",
                                "enum": [0, 1],
                                "description": "Gripper command: 0 opens, 1 closes.",
                            },
                        ],
                        "additionalItems": False,
                        "minItems": 7,
                        "maxItems": 7,
                    },
                },
                "model_id": {"type": "string"},
            },
            "required": ["actions", "model_id"],
        },
    )

    @property
    def is_idempotent(self) -> bool:
        return False

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        t0 = time.monotonic()
        n = int(args.get("num_actions", 1))
        # Synthetic: small cartesian delta actions with an open gripper.
        actions = [[0.01, 0.0, -0.005, 0.0, 0.0, 0.0, 0.0]] * n
        latency = (time.monotonic() - t0) * 1000
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"actions": actions, "model_id": "mock-vla-v0"},
            latency_ms=latency,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass
