"""AnyGraspTool — mock adapter for AnyGrasp / GraspAnything grasp pose server."""

from __future__ import annotations

import time
from typing import Any

from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class AnyGraspTool:
    """Mock grasp pose estimation tool.

    Phase 1: returns a synthetic 6-DOF grasp pose without contacting any server.
    Phase 2: will POST to an external AnyGrasp / GraspAnything server.
    """

    name = "grasp.estimate_pose"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="grasp.estimate_pose",
        description="Estimate a 6-DOF grasp pose for a detected object.",
        input_schema={
            "type": "object",
            "properties": {
                "detections": {
                    "type": "object",
                    "description": "Output from perception.detect_objects",
                },
                "object_name": {"type": "string"},
                "grasp_type": {
                    "type": "string",
                    "enum": ["top_down", "side", "any"],
                    "default": "any",
                },
            },
            "required": ["detections"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "pose": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z, rx, ry, rz] in robot base frame (metres / radians)",
                },
                "score": {"type": "number"},
                "grasp_type": {"type": "string"},
            },
        },
    )

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        t0 = time.monotonic()
        latency = (time.monotonic() - t0) * 1000
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={
                "pose": [0.45, 0.02, 0.30, 0.0, 0.0, 0.0],
                "score": 0.87,
                "grasp_type": args.get("grasp_type", "top_down"),
            },
            latency_ms=latency,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass
