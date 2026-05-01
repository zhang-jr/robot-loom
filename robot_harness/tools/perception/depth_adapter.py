"""DepthEstimationTool — mock adapter for monocular depth estimation server."""

from __future__ import annotations

import time
from typing import Any

from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class DepthEstimationTool:
    """Mock depth estimation tool.

    Phase 1: returns a synthetic depth value without contacting any server.
    Phase 2: will call an external depth server (e.g. Depth-Anything or ZoeDepth).
    """

    name = "perception.estimate_depth"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="perception.estimate_depth",
        description="Estimate metric depth for a bounding box or pixel coordinate.",
        input_schema={
            "type": "object",
            "properties": {
                "image_source": {"type": "string"},
                "bbox_xyxy": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Bounding box [x1, y1, x2, y2] in pixels",
                },
                "pixel_xy": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Single pixel coordinate [x, y]",
                },
            },
            "required": ["image_source"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "depth_m": {"type": "number", "description": "Metric depth in metres"},
                "confidence": {"type": "number"},
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
            output={"depth_m": 0.45, "confidence": 0.88},
            latency_ms=latency,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass
