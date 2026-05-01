"""VlacCriticTool — mock adapter for a VLAC-style progress critic server."""

from __future__ import annotations

import time
from typing import Any

from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class VlacCriticTool:
    """Mock VLAC-style progress critic tool.

    Phase 1: always returns "progress" with synthetic confidence.
    Phase 2: will POST current + reference frames to an external critic server.
    """

    name = "critic.judge_progress"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="critic.judge_progress",
        description=(
            "Evaluate task progress by comparing current observation against "
            "the goal description. Returns one of: progress | completion | failure | unchanged."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task_description": {"type": "string"},
                "current_image_source": {"type": "string"},
                "reference_image_source": {
                    "type": "string",
                    "description": "Optional goal image for comparison",
                },
            },
            "required": ["task_description"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "enum": ["progress", "completion", "failure", "unchanged"],
                },
                "confidence": {"type": "number"},
                "evidence": {"type": "string"},
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
                "state": "progress",
                "confidence": 0.75,
                "evidence": "mock: task appears to be progressing",
            },
            latency_ms=latency,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass
