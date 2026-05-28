"""SendMessageTool — generic log/notification tool (NativeTool).

Currently writes to the structured tracer.
TODO: route to configured channels (Telegram / Feishu / Discord).
"""

from __future__ import annotations

import time
from typing import Any

from robot_harness.observability.tracer import tracer
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class SendMessageTool:
    """Send a notification message to the operator channel."""

    name = "generic.send_message"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="generic.send_message",
        description="Send a status or notification message to the operator.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "level": {
                    "type": "string",
                    "enum": ["info", "warning", "error"],
                    "default": "info",
                },
                "channel": {
                    "type": "string",
                    "description": "Target channel id (optional)",
                },
            },
            "required": ["text"],
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
        tracer.event(
            "message.sent",
            trace_id=ctx.trace_id,
            robot_id=ctx.robot_id,
            level=args.get("level", "info"),
            text=args.get("text", ""),
            channel=args.get("channel", "default"),
        )
        latency = (time.monotonic() - t0) * 1000
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"delivered": True},
            latency_ms=latency,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass
