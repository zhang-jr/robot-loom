"""SendMessageTool — the Brain's fire-and-forget "report" to the user (ADR-023).

This is the report half of Talk: it delivers a one-way notification through the
Channel layer via the session-bound ``ctx.outbound`` handle. The tool stays
channel-agnostic — it never names Telegram vs CLI; routing is resolved upstream.
When no channel is wired (programmatic callers), ``ctx.outbound`` is None and the
message degrades to a tracer event so it remains observable.
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
        text = args.get("text", "")
        level = args.get("level", "info")
        tracer.event(
            "message.sent",
            trace_id=ctx.trace_id,
            robot_id=ctx.robot_id,
            level=level,
            text=text,
            channel=args.get("channel", "default"),
        )
        # Deliver through the session-bound channel when one is wired; otherwise the
        # tracer event above is the only record (programmatic / example callers).
        delivered = False
        if ctx.outbound is not None:
            await ctx.outbound.report(text, level)
            delivered = True
        latency = (time.monotonic() - t0) * 1000
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"delivered": delivered},
            latency_ms=latency,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass
