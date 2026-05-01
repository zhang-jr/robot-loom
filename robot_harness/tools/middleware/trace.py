"""Trace middleware — emits a structured span for every tool invocation."""

from __future__ import annotations

import time
from typing import Any

from robot_harness.observability.tracer import tracer
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.middleware.base import ToolMiddleware


class TraceMiddleware(ToolMiddleware):
    """Emits one span per invoke() call with name, latency, and outcome."""

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        start = time.perf_counter()
        outcome = "ok"
        try:
            result = await self._inner.invoke(args, ctx)
            if not result.success:
                outcome = "error"
            return result
        except Exception as exc:
            outcome = type(exc).__name__
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            tracer.event(
                "tool.invoke",
                tool_name=self.name,
                trace_id=ctx.trace_id,
                robot_id=ctx.robot_id,
                subtask_id=ctx.subtask_id,
                latency_ms=round(elapsed_ms, 3),
                outcome=outcome,
            )
