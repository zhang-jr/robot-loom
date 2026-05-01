"""Cancel middleware — checks the cancel token before forwarding invoke."""

from __future__ import annotations

from typing import Any

from robot_harness.errors import ToolCancelledError
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.middleware.base import ToolMiddleware


class CancelMiddleware(ToolMiddleware):
    """Raises :exc:`ToolCancelledError` if ``ctx.is_cancelled`` before call."""

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.is_cancelled:
            raise ToolCancelledError(
                f"Tool '{self.name}' invocation cancelled before start",
                tool_name=self.name,
                trace_id=ctx.trace_id,
                robot_id=ctx.robot_id,
            )
        return await self._inner.invoke(args, ctx)
