"""Timeout middleware — enforces ctx.timeout_s on every invocation."""

from __future__ import annotations

import asyncio
from typing import Any

from robot_harness.errors import ToolTimeoutError
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.middleware.base import ToolMiddleware


class TimeoutMiddleware(ToolMiddleware):
    """Cancels the inner invocation if it exceeds ``ctx.timeout_s``."""

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            return await asyncio.wait_for(
                self._inner.invoke(args, ctx),
                timeout=ctx.timeout_s,
            )
        except TimeoutError as exc:
            raise ToolTimeoutError(
                f"Tool '{self.name}' timed out after {ctx.timeout_s}s",
                tool_name=self.name,
                trace_id=ctx.trace_id,
                robot_id=ctx.robot_id,
            ) from exc
