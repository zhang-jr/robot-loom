"""Middleware base and composition helper.

Middleware wraps a Tool and intercepts invoke().  The chain is built once at
registration time.  Order matters — first in the list = outermost wrapper.

Example::

    tool = build_chain(raw_tool, [TraceMiddleware, TimeoutMiddleware])
"""

from __future__ import annotations

from typing import Any

from robot_harness.tools.base import Tool, ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class ToolMiddleware:
    """Wraps an inner Tool, delegating non-overridden methods transparently."""

    def __init__(self, inner: Tool) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def schema(self) -> ToolSchema:
        return self._inner.schema

    @property
    def backend(self) -> ToolBackend:
        return self._inner.backend

    @property
    def is_idempotent(self) -> bool:
        return self._inner.is_idempotent

    @property
    def is_cancellable(self) -> bool:
        return self._inner.is_cancellable

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return await self._inner.invoke(args, ctx)

    async def cancel(self, ctx: ToolContext) -> None:
        await self._inner.cancel(ctx)

    def __getattr__(self, item: str) -> Any:
        # Forward capability markers (hardware_bound, to_safety_command,
        # brain_visible, …) the base class does not model: wrapping a tool must
        # never strip them — a middleware-wrapped hardware tool that lost
        # ``hardware_bound`` would silently bypass the SafetyEnvelope gate.
        return getattr(self._inner, item)


def build_chain(tool: Tool, middleware_classes: list[type[ToolMiddleware]]) -> Tool:
    """Wrap *tool* with *middleware_classes*, outermost first."""
    result: Tool = tool
    for mw_cls in reversed(middleware_classes):
        wrapped = mw_cls(result)
        result = wrapped  # ToolMiddleware satisfies Tool (all @property members match)
    return result
