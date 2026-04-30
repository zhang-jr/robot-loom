"""Generic filesystem tool (NativeTool) — read and write local files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema

_MAX_READ_BYTES = 1_048_576  # 1 MiB


class ReadFileTool:
    """Read a local file and return its contents as text."""

    name = "fs_read"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="fs_read",
        description="Read a local file and return its text content.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            p = Path(args["path"])
            content = p.read_bytes()[:_MAX_READ_BYTES].decode(errors="replace")
            return ToolResult(
                tool_name=self.name,
                trace_id=ctx.trace_id,
                success=True,
                output={"content": content, "size_bytes": p.stat().st_size},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_name=self.name,
                trace_id=ctx.trace_id,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    async def cancel(self, ctx: ToolContext) -> None:
        pass


class WriteFileTool:
    """Write text content to a local file."""

    name = "fs_write"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="fs_write",
        description="Write text to a local file (overwrites if exists).",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    )

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            p = Path(args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"], encoding="utf-8")
            return ToolResult(
                tool_name=self.name,
                trace_id=ctx.trace_id,
                success=True,
                output={"path": str(p), "size_bytes": p.stat().st_size},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_name=self.name,
                trace_id=ctx.trace_id,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    async def cancel(self, ctx: ToolContext) -> None:
        pass
