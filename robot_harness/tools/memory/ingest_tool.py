"""MemoryIngestTool — bulk ingest a list of memory entries."""

from __future__ import annotations

import time
from typing import Any

from robot_harness.memory.base import Memory, MemoryEntry
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class MemoryIngestTool:
    """Batch-write multiple memory entries in one tool call."""

    name = "memory.ingest"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="memory.ingest",
        description="Batch-write multiple memory entries (e.g. after a perception sweep).",
        input_schema={
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "memory_type": {"type": "string"},
                            "content": {"type": "object"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["memory_type", "content"],
                    },
                },
            },
            "required": ["entries"],
        },
    )

    def __init__(self, memory: Memory) -> None:
        self._memory = memory

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        t0 = time.monotonic()
        ids: list[str] = []
        for raw in args.get("entries", []):
            entry = MemoryEntry(
                memory_type=raw["memory_type"],
                robot_id=ctx.robot_id,
                content=raw["content"],
                tags=raw.get("tags", []),
            )
            mid = await self._memory.write(entry)
            ids.append(mid)
        latency = (time.monotonic() - t0) * 1000
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"memory_ids": ids, "count": len(ids)},
            latency_ms=latency,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass
