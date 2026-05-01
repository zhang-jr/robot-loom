"""MemoryUpsertTool — tool wrapper for Memory.write() / upsert()."""

from __future__ import annotations

import time
from typing import Any

from robot_harness.memory.base import Memory, MemoryEntry
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class MemoryUpsertTool:
    """Write or update a memory entry of any sub-type."""

    name = "memory.upsert"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="memory.upsert",
        description="Write or update an entry in the robot memory store.",
        input_schema={
            "type": "object",
            "properties": {
                "memory_type": {
                    "type": "string",
                    "enum": ["object", "place", "episodic", "semantic"],
                },
                "content": {
                    "type": "object",
                    "description": "Structured content to store",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["memory_type", "content"],
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
        entry = MemoryEntry(
            memory_type=args["memory_type"],
            robot_id=ctx.robot_id,
            content=args["content"],
            tags=args.get("tags", []),
        )
        memory_id = await self._memory.write(entry)
        latency = (time.monotonic() - t0) * 1000
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"memory_id": memory_id},
            latency_ms=latency,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass
