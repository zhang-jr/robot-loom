"""MemoryQueryTool — tool wrapper for Memory.query()."""

from __future__ import annotations

import time
from typing import Any

from robot_harness.memory.base import Memory, MemoryQuery
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class MemoryQueryTool:
    """Query any sub-memory (object / place / episodic / semantic) via a unified tool."""

    name = "memory.query"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="memory.query",
        description="Query the robot memory store for relevant entries.",
        input_schema={
            "type": "object",
            "properties": {
                "memory_type": {
                    "type": "string",
                    "enum": ["object", "place", "episodic", "semantic"],
                },
                "text": {"type": "string", "description": "Natural language query"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tag filter",
                },
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["memory_type", "text"],
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
        q = MemoryQuery(
            memory_type=args["memory_type"],
            robot_id=ctx.robot_id,
            text=args.get("text", ""),
            tags=args.get("tags", []),
            top_k=int(args.get("top_k", 5)),
        )
        hits = await self._memory.query(q)
        latency = (time.monotonic() - t0) * 1000
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"hits": [h.model_dump() for h in hits], "count": len(hits)},
            latency_ms=latency,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass
