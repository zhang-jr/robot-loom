"""Memory-grounded navigation example.

Demonstrates the Memory 闭环:
1. Pre-populate EpisodicMemory with prior navigation episodes.
2. Brain queries memory before deciding — past failures inform retry strategy.
3. After task completion, the new episode is written back to memory.

Run:  python examples/memory_grounded_nav.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from robot_harness.brain.base import (
    BrainDecision,
    CriticSignal,
    ExecutionHistory,
    MemoryView,
    Task,
    ToolCallRequest,
)
from robot_harness.memory.base import MemoryEntry
from robot_harness.memory.episodic_memory import InMemoryEpisodicMemory
from robot_harness.memory.object_memory import InMemoryObjectMemory
from robot_harness.memory.place_memory import InMemoryPlaceMemory
from robot_harness.memory.semantic_memory import InMemorySemanticMemory
from robot_harness.runtime.agent_loop import AgentLoop
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema

# ---------------------------------------------------------------------------
# Composite memory that has all four sub-memories
# ---------------------------------------------------------------------------


class LocalMemory:
    def __init__(self) -> None:
        self.object = InMemoryObjectMemory()
        self.place = InMemoryPlaceMemory()
        self.episodic = InMemoryEpisodicMemory()
        self.semantic = InMemorySemanticMemory()

    async def write(self, entry: Any) -> str:
        if entry.memory_type == "episodic":
            return await self.episodic.write(entry)
        if entry.memory_type == "object":
            return await self.object.upsert(entry)
        if entry.memory_type == "place":
            return await self.place.upsert(entry)
        return await self.semantic.upsert(entry)

    async def query(self, q: Any) -> list:
        if q.memory_type == "episodic":
            return await self.episodic.query(q)
        if q.memory_type == "semantic":
            return await self.semantic.query(q)
        return []


# ---------------------------------------------------------------------------
# Mock navigation tool
# ---------------------------------------------------------------------------


class NavTool:
    name = "navigate_to"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="navigate_to",
        description="Navigate to a named place.",
        input_schema={
            "type": "object",
            "properties": {"place": {"type": "string"}},
            "required": ["place"],
        },
    )
    is_idempotent = True
    is_cancellable = False

    async def invoke(self, args: dict, ctx: ToolContext) -> ToolResult:
        place = args.get("place", "unknown")
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"arrived_at": place, "task_complete": True},
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Memory-aware mock Brain
# ---------------------------------------------------------------------------


class MemoryAwareBrain:
    """Brain that reads episodic memory hits and adjusts strategy."""

    def __init__(self) -> None:
        self._turn = 0

    async def decide(self, task: Task, memory_view: MemoryView, tools: list) -> BrainDecision:
        self._turn += 1

        # Inspect prior episode memory for relevant context
        prior_failures = [
            h for h in memory_view.episodic_hits if h.get("content", {}).get("outcome") == "failure"
        ]
        if prior_failures and self._turn == 1:
            print(
                f"  [Brain] Found {len(prior_failures)} prior failure(s) in memory — adjusting strategy"
            )

        return BrainDecision(
            decision_type="tool_call",
            tool_calls=[
                ToolCallRequest(
                    tool_name="navigate_to",
                    args={"place": "charging_station"},
                    call_id="nav_1",
                )
            ],
        )

    async def replan(self, history: ExecutionHistory, critic_signal: CriticSignal) -> BrainDecision:
        return BrainDecision(decision_type="give_up", message="no replan needed")

    @property
    def supports_streaming(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


async def main() -> None:
    memory = LocalMemory()

    # Pre-populate memory with a prior failed navigation episode
    await memory.write(
        MemoryEntry(
            memory_type="episodic",
            robot_id="mobile0",
            content={
                "task_id": "nav_prev_001",
                "description": "Navigate to charging_station",
                "outcome": "failure",
                "message": "Path blocked by obstacle",
            },
            tags=["episode", "failure", "mobile0"],
        )
    )
    print("Pre-populated memory with 1 prior failure episode.")

    ctx = HarnessContext.build(memory=memory)
    ctx.tool_registry.register(NavTool())

    brain = MemoryAwareBrain()
    loop = AgentLoop(brain=brain, ctx=ctx, max_turns=5)

    task = Task(
        task_id="nav_demo_001",
        description="Navigate to charging_station",
        robot_id="mobile0",
    )

    result = await loop.run(task)

    print("\n=== Result ===")
    print(f"Outcome : {result.outcome}")
    print(f"Turns   : {result.turns}")

    # Verify memory write-back
    from robot_harness.memory.base import MemoryQuery

    episodes = await memory.episodic.query(
        MemoryQuery(memory_type="episodic", robot_id="mobile0", text="charging_station")
    )
    print(f"\nEpisodicMemory now has {len(episodes)} records:")
    for ep in episodes:
        content = ep.content
        print(
            f"  - [{content.get('outcome', '?')}] {content.get('description', content.get('task_id', '?'))}"
        )


if __name__ == "__main__":
    asyncio.run(main())
