"""Tests for AgentLoop using mock Brain and mock Tool."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from robot_harness.brain.base import BrainDecision, CriticSignal, ExecutionHistory, MemoryView, Task
from robot_harness.runtime.agent_loop import AgentLoop
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema

# ---------------------------------------------------------------------------
# Mock Brain
# ---------------------------------------------------------------------------


class _MockBrain:
    """Configurable mock Brain for testing."""

    def __init__(self, decisions: list[BrainDecision]) -> None:
        self._decisions = list(decisions)
        self._call_count = 0

    @property
    def supports_streaming(self) -> bool:
        return False

    async def decide(self, task: Task, memory_view: MemoryView, tools: list[Any]) -> BrainDecision:
        if self._call_count < len(self._decisions):
            d = self._decisions[self._call_count]
        else:
            d = BrainDecision(decision_type="give_up", message="exhausted")
        self._call_count += 1
        return d

    async def replan(self, history: ExecutionHistory, critic_signal: CriticSignal) -> BrainDecision:
        return BrainDecision(decision_type="give_up", message="replan not supported in mock")


# ---------------------------------------------------------------------------
# Mock Tool
# ---------------------------------------------------------------------------


class _MockTool:
    name = "mock_tool"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="mock_tool",
        description="Mock tool",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self, output: dict[str, Any] | None = None) -> None:
        self._output = output or {"status": "done"}
        self.invoke_count = 0

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return True

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.invoke_count += 1
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output=self._output,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx() -> HarnessContext:
    return HarnessContext.build()


def _make_task(robot_id: str = "r0") -> Task:
    return Task(task_id=str(uuid.uuid4()), description="test task", robot_id=robot_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brain_returns_plan_immediately() -> None:
    decision = BrainDecision(decision_type="plan", message="All done!")
    brain = _MockBrain([decision])
    ctx = _make_ctx()
    loop = AgentLoop(brain, ctx, max_turns=5)
    result = await loop.run(_make_task())
    assert result.outcome == "success"
    assert result.turns == 1


@pytest.mark.asyncio
async def test_brain_gives_up() -> None:
    decision = BrainDecision(decision_type="give_up", message="Cannot do this")
    brain = _MockBrain([decision])
    ctx = _make_ctx()
    loop = AgentLoop(brain, ctx, max_turns=5)
    result = await loop.run(_make_task())
    assert result.outcome == "give_up"
    assert "Cannot" in result.message


@pytest.mark.asyncio
async def test_tool_call_then_plan() -> None:
    from robot_harness.brain.base import ToolCallRequest

    tool_call_decision = BrainDecision(
        decision_type="tool_call",
        tool_calls=[ToolCallRequest(tool_name="mock_tool", args={})],
    )
    plan_decision = BrainDecision(decision_type="plan", message="Done after tool call")

    mock_tool = _MockTool()
    brain = _MockBrain([tool_call_decision, plan_decision])
    ctx = _make_ctx()
    ctx.tool_registry.register(mock_tool)

    loop = AgentLoop(brain, ctx, max_turns=5)
    result = await loop.run(_make_task())
    assert result.outcome == "success"
    assert mock_tool.invoke_count == 1
    assert result.turns == 2


class _CapturingBrain:
    """Mock Brain that records the MemoryView it receives on each decide()."""

    def __init__(self, decisions: list[BrainDecision]) -> None:
        self._decisions = list(decisions)
        self._i = 0
        self.seen_views: list[MemoryView] = []

    @property
    def supports_streaming(self) -> bool:
        return False

    async def decide(self, task: Task, memory_view: MemoryView, tools: list[Any]) -> BrainDecision:
        self.seen_views.append(memory_view)
        d = (
            self._decisions[self._i]
            if self._i < len(self._decisions)
            else BrainDecision(decision_type="plan", message="done")
        )
        self._i += 1
        return d

    async def replan(self, history: ExecutionHistory, critic_signal: CriticSignal) -> BrainDecision:
        return BrainDecision(decision_type="give_up", message="n/a")


class _DetectTool:
    name = "perception.detect"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="perception.detect", description="detect", input_schema={"type": "object"}
    )

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"objects": [{"id": "cube", "label": "red cube", "confidence": 0.9}]},
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass


@pytest.mark.asyncio
async def test_observations_written_back_are_recalled_next_turn() -> None:
    """The observation→Brain edge: what a turn observes, the next turn recalls.

    Without this, the Brain re-observes forever (the close-loop spin). A
    detection written to object memory after turn 1 must appear in the
    MemoryView the Brain receives on turn 2.
    """
    from robot_harness.brain.base import ToolCallRequest
    from robot_harness.tools.memory.spatial_hub_adapter import SpatialHubMemory

    detect_call = BrainDecision(
        decision_type="tool_call",
        tool_calls=[ToolCallRequest(tool_name="perception.detect", args={})],
    )
    brain = _CapturingBrain([detect_call, BrainDecision(decision_type="plan", message="seen it")])
    ctx = HarnessContext.build(memory=SpatialHubMemory())
    ctx.tool_registry.register(_DetectTool())

    result = await AgentLoop(brain, ctx, max_turns=5).run(_make_task())

    assert result.outcome == "success"
    # turn 1: nothing observed yet
    assert brain.seen_views[0].object_hits == []
    # turn 2: the cube detected on turn 1 is recalled
    recalled = brain.seen_views[1].object_hits
    assert any(h["content"].get("label") == "red cube" for h in recalled)


@pytest.mark.asyncio
async def test_max_turns_exceeded_returns_incomplete() -> None:
    """Exhausting the turn budget is an expected terminal outcome, not an error.

    An agent that keeps acting without converging must not crash the harness:
    the loop returns an ``incomplete`` result (with the work so far) instead of
    raising. ``ReplanLoopExceededError`` is reserved for real replan
    non-convergence.
    """
    from robot_harness.brain.base import ToolCallRequest

    # Brain keeps returning tool calls forever
    decision = BrainDecision(
        decision_type="tool_call",
        tool_calls=[ToolCallRequest(tool_name="mock_tool", args={})],
    )

    mock_tool = _MockTool()
    brain = _MockBrain([decision] * 100)
    ctx = _make_ctx()
    ctx.tool_registry.register(mock_tool)

    loop = AgentLoop(brain, ctx, max_turns=3)
    result = await loop.run(_make_task())

    assert result.outcome == "incomplete"
    assert result.turns == 3
    # work-so-far is preserved (one tool call per turn), not discarded
    assert len(result.tool_results) == 3
    assert mock_tool.invoke_count == 3


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_result() -> None:
    from robot_harness.brain.base import ToolCallRequest

    decision = BrainDecision(
        decision_type="tool_call",
        tool_calls=[ToolCallRequest(tool_name="nonexistent_tool", args={})],
    )
    plan = BrainDecision(decision_type="plan", message="ok")

    brain = _MockBrain([decision, plan])
    ctx = _make_ctx()
    loop = AgentLoop(brain, ctx, max_turns=5)
    result = await loop.run(_make_task())
    # Should not crash — error result is recorded and loop continues
    assert result.outcome == "success"
    assert any(r.get("success") is False for r in result.tool_results)
