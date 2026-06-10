"""Tests for AgentLoop using mock Brain and mock Tool."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from robot_harness.brain.base import BrainDecision, Task
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

    async def decide(self, messages: list[Any], tools: list[Any], **_: Any) -> BrainDecision:
        if self._call_count < len(self._decisions):
            d = self._decisions[self._call_count]
        else:
            d = BrainDecision(decision_type="give_up", message="exhausted")
        self._call_count += 1
        return d


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
    """Mock Brain that snapshots the conversation it receives on each decide()."""

    def __init__(self, decisions: list[BrainDecision]) -> None:
        self._decisions = list(decisions)
        self._i = 0
        self.seen_messages: list[list[dict[str, Any]]] = []

    @property
    def supports_streaming(self) -> bool:
        return False

    async def decide(
        self, messages: list[dict[str, Any]], tools: list[Any], **_: Any
    ) -> BrainDecision:
        self.seen_messages.append(list(messages))  # snapshot the conversation so far
        d = (
            self._decisions[self._i]
            if self._i < len(self._decisions)
            else BrainDecision(decision_type="plan", message="done")
        )
        self._i += 1
        return d


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
async def test_observations_flow_back_as_tool_messages_next_turn() -> None:
    """The observation→Brain edge via native tool messages (ADR-025).

    A tool result becomes a ``role:tool`` message in the conversation the loop
    owns, so the next ``decide()`` sees it. Without this edge the Brain
    re-observes forever (the close-loop spin). What a turn observes, the next
    turn sees in its messages.
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
    # turn 1: only system + user, no tool result yet
    assert all(m["role"] != "tool" for m in brain.seen_messages[0])
    # turn 2: the cube detected on turn 1 is present as a role:tool message
    tool_msgs = [m for m in brain.seen_messages[1] if m["role"] == "tool"]
    assert tool_msgs and any("red cube" in m["content"] for m in tool_msgs)


@pytest.mark.asyncio
async def test_memory_query_registered_for_on_demand_recall() -> None:
    """Long-term recall is an on-demand brain-visible tool, not an auto-query (ADR-024)."""
    from robot_harness.tools.base import BrainProfile
    from robot_harness.tools.memory.spatial_hub_adapter import SpatialHubMemory

    ctx = HarnessContext.build(memory=SpatialHubMemory())
    assert "memory.query" in ctx.tool_registry
    specs = ctx.tool_registry.export_for_brain(BrainProfile(name="openai"))
    names = {(s.get("function") or s).get("name") for s in specs}
    assert "memory.query" in names


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
async def test_call_ids_synced_into_assistant_message() -> None:
    """Backfilled call ids must land in BOTH the parsed tool_calls and the raw
    assistant message (ISS: backends like local vLLM/Ollama may omit tool_call
    ids; a mismatch between assistant.tool_calls[].id and the following
    role:tool tool_call_id breaks the next decide())."""
    from robot_harness.brain.base import ToolCallRequest

    # Brain emits a tool call with NO id, and an assistant_message mirroring that.
    no_id_decision = BrainDecision(
        decision_type="tool_call",
        tool_calls=[ToolCallRequest(tool_name="mock_tool", args={})],
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "mock_tool", "arguments": "{}"},
                }
            ],
        },
    )
    brain = _CapturingBrain([no_id_decision, BrainDecision(decision_type="plan", message="done")])
    ctx = _make_ctx()
    ctx.tool_registry.register(_MockTool())

    result = await AgentLoop(brain, ctx, max_turns=5).run(_make_task())
    assert result.outcome == "success"

    # In the turn-2 conversation: the assistant turn and its tool result must
    # reference the same, non-empty id.
    convo = brain.seen_messages[1]
    assistant = next(m for m in convo if m["role"] == "assistant" and m.get("tool_calls"))
    tool_msg = next(m for m in convo if m["role"] == "tool")
    assert assistant["tool_calls"][0]["id"]
    assert assistant["tool_calls"][0]["id"] == tool_msg["tool_call_id"]


@pytest.mark.asyncio
async def test_mismatched_assistant_tool_calls_raises() -> None:
    """An assistant_message whose tool_calls disagree in count with the parsed
    decision is a Brain protocol violation — surfaced, never silently truncated."""
    from robot_harness.brain.base import ToolCallRequest
    from robot_harness.errors import BrainOutputInvalidError

    bad_decision = BrainDecision(
        decision_type="tool_call",
        tool_calls=[ToolCallRequest(tool_name="mock_tool", args={})],
        assistant_message={"role": "assistant", "content": None, "tool_calls": []},
    )
    brain = _MockBrain([bad_decision])
    ctx = _make_ctx()
    ctx.tool_registry.register(_MockTool())

    with pytest.raises(BrainOutputInvalidError, match="protocol violation"):
        await AgentLoop(brain, ctx, max_turns=5).run(_make_task())


@pytest.mark.asyncio
async def test_brain_receives_trace_context() -> None:
    """The loop passes its task-level trace_id and the task's robot_id to
    decide() so the Brain span/exceptions stay on the same trace (ADR-008:
    robot_id runs through the whole chain)."""

    class _CtxCapturingBrain:
        def __init__(self) -> None:
            self.kwargs: dict[str, str] = {}

        @property
        def supports_streaming(self) -> bool:
            return False

        async def decide(self, messages: list[Any], tools: list[Any], **kw: str) -> BrainDecision:
            self.kwargs = dict(kw)
            return BrainDecision(decision_type="plan", message="done")

    brain = _CtxCapturingBrain()
    ctx = _make_ctx()
    result = await AgentLoop(brain, ctx, max_turns=2).run(_make_task(robot_id="r7"))

    assert result.outcome == "success"
    assert brain.kwargs.get("robot_id") == "r7"
    assert brain.kwargs.get("trace_id") == result.trace_id


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
