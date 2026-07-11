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
async def test_respond_before_any_tool_is_nudged_once() -> None:
    """A text-only respond before ANY tool has run gets one nudge turn; a
    second respond is then accepted as final."""
    decisions = [
        BrainDecision(decision_type="respond", message="My plan is: 1. look 2. grasp"),
        BrainDecision(decision_type="respond", message="All done!"),
    ]
    brain = _MockBrain(decisions)
    ctx = _make_ctx()
    loop = AgentLoop(brain, ctx, max_turns=5)
    result = await loop.run(_make_task())
    assert result.outcome == "success"
    assert result.turns == 2
    assert result.message == "All done!"


@pytest.mark.asyncio
async def test_nudge_appends_user_turn_and_fires_only_once() -> None:
    """The nudge is a user message appended to the same conversation (ADR-025),
    and it fires at most once per task."""
    brain = _CapturingBrain(
        [
            BrainDecision(decision_type="respond", message="here is my plan"),
            BrainDecision(decision_type="respond", message="done"),
        ]
    )
    loop = AgentLoop(brain, _make_ctx(), max_turns=5)
    result = await loop.run(_make_task())
    assert result.outcome == "success"
    # Second decide() saw: system, task, assistant(respond), nudge user turn.
    second = brain.seen_messages[1]
    assert second[-1]["role"] == "user"
    assert "without calling any tools" in second[-1]["content"]
    assert len(brain.seen_messages) == 2  # no second nudge


@pytest.mark.asyncio
async def test_nudged_brain_can_recover_with_tool_calls() -> None:
    """After the nudge the Brain may switch to tool calls; the eventual respond
    (tools now run) terminates without another nudge."""
    from robot_harness.brain.base import ToolCallRequest

    decisions = [
        BrainDecision(decision_type="respond", message="I will call mock_tool"),
        BrainDecision(
            decision_type="tool_call",
            tool_calls=[ToolCallRequest(tool_name="mock_tool", args={})],
        ),
        BrainDecision(decision_type="respond", message="executed"),
    ]
    mock_tool = _MockTool()
    ctx = _make_ctx()
    ctx.tool_registry.register(mock_tool)
    loop = AgentLoop(_MockBrain(decisions), ctx, max_turns=5)
    result = await loop.run(_make_task())
    assert result.outcome == "success"
    assert mock_tool.invoke_count == 1
    assert result.turns == 3


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
async def test_tool_call_then_respond() -> None:
    from robot_harness.brain.base import ToolCallRequest

    tool_call_decision = BrainDecision(
        decision_type="tool_call",
        tool_calls=[ToolCallRequest(tool_name="mock_tool", args={})],
    )
    respond_decision = BrainDecision(decision_type="respond", message="Done after tool call")

    mock_tool = _MockTool()
    brain = _MockBrain([tool_call_decision, respond_decision])
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
            else BrainDecision(decision_type="respond", message="done")
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
    brain = _CapturingBrain(
        [detect_call, BrainDecision(decision_type="respond", message="seen it")]
    )
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
    brain = _CapturingBrain(
        [no_id_decision, BrainDecision(decision_type="respond", message="done")]
    )
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
            return BrainDecision(decision_type="respond", message="done")

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
    plan = BrainDecision(decision_type="respond", message="ok")

    brain = _MockBrain([decision, plan])
    ctx = _make_ctx()
    loop = AgentLoop(brain, ctx, max_turns=5)
    result = await loop.run(_make_task())
    # Should not crash — error result is recorded and loop continues
    assert result.outcome == "success"
    assert any(r.get("success") is False for r in result.tool_results)


# ---------------------------------------------------------------------------
# Critic supervision is best-effort: frame-fetch failures degrade (ISS-031)
# ---------------------------------------------------------------------------


class _NeverCalledCritic:
    """Critic that fails the test if judge() is ever reached."""

    def __init__(self) -> None:
        self.calls = 0

    async def judge(self, current: Any, reference: Any, task_description: str) -> Any:
        self.calls += 1
        raise AssertionError("critic must not run when the frame fetch failed")


@pytest.mark.asyncio
async def test_critic_frame_fetch_failure_skips_supervision_not_task() -> None:
    """A camera fetch failure (robot offline / camera 404) must skip the critic
    round — turn runs unsupervised — never crash the task (ISS-031)."""
    from robot_harness.brain.base import ToolCallRequest
    from robot_harness.errors import RobotOfflineError

    decisions = [
        BrainDecision(
            decision_type="tool_call",
            tool_calls=[ToolCallRequest(tool_name="mock_tool", args={})],
        ),
        BrainDecision(decision_type="respond", message="done"),
    ]
    ctx = _make_ctx()
    ctx.tool_registry.register(_MockTool())

    async def _broken_camera(robot_id: str, camera: str = "") -> Any:
        raise RobotOfflineError("camera endpoint 404", robot_id=robot_id)

    ctx.get_camera_frame = _broken_camera  # type: ignore[method-assign]
    critic = _NeverCalledCritic()
    loop = AgentLoop(_MockBrain(decisions), ctx, max_turns=5, critic=critic, critic_interval=1)
    result = await loop.run(_make_task())
    assert result.outcome == "success"
    assert critic.calls == 0


@pytest.mark.asyncio
async def test_camera_frame_resolves_declared_camera_from_config() -> None:
    """The critic path must ask for a camera the fleet config actually declares,
    not a hard-coded name (ISS-031)."""
    from robot_harness.config.schema import EmbodimentBackendConfig, HarnessConfig
    from robot_harness.embodiment.base import Frame

    class _RecordingCameraAdapter:
        robot_id = "r0"
        robot_type = "arm"

        def __init__(self) -> None:
            self.requested: list[str] = []

        async def get_camera_frame(self, camera: str) -> Frame:
            self.requested.append(camera)
            return Frame(camera=camera, robot_id="r0")

    cfg = HarnessConfig(
        robot_ids=["r0"],
        embodiments={"r0": EmbodimentBackendConfig(cameras=["head_cam"])},
    )
    ctx = HarnessContext.build(config=cfg)
    adapter = _RecordingCameraAdapter()
    ctx.embodiment_adapters["r0"] = adapter  # type: ignore[assignment]

    await ctx.get_camera_frame("r0")
    assert adapter.requested == ["head_cam"]

    # An explicit camera argument still wins over the declared default.
    await ctx.get_camera_frame("r0", camera="wrist_camera")
    assert adapter.requested == ["head_cam", "wrist_camera"]


# ---------------------------------------------------------------------------
# Failed tool results keep their structured output in the conversation (ISS-032)
# ---------------------------------------------------------------------------


def test_tool_message_failure_carries_stripped_output() -> None:
    """A failed/partial result's evidence must reach the Brain; only blobs are
    stripped — never the structured facts (ISS-032)."""
    import json

    loop = AgentLoop(_MockBrain([]), _make_ctx())
    msg = loop._tool_message(
        "call_1",
        {
            "success": False,
            "error": "robot_sdk.reactive_grasp outcome=partial (aborted_by=timeout): slipped",
            "error_type": "VerbFailed",
            "output": {
                "outcome": "partial",
                "evidence": "slipped",
                "aborted_by": "timeout",
                "image_b64": "SHOULD-NOT-LEAK",
            },
        },
    )
    payload = json.loads(msg["content"])
    assert payload["error_type"] == "VerbFailed"
    assert payload["output"]["evidence"] == "slipped"
    assert "image_b64" not in payload["output"]
    assert "SHOULD-NOT-LEAK" not in msg["content"]


# ---------------------------------------------------------------------------
# Untyped exceptions are normalized, never crash the task (ISS-034)
# ---------------------------------------------------------------------------


class _BuggyTool:
    name = "buggy_tool"
    backend: ToolBackend = "native"
    is_idempotent = True
    is_cancellable = False
    schema = ToolSchema(
        name="buggy_tool",
        description="Raises an untyped exception",
        input_schema={"type": "object", "properties": {}},
    )

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise KeyError("tool bug")

    async def cancel(self, ctx: ToolContext) -> None:
        pass


@pytest.mark.asyncio
async def test_untyped_tool_exception_becomes_failed_result() -> None:
    """A tool bug (or unparsed external payload) must become a failed result
    the Brain replans on — not ride the safety-violation path and crash the
    task without an episode (ISS-034)."""
    from robot_harness.brain.base import ToolCallRequest

    decisions = [
        BrainDecision(
            decision_type="tool_call",
            tool_calls=[ToolCallRequest(tool_name="buggy_tool", args={})],
        ),
        BrainDecision(decision_type="respond", message="recovered"),
    ]
    ctx = _make_ctx()
    ctx.tool_registry.register(_BuggyTool())
    loop = AgentLoop(_MockBrain(decisions), ctx, max_turns=5)
    result = await loop.run(_make_task())
    assert result.outcome == "success"
    assert result.tool_results[0]["success"] is False
    assert result.tool_results[0]["error_type"] == "KeyError"


@pytest.mark.asyncio
async def test_untyped_skill_exception_becomes_failed_result() -> None:
    """Same for a bug in a (user-provided) skill reached via skill.<name>."""
    from robot_harness.brain.base import ToolCallRequest
    from robot_harness.skill.base import SkillManifest

    class _BoomSkill:
        manifest = SkillManifest(name="boom", version="0.0.1")

        async def execute(self, subtask: Any, tools: Any, ctx: Any) -> Any:
            raise RuntimeError("skill bug")

        async def rollback(self, ctx: Any) -> None:
            pass

    decisions = [
        BrainDecision(
            decision_type="tool_call",
            tool_calls=[ToolCallRequest(tool_name="skill.boom", args={"robot_id": "r0"})],
        ),
        BrainDecision(decision_type="respond", message="recovered"),
    ]
    ctx = _make_ctx()
    ctx.skill_registry.register(_BoomSkill())
    loop = AgentLoop(_MockBrain(decisions), ctx, max_turns=5)
    result = await loop.run(_make_task())
    assert result.outcome == "success"
    assert result.tool_results[0]["success"] is False
    assert result.tool_results[0]["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_missing_required_args_fail_before_dispatch() -> None:
    """Brain args violating the tool's input schema are refused at the loop
    gate as a typed failed result — not a KeyError deep inside the tool."""
    from robot_harness.brain.base import ToolCallRequest

    class _StrictTool(_MockTool):
        name = "strict_tool"
        schema = ToolSchema(
            name="strict_tool",
            description="Requires foo",
            input_schema={
                "type": "object",
                "properties": {"foo": {"type": "string"}},
                "required": ["foo"],
            },
        )

    strict = _StrictTool()
    decisions = [
        BrainDecision(
            decision_type="tool_call",
            tool_calls=[ToolCallRequest(tool_name="strict_tool", args={})],
        ),
        BrainDecision(decision_type="respond", message="done"),
    ]
    ctx = _make_ctx()
    ctx.tool_registry.register(strict)
    loop = AgentLoop(_MockBrain(decisions), ctx, max_turns=5)
    result = await loop.run(_make_task())
    assert result.tool_results[0]["success"] is False
    assert result.tool_results[0]["error_type"] == "ToolSchemaViolationError"
    assert strict.invoke_count == 0


# ---------------------------------------------------------------------------
# Loop backstop deadline for hung tools (ISS-037)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hung_tool_hits_loop_backstop_deadline() -> None:
    """ctx.timeout_s is advisory; the loop's wait_for backstop must convert a
    hung tool into a failed ToolTimeoutError result instead of blocking the
    turn forever (ISS-037)."""
    import asyncio

    from robot_harness.brain.base import ToolCallRequest
    from robot_harness.config.schema import HarnessConfig, ToolConfig

    class _HangingTool(_MockTool):
        name = "hanging_tool"
        schema = ToolSchema(
            name="hanging_tool",
            description="Never returns",
            input_schema={"type": "object", "properties": {}},
        )

        async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            await asyncio.sleep(60)
            raise AssertionError("unreachable")

    decisions = [
        BrainDecision(
            decision_type="tool_call",
            tool_calls=[ToolCallRequest(tool_name="hanging_tool", args={})],
        ),
        BrainDecision(decision_type="respond", message="recovered"),
    ]
    cfg = HarnessConfig(tool=ToolConfig(default_timeout_s=0.3))
    ctx = HarnessContext.build(config=cfg)
    ctx.tool_registry.register(_HangingTool())
    loop = AgentLoop(_MockBrain(decisions), ctx, max_turns=5)
    result = await loop.run(_make_task())
    assert result.outcome == "success"
    assert result.tool_results[0]["success"] is False
    assert result.tool_results[0]["error_type"] == "ToolTimeoutError"


def test_tool_deadline_honors_verb_budget() -> None:
    """A verb's constraints.timeout_s may legitimately exceed the default —
    the backstop must not undercut the transport's own read deadline."""
    from robot_harness.config.schema import HarnessConfig, ToolConfig

    cfg = HarnessConfig(tool=ToolConfig(default_timeout_s=30.0))
    ctx = HarnessContext.build(config=cfg)
    loop = AgentLoop(_MockBrain([]), ctx)
    assert loop._tool_deadline_s({}) == 30.0
    assert loop._tool_deadline_s({"constraints": {"timeout_s": 90.0}}) == 100.0
    assert loop._tool_deadline_s({"constraints": {"timeout_s": "bogus"}}) == 30.0
