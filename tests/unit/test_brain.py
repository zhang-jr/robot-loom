"""Unit tests for Brain Protocol and LiteLLMBrain (mock LLM backend)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robot_harness.brain.base import (
    BrainDecision,
    CriticSignal,
    ExecutionHistory,
    MemoryView,
    Task,
    ToolCallRequest,
)
from robot_harness.brain.litellm_brain import LiteLLMBrain
from robot_harness.config.schema import BrainConfig
from robot_harness.errors import BrainOutputInvalidError

# ---------------------------------------------------------------------------
# Helpers — build litellm-style mock response objects
# ---------------------------------------------------------------------------


def _tool_call_response(tool_names: list[str]) -> MagicMock:
    """Mimics a litellm ModelResponse with tool_calls."""
    response = MagicMock()
    tcs = []
    for i, name in enumerate(tool_names):
        tc = MagicMock()
        tc.id = f"call_{i}"
        tc.function.name = name
        tc.function.arguments = '{"query": "red cup"}'
        tcs.append(tc)
    response.choices[0].message.tool_calls = tcs
    response.choices[0].message.content = None
    response.choices[0].finish_reason = "tool_calls"
    return response


def _text_response(text: str) -> MagicMock:
    """Mimics a litellm ModelResponse with plain text content."""
    response = MagicMock()
    response.choices[0].message.tool_calls = None
    response.choices[0].message.content = text
    response.choices[0].finish_reason = "stop"
    return response


def _make_task(description: str = "pick the red cup") -> Task:
    import uuid

    return Task(
        task_id=str(uuid.uuid4()),
        description=description,
        robot_id="robot-0",
    )


# ---------------------------------------------------------------------------
# Tests: data models (no LLM call needed)
# ---------------------------------------------------------------------------


def test_brain_decision_tool_call_model() -> None:
    d = BrainDecision(
        decision_type="tool_call",
        tool_calls=[ToolCallRequest(tool_name="perception.detect_objects", args={"query": "cup"})],
    )
    assert d.decision_type == "tool_call"
    assert d.tool_calls[0].tool_name == "perception.detect_objects"


def test_brain_decision_give_up_model() -> None:
    d = BrainDecision(decision_type="give_up", message="Cannot locate target")
    assert d.decision_type == "give_up"
    assert "Cannot" in d.message


def test_critic_signal_model() -> None:
    cs = CriticSignal(state="failure", confidence=0.8, evidence="object not moved")
    assert cs.state == "failure"
    assert cs.confidence == pytest.approx(0.8)


def test_task_model() -> None:
    t = _make_task()
    assert t.robot_id == "robot-0"
    assert t.description == "pick the red cup"


def test_memory_view_defaults() -> None:
    mv = MemoryView()
    assert mv.episodic_hits == []
    assert mv.object_hits == []


def test_execution_history_defaults() -> None:
    h = ExecutionHistory()
    assert h.turns == []
    assert h.last_critic_signal == ""


# ---------------------------------------------------------------------------
# Tests: LiteLLMBrain — tool_call path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brain_decide_returns_tool_call() -> None:
    brain = LiteLLMBrain(BrainConfig(model="openai/gpt-4o"))
    fake = _tool_call_response(["perception.detect_objects"])

    with patch("litellm.acompletion", new=AsyncMock(return_value=fake)):
        decision = await brain.decide(_make_task(), MemoryView(), [])

    assert decision.decision_type == "tool_call"
    assert len(decision.tool_calls) == 1
    assert decision.tool_calls[0].tool_name == "perception.detect_objects"


@pytest.mark.asyncio
async def test_brain_decide_multiple_tool_calls() -> None:
    brain = LiteLLMBrain(BrainConfig(model="openai/gpt-4o"))
    fake = _tool_call_response(["perception.detect_objects", "perception.estimate_depth"])

    with patch("litellm.acompletion", new=AsyncMock(return_value=fake)):
        decision = await brain.decide(_make_task(), MemoryView(), [])

    assert decision.decision_type == "tool_call"
    assert len(decision.tool_calls) == 2


# ---------------------------------------------------------------------------
# Tests: LiteLLMBrain — plan path (text response, no tool calls)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brain_decide_returns_plan_on_text_response() -> None:
    brain = LiteLLMBrain(BrainConfig(model="openai/gpt-4o"))
    fake = _text_response("Step 1: perceive. Step 2: grasp.")

    with patch("litellm.acompletion", new=AsyncMock(return_value=fake)):
        decision = await brain.decide(_make_task(), MemoryView(), [])

    assert decision.decision_type == "plan"
    assert "Step 1" in decision.plan


@pytest.mark.asyncio
async def test_brain_decide_give_up_on_negative_text() -> None:
    brain = LiteLLMBrain(BrainConfig(model="openai/gpt-4o"))
    fake = _text_response("I cannot complete this task — the object is not found.")

    with patch("litellm.acompletion", new=AsyncMock(return_value=fake)):
        decision = await brain.decide(_make_task(), MemoryView(), [])

    assert decision.decision_type == "give_up"


# ---------------------------------------------------------------------------
# Tests: LiteLLMBrain — replan path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brain_replan_with_critic_signal() -> None:
    brain = LiteLLMBrain(BrainConfig(model="openai/gpt-4o"))
    fake = _tool_call_response(["perception.detect_objects"])
    critic = CriticSignal(state="failure", confidence=0.7, evidence="object fell")
    history = ExecutionHistory(turns=[{"turn": 1, "result": "failed"}])

    with patch("litellm.acompletion", new=AsyncMock(return_value=fake)):
        decision = await brain.replan(history, critic)

    assert decision.decision_type in ("tool_call", "plan", "give_up")


# ---------------------------------------------------------------------------
# Tests: LiteLLMBrain — exception / error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brain_decide_raises_on_backend_exception() -> None:
    """LiteLLM errors are re-raised as BrainOutputInvalidError (not swallowed)."""
    brain = LiteLLMBrain(BrainConfig(model="openai/gpt-4o"))

    with patch("litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("network error"))):
        with pytest.raises(BrainOutputInvalidError, match="network error"):
            await brain.decide(_make_task(), MemoryView(), [])


@pytest.mark.asyncio
async def test_brain_replan_raises_on_backend_exception() -> None:
    brain = LiteLLMBrain(BrainConfig(model="openai/gpt-4o"))
    critic = CriticSignal(state="failure", confidence=0.5, evidence="stalled")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("timeout"))):
        with pytest.raises(BrainOutputInvalidError):
            await brain.replan(ExecutionHistory(), critic)


# ---------------------------------------------------------------------------
# Tests: protocol / property
# ---------------------------------------------------------------------------


def test_brain_supports_streaming_property() -> None:
    brain = LiteLLMBrain(BrainConfig())
    assert isinstance(brain.supports_streaming, bool)
