"""Unit tests for Brain Protocol and LiteLLMBrain (mock LLM backend)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robot_harness.brain.base import BrainDecision, Task, ToolCallRequest
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


def _messages() -> list[dict[str, Any]]:
    """A minimal native conversation — the AgentLoop owns this list (ADR-025)."""
    return [
        {"role": "system", "content": "You are a robot task planner."},
        {"role": "user", "content": "Task: pick the red cup"},
    ]


def _make_task(description: str = "pick the red cup") -> Task:
    import uuid

    return Task(task_id=str(uuid.uuid4()), description=description, robot_id="robot-0")


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


def test_task_model() -> None:
    t = _make_task()
    assert t.robot_id == "robot-0"
    assert t.description == "pick the red cup"


# ---------------------------------------------------------------------------
# Tests: LiteLLMBrain — tool_call path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brain_decide_returns_tool_call() -> None:
    brain = LiteLLMBrain(BrainConfig(model="openai/gpt-4o"))
    fake = _tool_call_response(["perception.detect_objects"])

    with patch("litellm.acompletion", new=AsyncMock(return_value=fake)):
        decision = await brain.decide(_messages(), [])

    assert decision.decision_type == "tool_call"
    assert len(decision.tool_calls) == 1
    assert decision.tool_calls[0].tool_name == "perception.detect_objects"


@pytest.mark.asyncio
async def test_brain_decide_sets_assistant_message_with_tool_calls() -> None:
    """The assistant turn is returned verbatim so the loop can append it to history
    before the tool results (native protocol, ADR-025)."""
    brain = LiteLLMBrain(BrainConfig(model="openai/gpt-4o"))
    fake = _tool_call_response(["perception.detect_objects"])

    with patch("litellm.acompletion", new=AsyncMock(return_value=fake)):
        decision = await brain.decide(_messages(), [])

    msg = decision.assistant_message
    assert msg is not None and msg["role"] == "assistant"
    assert msg["tool_calls"][0]["id"] == "call_0"
    assert msg["tool_calls"][0]["function"]["name"] == "perception.detect_objects"


@pytest.mark.asyncio
async def test_brain_decide_multiple_tool_calls() -> None:
    brain = LiteLLMBrain(BrainConfig(model="openai/gpt-4o"))
    fake = _tool_call_response(["perception.detect_objects", "perception.estimate_depth"])

    with patch("litellm.acompletion", new=AsyncMock(return_value=fake)):
        decision = await brain.decide(_messages(), [])

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
        decision = await brain.decide(_messages(), [])

    assert decision.decision_type == "plan"
    assert "Step 1" in decision.plan


@pytest.mark.asyncio
async def test_brain_decide_give_up_on_negative_text() -> None:
    brain = LiteLLMBrain(BrainConfig(model="openai/gpt-4o"))
    fake = _text_response("I cannot complete this task — the object is not found.")

    with patch("litellm.acompletion", new=AsyncMock(return_value=fake)):
        decision = await brain.decide(_messages(), [])

    assert decision.decision_type == "give_up"


# ---------------------------------------------------------------------------
# Tests: LiteLLMBrain — exception / error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brain_decide_raises_on_backend_exception() -> None:
    """LiteLLM errors are re-raised as BrainOutputInvalidError (not swallowed)."""
    brain = LiteLLMBrain(BrainConfig(model="openai/gpt-4o"))

    with patch("litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("network error"))):
        with pytest.raises(BrainOutputInvalidError, match="network error"):
            await brain.decide(_messages(), [])


@pytest.mark.asyncio
async def test_brain_errors_carry_trace_context() -> None:
    """Caller-provided trace_id/robot_id tag Brain exceptions, keeping the Brain
    layer on the same trace as the task's tool/critic/memory spans (ADR-008)."""
    brain = LiteLLMBrain(BrainConfig(model="openai/gpt-4o"))

    with patch("litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(BrainOutputInvalidError) as excinfo:
            await brain.decide(_messages(), [], trace_id="trace-42", robot_id="r0")

    assert excinfo.value.trace_id == "trace-42"
    assert excinfo.value.robot_id == "r0"


@pytest.mark.asyncio
async def test_brain_decision_uses_caller_trace_id() -> None:
    """decide() must not mint its own trace_id when the loop provides one."""
    brain = LiteLLMBrain(BrainConfig(model="openai/gpt-4o"))
    fake = _text_response("Step 1: perceive.")

    with patch("litellm.acompletion", new=AsyncMock(return_value=fake)):
        decision = await brain.decide(_messages(), [], trace_id="trace-42", robot_id="r0")

    assert decision.trace_id == "trace-42"


# ---------------------------------------------------------------------------
# Tests: protocol / property
# ---------------------------------------------------------------------------


def test_brain_supports_streaming_property() -> None:
    brain = LiteLLMBrain(BrainConfig())
    assert isinstance(brain.supports_streaming, bool)
