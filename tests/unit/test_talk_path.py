"""Tests for the Talk path (ADR-023): report delivery + ask_user round-trip.

Covers both halves of Talk:
- ``generic.send_message`` delivering through the session-bound outbound handle
  (and degrading to a tracer-only no-op when no channel is wired);
- ``ask_user`` suspending the AgentLoop, awaiting a reply, and resuming the same
  conversation — plus the unanswered (timeout / no-channel) fallbacks;
- the ChannelManager routing a reply to a pending question instead of starting a
  new Task.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest

from robot_harness.brain.base import BrainDecision, Task
from robot_harness.channels.base import ChannelMessage, ChannelResponse
from robot_harness.channels.manager import ChannelManager
from robot_harness.channels.outbound import ChannelOrigin
from robot_harness.errors import ChannelUnavailable
from robot_harness.runtime.agent_loop import AgentLoop, AgentResult
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.tools.base import ToolContext
from robot_harness.tools.generic.message import SendMessageTool

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeOutbound:
    """Records reports and serves canned replies (or raises) for asks."""

    def __init__(
        self, replies: list[str] | None = None, raise_exc: Exception | None = None
    ) -> None:
        self.reports: list[tuple[str, str]] = []
        self.asks: list[str] = []
        self._replies = list(replies or [])
        self._raise = raise_exc

    async def report(self, text: str, level: str = "info") -> None:
        self.reports.append((text, level))

    async def ask(self, prompt: str, *, timeout_s: float) -> str:
        self.asks.append(prompt)
        if self._raise is not None:
            raise self._raise
        return self._replies.pop(0)


class _MockBrain:
    def __init__(self, decisions: list[BrainDecision]) -> None:
        self._decisions = list(decisions)
        self._i = 0

    @property
    def supports_streaming(self) -> bool:
        return False

    async def decide(self, messages: list[Any], tools: list[Any], **_: Any) -> BrainDecision:
        d = (
            self._decisions[self._i]
            if self._i < len(self._decisions)
            else BrainDecision(decision_type="give_up", message="exhausted")
        )
        self._i += 1
        return d


class _FakeChannel:
    channel_name = "fake"

    def __init__(self) -> None:
        self.sent: list[ChannelResponse] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def listen(self) -> AsyncIterator[ChannelMessage]:
        return
        yield  # make this an async generator

    async def send(self, response: ChannelResponse) -> None:
        self.sent.append(response)


class _FakeFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[Task, Any]] = []

    async def __call__(self, task: Task, *, outbound: Any) -> AgentResult:
        self.calls.append((task, outbound))
        return AgentResult(
            task_id=task.task_id, robot_id=task.robot_id, outcome="success", message="ok"
        )


def _task(robot_id: str = "r0") -> Task:
    return Task(task_id=str(uuid.uuid4()), description="t", robot_id=robot_id)


# ---------------------------------------------------------------------------
# send_message (report half)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_message_delivers_via_outbound() -> None:
    out = _FakeOutbound()
    ctx = ToolContext(trace_id="t", robot_id="r0", outbound=out)
    result = await SendMessageTool().invoke({"text": "arrived", "level": "warning"}, ctx)
    assert result.success
    assert result.output["delivered"] is True
    assert out.reports == [("arrived", "warning")]


@pytest.mark.asyncio
async def test_send_message_without_channel_is_tracer_only() -> None:
    ctx = ToolContext(trace_id="t", robot_id="r0")  # outbound=None
    result = await SendMessageTool().invoke({"text": "hi"}, ctx)
    assert result.success
    assert result.output["delivered"] is False


# ---------------------------------------------------------------------------
# ask_user (round-trip half)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_user_resumes_after_reply() -> None:
    brain = _MockBrain(
        [
            BrainDecision(decision_type="ask_user", message="which cup?"),
            BrainDecision(decision_type="plan", message="done"),
        ]
    )
    out = _FakeOutbound(replies=["the red one"])
    loop = AgentLoop(brain, HarnessContext.build(), max_turns=5)
    result = await loop.run(_task(), outbound=out)
    assert result.outcome == "success"
    assert out.asks == ["which cup?"]


@pytest.mark.asyncio
async def test_ask_user_timeout_is_incomplete() -> None:
    brain = _MockBrain([BrainDecision(decision_type="ask_user", message="which cup?")])
    out = _FakeOutbound(raise_exc=TimeoutError())
    loop = AgentLoop(brain, HarnessContext.build(), max_turns=5)
    result = await loop.run(_task(), outbound=out)
    assert result.outcome == "incomplete"
    assert "which cup?" in result.message


@pytest.mark.asyncio
async def test_ask_user_without_channel_is_incomplete() -> None:
    brain = _MockBrain([BrainDecision(decision_type="ask_user", message="which cup?")])
    loop = AgentLoop(brain, HarnessContext.build(), max_turns=5)
    result = await loop.run(_task())  # no outbound → NullOutbound.ask raises
    assert result.outcome == "incomplete"


@pytest.mark.asyncio
async def test_ask_user_exceeds_limit_gives_up() -> None:
    brain = _MockBrain([BrainDecision(decision_type="ask_user", message="q?")] * 5)
    out = _FakeOutbound(replies=["a", "a", "a", "a", "a"])
    loop = AgentLoop(brain, HarnessContext.build(), max_turns=10, max_ask_user=2)
    result = await loop.run(_task(), outbound=out)
    assert result.outcome == "give_up"


# ---------------------------------------------------------------------------
# ChannelManager routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reply_routed_to_pending_question_not_new_task() -> None:
    factory = _FakeFactory()
    manager = ChannelManager(factory)
    channel = _FakeChannel()
    manager.register(channel)
    origin = ChannelOrigin(channel="fake", user_id="u1")

    ask = asyncio.create_task(manager.ask(origin, "which?", timeout_s=5))
    await asyncio.sleep(0)  # let ask register the pending future + send the prompt
    await manager._handle(channel, ChannelMessage(channel="fake", user_id="u1", text="red"))
    reply = await ask

    assert reply == "red"
    assert factory.calls == []  # the reply did NOT spawn a new task
    assert channel.sent[0].text == "which?"


@pytest.mark.asyncio
async def test_non_reply_message_spawns_task() -> None:
    factory = _FakeFactory()
    manager = ChannelManager(factory)
    channel = _FakeChannel()
    manager.register(channel)

    await manager._handle(channel, ChannelMessage(channel="fake", user_id="u1", text="do it"))

    assert len(factory.calls) == 1
    assert factory.calls[0][0].description == "do it"


@pytest.mark.asyncio
async def test_send_to_unregistered_channel_raises() -> None:
    manager = ChannelManager(_FakeFactory())
    with pytest.raises(ChannelUnavailable):
        await manager.send(ChannelOrigin(channel="nope", user_id="u1"), "hi")
