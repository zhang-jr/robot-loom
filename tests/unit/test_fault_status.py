"""FAULT latch surfacing (physical-executor contract v0.2).

A per-robot agent_server latches FAULT when a verb ends on safety/hardware, when
/abort is called, or when it restarts with no clean shutdown behind it. While
latched it advertises zero verbs, so the existing capability gate already stops
the fleet planning with that body — what these tests cover is the half the gate
cannot carry: *why* it is out of service, all the way to the Brain's opening turn.

  * adapter — fault_status() reads /health.robot_status / fault, and stays quiet
    (None) for a healthy body AND for a v0.1 server that reports neither key
  * context — faulted_robots() collects across the fleet, omits unreachable
    robots rather than inventing a fault for them
  * loop    — the opening user turn names the latched body, the remedy, and
    forbids retrying; the assigned robot is listed first
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from robot_harness.brain.base import BrainDecision, Task
from robot_harness.embodiment.base import RobotFault, SupportsFaultStatus
from robot_harness.embodiment.interface.http import HttpAgentServerClient
from robot_harness.embodiment.real.agent_server import RealAgentServerAdapter
from robot_harness.errors import RobotOfflineError
from robot_harness.runtime.agent_loop import AgentLoop
from robot_harness.runtime.harness_context import HarnessContext


def _health_adapter(payload: dict[str, Any], robot_id: str = "r0") -> RealAgentServerAdapter:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json=payload)

    client = HttpAgentServerClient(
        "http://robot",
        robot_id,
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )
    return RealAgentServerAdapter(robot_id, client=client)


# ---------------------------------------------------------------------------
# Adapter — /health.robot_status / fault
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fault_status_reads_latched_fault() -> None:
    adapter = _health_adapter(
        {
            "status": "ok",
            "robot_status": "fault",
            "available_verbs": [],
            "fault": {
                "code": "safety_violation",
                "reason": "workspace envelope exceeded on joint 3",
                "since_s": 42.7,
            },
        },
        "go2",
    )
    fault = await adapter.fault_status()
    assert fault is not None
    assert fault.robot_id == "go2"
    assert fault.code == "safety_violation"
    assert "joint 3" in fault.reason
    assert fault.since_s == pytest.approx(42.7)


@pytest.mark.asyncio
async def test_fault_status_is_none_for_a_healthy_body() -> None:
    adapter = _health_adapter({"status": "ok", "robot_status": "ready", "fault": None})
    assert await adapter.fault_status() is None


@pytest.mark.asyncio
async def test_fault_status_is_none_for_a_v01_server() -> None:
    """A server predating the latch reports neither key. It is a legal backend,
    not a broken one — the absence must not read as a fault."""
    adapter = _health_adapter({"status": "ok", "available_verbs": ["home"]})
    assert await adapter.fault_status() is None


@pytest.mark.asyncio
async def test_busy_is_not_a_fault() -> None:
    adapter = _health_adapter({"status": "ok", "robot_status": "busy"})
    assert await adapter.fault_status() is None


@pytest.mark.asyncio
async def test_malformed_fault_object_still_reports_the_fault() -> None:
    """That the body is latched matters more than why: a typo upstream must not
    cost the whole signal."""
    adapter = _health_adapter(
        {"robot_status": "fault", "fault": "workspace exceeded"}  # a string, not an object
    )
    fault = await adapter.fault_status()
    assert fault is not None
    assert fault.code == ""


@pytest.mark.asyncio
async def test_non_numeric_since_s_degrades_to_zero() -> None:
    adapter = _health_adapter(
        {"robot_status": "fault", "fault": {"code": "cold_start", "since_s": "a while"}}
    )
    fault = await adapter.fault_status()
    assert fault is not None
    assert fault.code == "cold_start"
    assert fault.since_s == 0.0


# ---------------------------------------------------------------------------
# HarnessContext.faulted_robots — fleet probe
# ---------------------------------------------------------------------------


class _StubFaultAdapter:
    """SupportsFaultStatus stand-in with a fixed answer."""

    def __init__(self, fault: RobotFault | None, *, offline: bool = False) -> None:
        self._fault = fault
        self._offline = offline

    async def fault_status(self) -> RobotFault | None:
        if self._offline:
            raise RobotOfflineError("stub /health unreachable", robot_id="stub")
        return self._fault


class _NoFaultCapabilityAdapter:
    """An adapter without the SupportsFaultStatus capability."""


def _ctx(adapters: dict[str, Any]) -> HarnessContext:
    ctx = HarnessContext.build()
    ctx.embodiment_adapters = adapters
    return ctx


def test_stub_satisfies_the_protocol() -> None:
    assert isinstance(_StubFaultAdapter(None), SupportsFaultStatus)
    assert not isinstance(_NoFaultCapabilityAdapter(), SupportsFaultStatus)


@pytest.mark.asyncio
async def test_faulted_robots_collects_only_latched_bodies() -> None:
    ctx = _ctx(
        {
            "go2": _StubFaultAdapter(RobotFault(robot_id="go2", code="operator_abort")),
            "arm1": _StubFaultAdapter(None),
        }
    )
    faults = await ctx.faulted_robots()
    assert set(faults) == {"go2"}
    assert faults["go2"].code == "operator_abort"


@pytest.mark.asyncio
async def test_unreachable_robot_is_omitted_not_guessed_at() -> None:
    """Offline is its own condition with its own error path; inventing a fault
    for it would put words in the operator's mouth."""
    ctx = _ctx({"go2": _StubFaultAdapter(None, offline=True)})
    assert await ctx.faulted_robots() == {}


@pytest.mark.asyncio
async def test_backends_without_the_capability_are_skipped() -> None:
    ctx = _ctx({"legacy": _NoFaultCapabilityAdapter()})
    assert await ctx.faulted_robots() == {}


# ---------------------------------------------------------------------------
# AgentLoop — the opening turn says which body is down and why
# ---------------------------------------------------------------------------


def test_format_faults_leads_with_the_assigned_robot() -> None:
    task = Task(task_id="t", description="d", robot_id="arm1")
    faults = {
        "go2": RobotFault(robot_id="go2", code="cold_start", reason="pose unverified"),
        "arm1": RobotFault(robot_id="arm1", code="safety_violation", reason="envelope"),
    }
    text = AgentLoop._format_faults(task, faults)
    assert text.index("arm1") < text.index("go2")
    assert "assigned to this task is FAULTED" in text
    assert "safety_violation: envelope" in text
    assert "POST /reset" in text
    assert "Do not retry" in text


def test_format_faults_when_the_assigned_robot_is_fine() -> None:
    task = Task(task_id="t", description="d", robot_id="arm1")
    text = AgentLoop._format_faults(task, {"go2": RobotFault(robot_id="go2", code="cold_start")})
    assert "These robots are FAULTED" in text
    assert "go2 (cold_start)" in text


def test_format_faults_is_empty_when_nothing_is_latched() -> None:
    task = Task(task_id="t", description="d", robot_id="arm1")
    assert AgentLoop._format_faults(task, {}) == ""


class _CapturingBrain:
    """Records the conversation it is given, then immediately completes."""

    def __init__(self) -> None:
        self.seen_messages: list[Any] = []

    @property
    def supports_streaming(self) -> bool:
        return False

    async def decide(self, messages: list[Any], tools: list[Any], **_: Any) -> BrainDecision:
        self.seen_messages = list(messages)
        return BrainDecision(decision_type="respond", message="done")


@pytest.mark.asyncio
async def test_opening_user_turn_names_the_faulted_robot(tmp_workspace: Path) -> None:
    """Without this the Brain either watches its tools vanish for no stated
    reason (single robot) or keeps addressing a body that refuses every verb."""
    ctx = HarnessContext.build()
    ctx.embodiment_adapters = {
        "robot-0": _StubFaultAdapter(
            RobotFault(robot_id="robot-0", code="operator_abort", reason="stopped on request")
        )
    }
    brain = _CapturingBrain()
    task = Task(task_id=str(uuid.uuid4()), description="test task", robot_id="robot-0")
    await AgentLoop(brain, ctx, max_turns=2).run(task)

    user = brain.seen_messages[1]
    assert user["role"] == "user"
    assert "Assigned robot: robot-0" in user["content"]
    assert "FAULTED" in user["content"]
    assert "operator_abort: stopped on request" in user["content"]


@pytest.mark.asyncio
async def test_opening_user_turn_is_unchanged_with_no_faults(tmp_workspace: Path) -> None:
    brain = _CapturingBrain()
    task = Task(task_id=str(uuid.uuid4()), description="test task", robot_id="robot-0")
    await AgentLoop(brain, HarnessContext.build(), max_turns=2).run(task)

    assert "FAULTED" not in brain.seen_messages[1]["content"]
