"""Tests for the workspace standing-context overlay (ADR-035)."""

from __future__ import annotations

import io
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from robot_harness.brain.base import BrainDecision, Task
from robot_harness.brain.prompt_assembly import OVERLAY_WARN_CHARS, workspace_prompt_overlay
from robot_harness.observability.tracer import tracer
from robot_harness.runtime.agent_loop import AgentLoop
from robot_harness.runtime.harness_context import HarnessContext


@pytest.fixture()
def trace_buffer() -> Iterator[io.StringIO]:
    """Redirect the global tracer's JSON sink to a buffer for the test."""
    buf = io.StringIO()
    old = tracer._sink
    tracer.configure(buf)
    yield buf
    tracer.configure(old)


_MISSION = "# Mission\n\nPatrol the lab; recharge below 20%."
_ROBOT = '# Fleet\n\n"the table" is at map pose [1.5, 0.0].'


# ---------------------------------------------------------------------------
# workspace_prompt_overlay
# ---------------------------------------------------------------------------


def test_no_workspace_files_returns_none(tmp_workspace: Path) -> None:
    assert workspace_prompt_overlay() is None


def test_blank_files_are_skipped(tmp_workspace: Path) -> None:
    (tmp_workspace / "MISSION.md").write_text("   \n\n  ", encoding="utf-8")
    (tmp_workspace / "ROBOT.md").write_text("", encoding="utf-8")
    assert workspace_prompt_overlay() is None


def test_mission_only(tmp_workspace: Path) -> None:
    (tmp_workspace / "MISSION.md").write_text(_MISSION, encoding="utf-8")
    overlay = workspace_prompt_overlay()
    assert overlay is not None
    assert "## Mission (MISSION.md)" in overlay
    assert "recharge below 20%" in overlay
    assert "ROBOT.md" not in overlay


def test_robot_only(tmp_workspace: Path) -> None:
    (tmp_workspace / "ROBOT.md").write_text(_ROBOT, encoding="utf-8")
    overlay = workspace_prompt_overlay()
    assert overlay is not None
    assert "## Fleet & site notes (ROBOT.md)" in overlay
    assert "[1.5, 0.0]" in overlay
    assert "MISSION.md" not in overlay


def test_both_files_mission_first(tmp_workspace: Path) -> None:
    (tmp_workspace / "MISSION.md").write_text(_MISSION, encoding="utf-8")
    (tmp_workspace / "ROBOT.md").write_text(_ROBOT, encoding="utf-8")
    overlay = workspace_prompt_overlay()
    assert overlay is not None
    assert overlay.index("## Mission (MISSION.md)") < overlay.index(
        "## Fleet & site notes (ROBOT.md)"
    )


def test_injection_is_traced(tmp_workspace: Path, trace_buffer: io.StringIO) -> None:
    (tmp_workspace / "MISSION.md").write_text(_MISSION, encoding="utf-8")
    assert workspace_prompt_overlay(trace_id="t-1") is not None
    assert "brain.workspace_overlay" in trace_buffer.getvalue()


def test_oversize_overlay_warns_but_is_not_truncated(
    tmp_workspace: Path, trace_buffer: io.StringIO
) -> None:
    big = "# Mission\n\n" + ("all work and no play " * (OVERLAY_WARN_CHARS // 20))
    (tmp_workspace / "MISSION.md").write_text(big, encoding="utf-8")
    overlay = workspace_prompt_overlay()
    assert overlay is not None
    # User assets are injected whole — the budget is a warning, never a knife.
    assert big.strip() in overlay
    assert "brain.workspace_overlay_oversize" in trace_buffer.getvalue()


# ---------------------------------------------------------------------------
# AgentLoop wiring — the overlay rides in the opening system turn
# ---------------------------------------------------------------------------


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


def _task() -> Task:
    return Task(task_id=str(uuid.uuid4()), description="test task", robot_id="robot-0")


@pytest.mark.asyncio
async def test_system_prompt_carries_workspace_overlay(tmp_workspace: Path) -> None:
    (tmp_workspace / "MISSION.md").write_text(_MISSION, encoding="utf-8")
    (tmp_workspace / "ROBOT.md").write_text(_ROBOT, encoding="utf-8")
    brain = _CapturingBrain()
    await AgentLoop(brain, HarnessContext.build(), max_turns=2).run(_task())

    system = brain.seen_messages[0]
    assert system["role"] == "system"
    assert "recharge below 20%" in system["content"]
    assert "[1.5, 0.0]" in system["content"]
    # Standing context supplements the harness prompt, never replaces it.
    assert "robot task planner" in system["content"]


@pytest.mark.asyncio
async def test_system_prompt_unchanged_without_workspace_files(tmp_workspace: Path) -> None:
    brain = _CapturingBrain()
    await AgentLoop(brain, HarnessContext.build(), max_turns=2).run(_task())

    system = brain.seen_messages[0]
    assert system["role"] == "system"
    assert "Operator standing context" not in system["content"]


@pytest.mark.asyncio
async def test_opening_user_turn_states_assigned_robot(tmp_workspace: Path) -> None:
    """The robot assignment is a conversation fact, not a ROBOT.md inference:
    the Brain fills robot_id args from what it reads, and a stale workspace
    ROBOT.md otherwise misaddresses every call."""
    brain = _CapturingBrain()
    await AgentLoop(brain, HarnessContext.build(), max_turns=2).run(_task())

    user = brain.seen_messages[1]
    assert user["role"] == "user"
    assert "Task: test task" in user["content"]
    assert "Assigned robot: robot-0" in user["content"]
