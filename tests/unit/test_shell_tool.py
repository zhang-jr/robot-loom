"""Tests for ShellTool — timeout must stop the process, not orphan it (ISS-038)."""

from __future__ import annotations

import sys
import time

import pytest

from robot_harness.tools.base import ToolContext
from robot_harness.tools.generic.shell import ShellTool


def _py(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'


@pytest.mark.asyncio
async def test_shell_runs_command_and_captures_output() -> None:
    tool = ShellTool()
    res = await tool.invoke({"command": _py("print('hi')")}, ToolContext.create("r0"))
    assert res.success is True
    assert res.output is not None
    assert res.output["stdout"].strip() == "hi"
    assert res.output["exit_code"] == 0


@pytest.mark.asyncio
async def test_shell_timeout_kills_process_and_reports() -> None:
    """'Timed out' must mean STOPPED: the subprocess is killed and reaped, and
    the result says so — never a background process left running (ISS-038)."""
    tool = ShellTool()
    t0 = time.monotonic()
    res = await tool.invoke(
        {"command": _py("import time; time.sleep(30)"), "timeout_s": 0.5},
        ToolContext.create("r0"),
    )
    elapsed = time.monotonic() - t0
    assert res.success is False
    assert res.error_type == "TimeoutError"
    assert "process killed" in (res.error or "")
    # kill + reap returns promptly — it must not wait out the child's sleep.
    assert elapsed < 10


def test_shell_does_not_advertise_cancel_it_cannot_do() -> None:
    # cancel() cannot reach the running subprocess; advertising cancellable
    # would make the AgentLoop's sibling-cancel silently no-op on it.
    assert ShellTool().is_cancellable is False
