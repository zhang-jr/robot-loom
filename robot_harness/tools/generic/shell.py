"""Generic shell execution tool (NativeTool)."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from typing import Any

from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill the shell AND its children, then reap.

    Terminating only the spawned shell orphans its children, which keep running
    and hold the output pipes open — so ``communicate()`` blocks on the surviving
    child's pipes instead of returning. Both branches take the whole tree down:

    - Windows: ``taskkill /T`` walks the PID tree.
    - POSIX: the child is launched in its own session (``start_new_session`` in
      ``invoke``), making it a process-group leader, so one ``killpg`` reaps the
      whole group.
    """
    if sys.platform == "win32":
        killer = await asyncio.create_subprocess_exec(
            "taskkill",
            "/F",
            "/T",
            "/PID",
            str(proc.pid),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
    await proc.communicate()


class ShellTool:
    """Run a shell command and return stdout/stderr.

    Intended for general-purpose scripting tasks.  Do NOT expose this tool
    in production deployments where user input can reach it (command injection).
    """

    name = "shell_run"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="shell_run",
        description="Run a shell command and return its stdout and exit code.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout_s": {"type": "number", "default": 10},
            },
            "required": ["command"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "exit_code": {"type": "integer"},
            },
        },
    )

    @property
    def is_idempotent(self) -> bool:
        return False

    @property
    def is_cancellable(self) -> bool:
        # cancel() cannot reach the running subprocess (no handle survives the
        # call boundary), so don't advertise a cancel capability that would
        # no-op — the timeout below is the enforcement path.
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = args["command"]
        timeout = float(args.get("timeout_s", 10))
        # POSIX: start_new_session makes the child a process-group leader so a
        # timeout can killpg the whole tree (see _kill_process_tree). Windows
        # walks the tree via taskkill /T instead and needs no session flag.
        extra: dict[str, Any] = {} if sys.platform == "win32" else {"start_new_session": True}
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **extra,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            # "Timed out" must mean STOPPED, not "still running in the
            # background" (ISS-038): kill the process tree and reap it so
            # neither the processes nor their pipes leak.
            await _kill_process_tree(proc)
            return ToolResult(
                tool_name=self.name,
                trace_id=ctx.trace_id,
                success=False,
                error=f"Command timed out after {timeout}s (process killed)",
                error_type="TimeoutError",
            )
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=(proc.returncode == 0),
            output={
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
                "exit_code": proc.returncode,
            },
        )

    async def cancel(self, ctx: ToolContext) -> None:
        ctx.cancel()
