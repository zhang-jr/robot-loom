"""Generic shell execution tool (NativeTool)."""

from __future__ import annotations

import asyncio
from typing import Any

from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


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
        return True

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = args["command"]
        timeout = float(args.get("timeout_s", 10))
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
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
        except TimeoutError:
            return ToolResult(
                tool_name=self.name,
                trace_id=ctx.trace_id,
                success=False,
                error=f"Command timed out after {timeout}s",
                error_type="TimeoutError",
            )

    async def cancel(self, ctx: ToolContext) -> None:
        ctx.cancel()
