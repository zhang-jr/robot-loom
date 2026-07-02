"""Loopback integration test: harness-as-MCP-server over real Streamable HTTP.

Starts HarnessMCPServer on 127.0.0.1 and connects with the harness's own
MCPClientSession playing the role of an external agent — proving the reverse
direction end-to-end over the same transport a real external client would use.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

from robot_harness.config.schema import SafetyConfig
from robot_harness.safety.audit_log import SafetyAuditLog
from robot_harness.safety.envelope import SafetyEnvelope
from robot_harness.tools.base import ToolContext, ToolRegistry, ToolResult
from robot_harness.tools.mcp.client import MCPClientSession
from robot_harness.tools.mcp.server import HarnessMCPServer
from robot_harness.tools.schema import ToolBackend, ToolSchema


class PingTool:
    name = "test.ping"
    backend: ToolBackend = "inproc"
    is_idempotent = True
    is_cancellable = False
    schema = ToolSchema(
        name="test.ping",
        description="Reply pong",
        input_schema={"type": "object", "properties": {}},
        output_schema={
            "type": "object",
            "properties": {"pong": {"type": "boolean"}},
            "required": ["pong"],
        },
    )

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(
            tool_name=self.name, trace_id=ctx.trace_id, success=True, output={"pong": True}
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_listening(url: str, timeout_s: float = 10.0) -> None:
    async with httpx.AsyncClient() as client:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            try:
                # Any HTTP status means the server socket is up; the MCP
                # endpoint rejects plain GETs, which is fine.
                await client.get(url, timeout=1.0)
                return
            except httpx.TransportError:
                await asyncio.sleep(0.1)
    raise RuntimeError(f"MCP server at {url} not listening within {timeout_s}s")


@pytest_asyncio.fixture()
async def mcp_url(tmp_path: Path) -> AsyncIterator[str]:
    registry = ToolRegistry()
    registry.register(PingTool())
    envelope = SafetyEnvelope(
        SafetyConfig(), audit_log=SafetyAuditLog(path=tmp_path / "audit.jsonl")
    )
    server = HarnessMCPServer(registry, envelope)

    port = _free_port()
    task = asyncio.create_task(server.start(f"127.0.0.1:{port}"))
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        await _wait_listening(url)
        yield url
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_external_client_lists_and_calls_over_http(mcp_url: str) -> None:
    async with MCPClientSession(mcp_url) as session:
        listed = await session.list_tools()
        assert "test.ping" in {t.name for t in listed.tools}

        result = await session.call_tool("test.ping", {})
        assert result.isError is False
        assert result.structuredContent == {"pong": True}
