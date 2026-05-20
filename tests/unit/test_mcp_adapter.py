"""Unit tests for the MCP client adapter and harness MCP server stub."""

from __future__ import annotations

import pytest

from robot_harness.errors import (
    ToolBackendUnreachableError,
    ToolCancelledError,
)
from robot_harness.tools.base import ToolContext, ToolRegistry
from robot_harness.tools.mcp.client import MCPClientSession, MCPTool
from robot_harness.tools.mcp.server import HarnessMCPServer
from tests._helpers.mcp import (
    FakeMCPClientSession,
    make_call_result,
    make_list_tools_result,
)


def _ctx(timeout_s: float = 5.0) -> ToolContext:
    return ToolContext.create("robot-0", timeout_s=timeout_s)


def _make_tool(
    name: str = "fake.tool",
    server_url: str = "http://fake:9000",
) -> MCPTool:
    return MCPTool(
        tool_name=name,
        description="A fake tool used in unit tests.",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        server_url=server_url,
    )


def _attach_fake(tool: MCPTool, fake: FakeMCPClientSession) -> None:
    """Replace the tool's session factory with one yielding ``fake``."""

    def _factory(_url: str, _timeout: float) -> FakeMCPClientSession:
        return fake

    tool._session_factory = _factory  # type: ignore[method-assign,assignment]


# ---------------------------------------------------------------------------
# MCPTool — interface + schema
# ---------------------------------------------------------------------------


def test_mcp_tool_properties() -> None:
    tool = _make_tool("my_server.do_thing", server_url="http://localhost:9000")
    assert tool.name == "my_server.do_thing"
    assert tool.backend == "mcp"
    assert tool.server_url == "http://localhost:9000"
    assert tool.schema.description.startswith("A fake")
    assert tool.is_idempotent is True
    assert tool.is_cancellable is False


def test_mcp_tool_requires_server_url() -> None:
    with pytest.raises(ValueError, match="server_url"):
        MCPTool(
            tool_name="x",
            description="d",
            input_schema={"type": "object"},
            server_url="",
        )


def test_mcp_tool_can_be_registered() -> None:
    registry = ToolRegistry()
    registry.register(_make_tool("mcp.test"))
    assert "mcp.test" in registry


def test_mcp_tool_schema_to_mcp_format() -> None:
    tool = _make_tool("mcp.tool")
    mcp_def = tool.schema.to_mcp_tool()
    assert mcp_def["name"] == "mcp.tool"
    assert "inputSchema" in mcp_def


# ---------------------------------------------------------------------------
# MCPTool.invoke — marshalling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_marshals_structured_content() -> None:
    tool = _make_tool()
    fake = FakeMCPClientSession(
        responses={"fake.tool": make_call_result(structured={"value": 42, "ok": True})},
    )
    _attach_fake(tool, fake)

    result = await tool.invoke({"x": 1}, _ctx())

    assert result.success is True
    assert result.tool_name == "fake.tool"
    assert result.output == {"value": 42, "ok": True}
    assert len(fake.calls) == 1
    assert fake.calls[0].args == {"x": 1}


@pytest.mark.asyncio
async def test_invoke_falls_back_to_text_content() -> None:
    tool = _make_tool()
    fake = FakeMCPClientSession(
        responses={"fake.tool": make_call_result(text="hello world")},
    )
    _attach_fake(tool, fake)

    result = await tool.invoke({}, _ctx())

    assert result.success is True
    assert result.output == {"text": "hello world"}


@pytest.mark.asyncio
async def test_invoke_marshals_is_error() -> None:
    tool = _make_tool()
    fake = FakeMCPClientSession(
        responses={
            "fake.tool": make_call_result(structured={"reason": "model not loaded"}, is_error=True)
        },
    )
    _attach_fake(tool, fake)

    result = await tool.invoke({}, _ctx())

    assert result.success is False
    assert result.error_type == "MCPToolError"
    assert "model not loaded" in (result.error or "")


# ---------------------------------------------------------------------------
# MCPTool.invoke — error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_returns_failure_on_backend_unreachable() -> None:
    tool = _make_tool()
    fake = FakeMCPClientSession(
        responses={},
        errors={"fake.tool": ToolBackendUnreachableError("server down", tool_name="fake.tool")},
    )
    _attach_fake(tool, fake)

    result = await tool.invoke({}, _ctx())

    assert result.success is False
    assert result.error_type == "ToolBackendUnreachableError"
    assert "server down" in (result.error or "")


@pytest.mark.asyncio
async def test_invoke_raises_when_pre_cancelled() -> None:
    tool = _make_tool()
    fake = FakeMCPClientSession(responses={"fake.tool": make_call_result(structured={})})
    _attach_fake(tool, fake)

    ctx = _ctx()
    ctx.cancel()

    with pytest.raises(ToolCancelledError):
        await tool.invoke({}, ctx)

    # The fake session should never have been entered.
    assert fake.calls == []


@pytest.mark.asyncio
async def test_invoke_records_latency() -> None:
    tool = _make_tool()
    fake = FakeMCPClientSession(responses={"fake.tool": make_call_result(structured={})})
    _attach_fake(tool, fake)

    result = await tool.invoke({}, _ctx())

    assert result.success is True
    assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# MCPClientSession — URL handling
# ---------------------------------------------------------------------------


def test_session_rejects_empty_url() -> None:
    with pytest.raises(ValueError, match="server_url"):
        MCPClientSession("")


@pytest.mark.asyncio
async def test_session_rejects_unknown_scheme() -> None:
    session = MCPClientSession("weird://nowhere")
    with pytest.raises(ToolBackendUnreachableError, match="scheme"):
        await session.__aenter__()


@pytest.mark.asyncio
async def test_session_rejects_stdio_without_command() -> None:
    session = MCPClientSession("stdio://")
    with pytest.raises(ToolBackendUnreachableError, match="no command"):
        await session.__aenter__()


def test_session_raises_if_used_before_enter() -> None:
    """Accessing .session before entering should raise."""
    session = MCPClientSession("http://localhost:9000")
    with pytest.raises(ToolBackendUnreachableError, match="not initialized"):
        _ = session.session


# ---------------------------------------------------------------------------
# HarnessMCPServer (still a Phase-2 stub for reverse direction)
# ---------------------------------------------------------------------------


def test_harness_mcp_server_list_tools_empty_registry() -> None:
    server = HarnessMCPServer(ToolRegistry())
    assert server.list_tools() == []


def test_harness_mcp_server_list_tools_non_empty() -> None:
    registry = ToolRegistry()
    registry.register(_make_tool("perception.detect_objects"))
    server = HarnessMCPServer(registry)

    tools = server.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "perception.detect_objects"
    assert "inputSchema" in tools[0]


@pytest.mark.asyncio
async def test_harness_mcp_server_start_raises_not_implemented() -> None:
    server = HarnessMCPServer(ToolRegistry())
    with pytest.raises(NotImplementedError):
        await server.start()


# ---------------------------------------------------------------------------
# FakeMCPClientSession self-check (sanity: the fake itself behaves)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_session_async_context() -> None:
    fake = FakeMCPClientSession(
        list_tools_result=make_list_tools_result(["a", "b"]),
    )
    async with fake as session:
        assert session.entered is True
        result = await session.list_tools()
        assert {t.name for t in result.tools} == {"a", "b"}
    assert fake.entered is False


@pytest.mark.asyncio
async def test_fake_session_raises_on_unknown_tool() -> None:
    fake = FakeMCPClientSession(responses={})
    async with fake as session:
        with pytest.raises(KeyError):
            await session.call_tool("never_configured", {})


@pytest.mark.asyncio
async def test_fake_session_records_calls() -> None:
    fake = FakeMCPClientSession(responses={"x": make_call_result(structured={"ok": 1})})
    async with fake as session:
        await session.call_tool("x", {"a": 1}, timeout_s=2.0)
        await session.call_tool("x", {"a": 2}, timeout_s=None)
    assert len(fake.calls) == 2
    assert fake.calls[0].args == {"a": 1}
    assert fake.calls[0].timeout_s == 2.0
    assert fake.calls[1].timeout_s is None
