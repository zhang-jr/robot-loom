"""Unit tests for MCP client/server stubs (Phase 1 interface verification)."""

from __future__ import annotations

import pytest

from robot_harness.tools.base import ToolRegistry
from robot_harness.tools.mcp.client import MCPClientSession, MCPTool
from robot_harness.tools.mcp.server import HarnessMCPServer

# ---------------------------------------------------------------------------
# MCPTool interface
# ---------------------------------------------------------------------------


def test_mcp_tool_properties() -> None:
    tool = MCPTool(
        tool_name="my_server.do_thing",
        description="Does a thing.",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        server_url="http://localhost:9000",
    )
    assert tool.name == "my_server.do_thing"
    assert tool.backend == "mcp"
    assert tool.schema.description == "Does a thing."


def test_mcp_tool_can_be_registered() -> None:
    registry = ToolRegistry()
    tool = MCPTool(
        tool_name="mcp.test",
        description="test",
        input_schema={"type": "object"},
    )
    registry.register(tool)
    assert "mcp.test" in registry


@pytest.mark.asyncio
async def test_mcp_tool_invoke_raises_not_implemented() -> None:
    from robot_harness.tools.base import ToolContext

    tool = MCPTool(
        tool_name="mcp.not_ready",
        description="stub",
        input_schema={"type": "object"},
    )
    ctx = ToolContext.create("robot-0")
    with pytest.raises(NotImplementedError):
        await tool.invoke({}, ctx)


def test_mcp_tool_schema_to_mcp_format() -> None:
    tool = MCPTool(
        tool_name="mcp.tool",
        description="desc",
        input_schema={"type": "object"},
    )
    mcp_def = tool.schema.to_mcp_tool()
    assert mcp_def["name"] == "mcp.tool"
    assert "inputSchema" in mcp_def


# ---------------------------------------------------------------------------
# MCPClientSession interface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_client_session_context_manager() -> None:
    async with MCPClientSession("http://localhost:9000") as session:
        # Phase 1: no error on enter/exit (no real connection)
        assert session is not None


@pytest.mark.asyncio
async def test_mcp_client_call_tool_raises_not_implemented() -> None:
    session = MCPClientSession("http://localhost:9000")
    with pytest.raises(NotImplementedError):
        await session.call_tool("some_tool", {})


@pytest.mark.asyncio
async def test_mcp_client_list_tools_raises_not_implemented() -> None:
    session = MCPClientSession("http://localhost:9000")
    with pytest.raises(NotImplementedError):
        await session.list_tools()


# ---------------------------------------------------------------------------
# HarnessMCPServer interface
# ---------------------------------------------------------------------------


def test_harness_mcp_server_list_tools_empty_registry() -> None:
    registry = ToolRegistry()
    server = HarnessMCPServer(registry)
    tools = server.list_tools()
    assert tools == []


def test_harness_mcp_server_list_tools_non_empty() -> None:
    from robot_harness.tools.perception.yolo_adapter import YoloDetectionTool

    registry = ToolRegistry()
    registry.register(YoloDetectionTool())
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
