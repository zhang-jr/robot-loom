"""MCPTool — client adapter wrapping an external MCP server.

Phase 1: interface stub — MCP SDK selection is pending (ADR-002 / TODO Phase 0.3).
Phase 2: will use the official MCP Python SDK with stdio + Streamable HTTP transports.
"""

from __future__ import annotations

from typing import Any

from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class MCPTool:
    """Wraps a single MCP server tool as a harness Tool.

    Args:
        tool_name: The tool name as declared by the MCP server.
        description: Human-readable description.
        input_schema: JSON Schema dict for the tool's input.
        server_url: Transport URL (stdio command or Streamable HTTP endpoint).
    """

    def __init__(
        self,
        tool_name: str,
        description: str,
        input_schema: dict[str, Any],
        server_url: str = "",
    ) -> None:
        self._name = tool_name
        self._server_url = server_url
        self._schema = ToolSchema(
            name=tool_name,
            description=description,
            input_schema=input_schema,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    @property
    def backend(self) -> ToolBackend:
        return "mcp"

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        # Phase 2: open MCP session, call tool, close session
        raise NotImplementedError(
            "MCPTool.invoke() requires MCP SDK — install 'mcp' and implement the transport. "
            "See TODO Phase 0.3 and ADR-002."
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass


class MCPClientSession:
    """Placeholder for a persistent MCP client session.

    Phase 2 implementation notes:
    - stdio transport: spawn the server process, communicate over stdin/stdout.
    - Streamable HTTP: connect to the server's HTTP endpoint with SSE.
    - Session lifecycle: open at harness startup, close at shutdown.
    """

    def __init__(self, server_url: str) -> None:
        self._server_url = server_url

    async def __aenter__(self) -> MCPClientSession:
        # Phase 2: initialise MCP handshake
        return self

    async def __aexit__(self, *_: Any) -> None:
        # Phase 2: send MCP shutdown notification
        pass

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        raise NotImplementedError("MCPClientSession requires MCP SDK (Phase 2)")

    async def list_tools(self) -> list[dict[str, Any]]:
        raise NotImplementedError("MCPClientSession requires MCP SDK (Phase 2)")
