"""MCPServer — exposes the harness itself as an MCP server (reverse direction).

Currently an interface stub.
"""

from __future__ import annotations

from typing import Any

from robot_harness.tools.base import ToolRegistry


# TODO (ADR-002): implement the full MCP server protocol so external agents can
# call harness tools via MCP.
class HarnessMCPServer:
    """Wraps the harness ToolRegistry as an MCP server.

    External MCP clients (e.g. Claude Desktop, other agent harnesses) can
    connect to this server and call any registered tool as if it were a
    native MCP tool.

    Implementation notes for when this is wired in:
    - Use the official MCP Python SDK server primitives.
    - Expose all tools in ToolRegistry via ``tools/list`` and ``tools/call``.
    - Expose SkillManifests via ``prompts/list`` (MCP prompt primitive).
    - Large observation data (images, point clouds) via ``resources/read``.
    """

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._registry = tool_registry

    async def start(self, transport: str = "stdio") -> None:
        """Start the MCP server on the given transport.

        Args:
            transport: ``"stdio"`` or an HTTP bind address like ``"0.0.0.0:8080"``.
        """
        raise NotImplementedError(
            "HarnessMCPServer.start() requires MCP SDK — "
            "install 'mcp' and implement the transport. See ADR-002."
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """Return all registered tools in MCP tool-definition format."""
        return [s.to_mcp_tool() for s in self._registry.list_schemas()]
