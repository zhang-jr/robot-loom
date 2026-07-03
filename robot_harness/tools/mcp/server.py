"""HarnessMCPServer — exposes the harness itself as an MCP server (reverse direction).

Normal direction: the harness is an MCP *client* consuming external capability
servers (see :mod:`robot_harness.tools.mcp.client`). This module is the reverse:
external MCP clients (Claude Desktop, other agent harnesses) connect here and
drive the harness's registered tools via ``tools/list`` / ``tools/call``.

Safety: external calls never pass through the AgentLoop, so its per-call
SafetyEnvelope gate does not cover them — the same hole skill-internal calls
would have. Every call therefore goes through :class:`SafetyGatedToolRegistry`,
which runs ``SafetyEnvelope.check()`` before any hardware-bound tool dispatches
(ADR-007). A refused command is reported back to the client as an MCP error
result; refusal at the pre-dispatch gate IS the stop — nothing was dispatched,
the envelope has already persisted the audit entry, and shutting down the whole
server on a refused pre-check would let any external client deny service to the
rest of the fleet.

Visibility: only ``brain_visible`` tools are listed — an external planning agent
gets exactly the vocabulary the internal Brain plans over, not verb-internal
sub-capabilities (see :class:`robot_harness.tools.base.Tool`).

Transports (ADR-002): stdio (spawned by the client) and Streamable HTTP
(``/mcp`` endpoint, standalone process). Both come with the official ``mcp``
SDK; no extra dependencies.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import mcp.types as mcp_types
from mcp.server.lowlevel import Server

from robot_harness.errors import (
    EmbodimentError,
    SafetyEnvelopeViolation,
    ToolError,
    ToolNotFoundError,
)
from robot_harness.observability.tracer import tracer
from robot_harness.runtime.skill_tools import SafetyGatedToolRegistry
from robot_harness.safety.envelope import SafetyEnvelope
from robot_harness.tools.base import ToolContext, ToolRegistry, ToolResult
from robot_harness.tools.schema import ToolSchema


class HarnessMCPServer:
    """Wraps the harness ToolRegistry as an MCP server.

    Args:
        tool_registry: The harness tool registry to expose.
        safety_envelope: Mandatory — hardware-bound tools are gated through it
            exactly as the AgentLoop gates Brain-issued calls. There is no
            envelope-less constructor on purpose.
        server_name: Advertised MCP server name.
        default_timeout_s: Per-call ToolContext timeout for external calls.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        safety_envelope: SafetyEnvelope,
        *,
        server_name: str = "robot-loom",
        default_timeout_s: float = 30.0,
    ) -> None:
        self._registry = tool_registry
        self._gated = SafetyGatedToolRegistry(tool_registry, safety_envelope)
        self._default_timeout_s = default_timeout_s
        self._server = self._build_server(server_name)

    @property
    def server(self) -> Server:
        """Underlying low-level MCP Server — for embedding or in-memory tests."""
        return self._server

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the exposed tools in MCP tool-definition format."""
        return [s.to_mcp_tool() for s in self._visible_schemas()]

    # ------------------------------------------------------------------
    # MCP protocol handlers
    # ------------------------------------------------------------------

    def _visible_schemas(self) -> list[ToolSchema]:
        """Schemas of brain-visible tools — the external planning vocabulary."""
        return [
            schema
            for schema in self._registry.list_schemas()
            if getattr(self._registry.get(schema.name), "brain_visible", True)
        ]

    def _build_server(self, server_name: str) -> Server:
        server: Server = Server(server_name)

        @server.list_tools()
        async def _list_tools() -> list[mcp_types.Tool]:
            return [mcp_types.Tool(**s.to_mcp_tool()) for s in self._visible_schemas()]

        @server.call_tool()
        async def _call_tool(
            name: str, arguments: dict[str, Any]
        ) -> dict[str, Any] | mcp_types.CallToolResult:
            return await self._handle_call(name, arguments)

        return server

    async def _handle_call(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | mcp_types.CallToolResult:
        """Invoke one harness tool on behalf of an external MCP client.

        Success → the tool's ``output`` dict as structured content (the SDK
        validates it against the published outputSchema — a mismatch is a real
        contract violation and surfaces as an error). Failure → an explicit
        ``isError`` result. The SDK has already validated ``arguments`` against
        the published inputSchema before this handler runs.
        """
        ctx = ToolContext.create(
            str(arguments.get("robot_id", "")),
            timeout_s=self._default_timeout_s,
        )
        tracer.event(
            "mcp_server.call",
            tool_name=name,
            trace_id=ctx.trace_id,
            robot_id=ctx.robot_id,
        )

        try:
            tool = self._gated.get(name)
        except ToolNotFoundError as exc:
            return _error_result(str(exc), error_type="ToolNotFoundError")

        try:
            result = await tool.invoke(arguments, ctx)
        except SafetyEnvelopeViolation as exc:
            # NOT catch-and-continue: the pre-dispatch gate already refused the
            # command (nothing reached hardware) and persisted the audit entry.
            # This only translates the refusal into an MCP error result instead
            # of letting the SDK stringify it anonymously.
            tracer.event(
                "mcp_server.safety_refused",
                tool_name=name,
                trace_id=ctx.trace_id,
                robot_id=ctx.robot_id,
                violated_rules=exc.violated_rules,
            )
            return _error_result(
                f"SafetyEnvelope refused the command: {exc}",
                error_type="SafetyEnvelopeViolation",
            )
        except (ToolError, EmbodimentError) as exc:
            # Same policy as AgentLoop._invoke_tool: a backend/robot fault is a
            # failed call the caller can react to, not a server crash.
            return _error_result(str(exc), error_type=type(exc).__name__)

        tracer.event(
            "mcp_server.result",
            tool_name=name,
            trace_id=ctx.trace_id,
            success=result.success,
            latency_ms=result.latency_ms,
        )
        if not result.success:
            return _failure_to_result(result)
        return result.output if result.output is not None else {}

    # ------------------------------------------------------------------
    # Transports
    # ------------------------------------------------------------------

    async def start(self, transport: str = "stdio") -> None:
        """Start the MCP server on the given transport (blocks until shutdown).

        Args:
            transport: ``"stdio"`` or an HTTP bind address like ``"0.0.0.0:8080"``
                (serves Streamable HTTP on ``/mcp``).
        """
        if transport == "stdio":
            await self._run_stdio()
            return
        host, _, port_s = transport.removeprefix("http://").partition(":")
        if not host or not port_s.isdigit():
            raise ValueError(
                f"Invalid MCP transport {transport!r}: expected 'stdio' or 'host:port'"
            )
        await self._run_streamable_http(host, int(port_s))

    async def _run_stdio(self) -> None:
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read, write):
            await self._server.run(read, write, self._server.create_initialization_options())

    async def _run_streamable_http(self, host: str, port: int) -> None:
        import uvicorn
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.routing import Mount

        # Stateless: every tool call is self-contained (ToolContext is created
        # per call), so clients don't need session resumption.
        manager = StreamableHTTPSessionManager(app=self._server, stateless=True)

        @contextlib.asynccontextmanager
        async def lifespan(_app: Starlette) -> AsyncIterator[None]:
            async with manager.run():
                tracer.event("mcp_server.listening", host=host, port=port, path="/mcp")
                yield

        # Mounted at the root: Mount("/mcp") would 307-redirect the bare "/mcp"
        # POST to "/mcp/", which MCP clients do not follow. The session manager
        # ignores the path, so both ".../mcp" and "/" work as endpoint URLs.
        app = Starlette(
            routes=[Mount("/", app=manager.handle_request)],
            lifespan=lifespan,
        )
        server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
        await server.serve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_result(message: str, *, error_type: str) -> mcp_types.CallToolResult:
    """Build an explicit MCP error result (bypasses output-schema validation)."""
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=message)],
        structuredContent={"error": message, "error_type": error_type},
        isError=True,
    )


def _failure_to_result(result: ToolResult) -> mcp_types.CallToolResult:
    payload: dict[str, Any] = {
        "error": result.error or "tool reported failure",
        "error_type": result.error_type or "ToolFailure",
    }
    if result.output:
        payload["output"] = result.output
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=payload["error"])],
        structuredContent=payload,
        isError=True,
    )
