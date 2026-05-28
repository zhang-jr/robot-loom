"""MCP client adapter — wraps the official ``mcp`` Python SDK as harness tools.

Two layers:

* :class:`MCPClientSession` — thin async-context wrapper over
  :class:`mcp.ClientSession`, picking the transport from the URL scheme.
* :class:`MCPTool` — implements the :class:`Tool` Protocol by opening a session
  per invocation and marshalling :class:`mcp.types.CallToolResult` into
  :class:`ToolResult`.

URL scheme conventions:

* ``http://...`` / ``https://...`` → Streamable HTTP transport (separate process).
* ``stdio://command arg1 arg2`` → spawn local subprocess (stdio transport).

Per-call sessions keep the implementation simple and stateless.  Production
deployments that need lower latency should wrap a long-lived ``MCPClientSession``
externally (TODO: connection pool, tracked in Phase 2 follow-up).
"""

from __future__ import annotations

import time
from contextlib import AsyncExitStack
from datetime import timedelta
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlsplit

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ListToolsResult

from robot_harness.errors import (
    ToolBackendUnreachableError,
    ToolCancelledError,
    ToolError,
    ToolTimeoutError,
)
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class MCPClientSession:
    """Async-context wrapper around :class:`mcp.ClientSession`.

    Use as::

        async with MCPClientSession("http://localhost:8765") as session:
            result = await session.call_tool("perception.detect_objects", {...})

    Raises :class:`ToolBackendUnreachableError` if the transport cannot be
    established or the MCP handshake fails.
    """

    def __init__(self, server_url: str, *, init_timeout_s: float = 30.0) -> None:
        if not server_url:
            raise ValueError("MCPClientSession requires a non-empty server_url")
        self._server_url = server_url
        self._init_timeout_s = init_timeout_s
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    @property
    def server_url(self) -> str:
        return self._server_url

    @property
    def session(self) -> ClientSession:
        """Underlying mcp ClientSession (raises if not yet entered)."""
        if self._session is None:
            raise ToolBackendUnreachableError(
                "MCP session not initialized — use 'async with MCPClientSession(...)'",
                module_name="tools.mcp.client",
            )
        return self._session

    async def __aenter__(self) -> Self:
        stack = AsyncExitStack()
        try:
            read, write = await self._open_transport(stack)
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except ToolError:
            await stack.aclose()
            raise
        except Exception as exc:
            await stack.aclose()
            raise ToolBackendUnreachableError(
                f"MCP session init failed for {self._server_url!r}: {exc}",
                module_name="tools.mcp.client",
            ) from exc

        self._stack = stack
        self._session = session
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def _open_transport(self, stack: AsyncExitStack) -> tuple[Any, Any]:
        """Pick a transport based on the URL scheme and enter it on ``stack``.

        Returns the ``(read_stream, write_stream)`` pair used by ``ClientSession``.
        """
        scheme = urlsplit(self._server_url).scheme.lower()
        if scheme in ("http", "https"):
            read, write, _get_session_id = await stack.enter_async_context(
                streamable_http_client(self._server_url, timeout=self._init_timeout_s)
            )
            return read, write
        if scheme == "stdio":
            command_line = self._server_url[len("stdio://") :].strip()
            if not command_line:
                raise ToolBackendUnreachableError(
                    f"Invalid stdio MCP URL (no command): {self._server_url!r}",
                    module_name="tools.mcp.client",
                )
            parts = command_line.split()
            params = StdioServerParameters(command=parts[0], args=parts[1:])
            read, write = await stack.enter_async_context(stdio_client(params))
            return read, write
        raise ToolBackendUnreachableError(
            f"Unsupported MCP transport scheme {scheme!r} in URL {self._server_url!r}; "
            "expected http://, https://, or stdio://",
            module_name="tools.mcp.client",
        )

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> CallToolResult:
        """Forward a tool call to the MCP server.

        Maps ``McpError`` → :class:`ToolBackendUnreachableError`,
        ``TimeoutError`` → :class:`ToolTimeoutError`.
        """
        read_timeout = timedelta(seconds=timeout_s) if timeout_s and timeout_s > 0 else None
        try:
            return await self.session.call_tool(name, args, read_timeout_seconds=read_timeout)
        except McpError as exc:
            raise ToolBackendUnreachableError(
                f"MCP server returned error for tool {name!r}: {exc}",
                tool_name=name,
                module_name="tools.mcp.client",
            ) from exc
        except TimeoutError as exc:
            raise ToolTimeoutError(
                f"MCP call_tool({name!r}) timed out after {timeout_s}s",
                tool_name=name,
                module_name="tools.mcp.client",
            ) from exc

    async def list_tools(self) -> ListToolsResult:
        """Return the server's tool catalogue."""
        try:
            return await self.session.list_tools()
        except McpError as exc:
            raise ToolBackendUnreachableError(
                f"MCP list_tools failed: {exc}",
                module_name="tools.mcp.client",
            ) from exc


class MCPTool:
    """Wraps a single remote MCP tool as a harness :class:`Tool`.

    A fresh :class:`MCPClientSession` is opened per ``invoke`` call.  This keeps
    the implementation stateless and easy to reason about; long-running
    deployments should layer a connection pool on top (TODO: Phase 2 follow-up).

    Args:
        tool_name: Tool identifier as published by the MCP server (must match).
        description: Human-readable description (forwarded to the Brain).
        input_schema: JSON Schema for the tool's input.
        server_url: Transport URL.  See :class:`MCPClientSession` for schemes.
        output_schema: Optional JSON Schema for the tool's output.
        is_idempotent: Whether repeated calls with the same args are safe.
    """

    def __init__(
        self,
        tool_name: str,
        description: str,
        input_schema: dict[str, Any],
        server_url: str,
        *,
        output_schema: dict[str, Any] | None = None,
        is_idempotent: bool = True,
    ) -> None:
        if not server_url:
            raise ValueError(f"MCPTool {tool_name!r} requires non-empty server_url")
        self._name = tool_name
        self._server_url = server_url
        self._is_idempotent = is_idempotent
        self._schema = ToolSchema(
            name=tool_name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
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
    def server_url(self) -> str:
        return self._server_url

    @property
    def is_idempotent(self) -> bool:
        return self._is_idempotent

    @property
    def is_cancellable(self) -> bool:
        # Mid-call cancellation through MCP transport is not yet supported;
        # cancellation is honored at the boundary (before opening a session).
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        t0 = time.monotonic()

        if ctx.is_cancelled:
            raise ToolCancelledError(
                f"Tool {self._name!r} cancelled before dispatch",
                trace_id=ctx.trace_id,
                robot_id=ctx.robot_id,
                tool_name=self._name,
                module_name="tools.mcp.client",
            )

        try:
            session_cm = self._session_factory(self._server_url, ctx.timeout_s)
            async with session_cm as session:
                result = await session.call_tool(
                    self._name,
                    args,
                    timeout_s=ctx.timeout_s if ctx.timeout_s > 0 else None,
                )
        except (ToolBackendUnreachableError, ToolTimeoutError) as exc:
            return _failure_result(
                tool_name=self._name,
                trace_id=ctx.trace_id,
                exc=exc,
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        latency_ms = (time.monotonic() - t0) * 1000
        return _marshal_call_result(
            result, tool_name=self._name, trace_id=ctx.trace_id, latency_ms=latency_ms
        )

    async def cancel(self, ctx: ToolContext) -> None:
        ctx.cancel()

    # ------------------------------------------------------------------
    # Hook for tests: override the session factory to inject a fake session
    # without monkey-patching the module.
    # ------------------------------------------------------------------
    def _session_factory(self, server_url: str, init_timeout_s: float) -> Any:
        """Construct the session async-context.  Override for tests."""
        timeout = init_timeout_s if init_timeout_s > 0 else 30.0
        return MCPClientSession(server_url, init_timeout_s=timeout)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _marshal_call_result(
    result: CallToolResult,
    *,
    tool_name: str,
    trace_id: str,
    latency_ms: float,
) -> ToolResult:
    """Translate an MCP ``CallToolResult`` into a harness ``ToolResult``.

    Prefers ``structuredContent`` when the server provides it; otherwise
    concatenates ``content`` text blocks into a ``{"text": "..."}`` payload.
    """
    output: dict[str, Any]
    if result.structuredContent is not None:
        output = result.structuredContent
    else:
        text_parts: list[str] = []
        for block in result.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
        output = {"text": "\n".join(text_parts)} if text_parts else {}

    if result.isError:
        return ToolResult(
            tool_name=tool_name,
            trace_id=trace_id,
            success=False,
            output=output or None,
            error=str(output) or "MCP server reported isError=True",
            error_type="MCPToolError",
            latency_ms=latency_ms,
        )

    return ToolResult(
        tool_name=tool_name,
        trace_id=trace_id,
        success=True,
        output=output,
        latency_ms=latency_ms,
    )


def _failure_result(
    *,
    tool_name: str,
    trace_id: str,
    exc: ToolError,
    latency_ms: float,
) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        trace_id=trace_id,
        success=False,
        error=str(exc),
        error_type=type(exc).__name__,
        latency_ms=latency_ms,
    )
