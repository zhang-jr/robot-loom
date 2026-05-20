"""Test fakes for the MCP client layer.

``FakeMCPClientSession`` is an async-context-manager-shaped stand-in for
:class:`robot_harness.tools.mcp.client.MCPClientSession`.  It records every
call and returns scripted responses, so tests don't need a real MCP server.

Typical usage::

    fake = FakeMCPClientSession(
        responses={
            "perception.detect_objects": CallToolResult(
                content=[],
                structuredContent={"detections": []},
                isError=False,
            ),
        },
    )
    tool = MCPTool(..., server_url="http://fake")
    tool._session_factory = lambda url, timeout: fake  # type: ignore[attr-defined]
    result = await tool.invoke({...}, ctx)

The fake also supports raising errors via ``error_for``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

from mcp.types import CallToolResult, ListToolsResult, Tool


@dataclass
class CallRecord:
    """One captured ``call_tool`` invocation."""

    name: str
    args: dict[str, Any]
    timeout_s: float | None


class FakeMCPClientSession:
    """In-memory stand-in for :class:`MCPClientSession`.

    Args:
        responses: ``tool_name → CallToolResult`` mapping.  Missing names raise.
        errors: ``tool_name → exception`` mapping.  Raised instead of returning.
        list_tools_result: What :meth:`list_tools` should return.
        server_url: Reported by :attr:`server_url` (purely cosmetic).
    """

    def __init__(
        self,
        *,
        responses: dict[str, CallToolResult] | None = None,
        errors: dict[str, BaseException] | None = None,
        list_tools_result: ListToolsResult | None = None,
        server_url: str = "fake://test",
    ) -> None:
        self._responses: dict[str, CallToolResult] = responses or {}
        self._errors: dict[str, BaseException] = errors or {}
        self._list_tools_result: ListToolsResult = list_tools_result or ListToolsResult(tools=[])
        self._server_url = server_url
        self.calls: list[CallRecord] = []
        self.list_tools_calls: int = 0
        self._entered: bool = False

    @property
    def server_url(self) -> str:
        return self._server_url

    @property
    def entered(self) -> bool:
        return self._entered

    async def __aenter__(self) -> Self:
        self._entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._entered = False

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> CallToolResult:
        self.calls.append(CallRecord(name=name, args=dict(args), timeout_s=timeout_s))
        if name in self._errors:
            raise self._errors[name]
        if name not in self._responses:
            raise KeyError(
                f"FakeMCPClientSession has no scripted response for tool {name!r}. "
                f"Configured: {sorted(self._responses)}"
            )
        return self._responses[name]

    async def list_tools(self) -> ListToolsResult:
        self.list_tools_calls += 1
        return self._list_tools_result


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def make_call_result(
    structured: dict[str, Any] | None = None,
    *,
    is_error: bool = False,
    text: str | None = None,
) -> CallToolResult:
    """Build a ``CallToolResult`` for tests.

    Provide ``structured`` for the common case where the server returns a JSON
    object via ``structuredContent``.  Use ``text`` for legacy / unstructured
    responses.
    """
    from mcp.types import TextContent

    content: list[Any] = []
    if text is not None:
        content.append(TextContent(type="text", text=text))
    return CallToolResult(
        content=content,
        structuredContent=structured,
        isError=is_error,
    )


def make_list_tools_result(tool_names: list[str]) -> ListToolsResult:
    """Build a ``ListToolsResult`` from bare tool names (empty schemas)."""
    return ListToolsResult(
        tools=[
            Tool(name=n, description=f"fake tool {n}", inputSchema={"type": "object"})
            for n in tool_names
        ]
    )
