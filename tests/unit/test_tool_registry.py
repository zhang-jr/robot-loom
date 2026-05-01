"""Tests for ToolRegistry, ToolSchema, and middleware chain."""

from __future__ import annotations

from typing import Any

import pytest

from robot_harness.errors import (
    ToolCancelledError,
    ToolNotFoundError,
    ToolSchemaViolationError,
    ToolTimeoutError,
)
from robot_harness.tools.base import BrainProfile, ToolContext, ToolRegistry, ToolResult
from robot_harness.tools.middleware.base import build_chain
from robot_harness.tools.middleware.cancel import CancelMiddleware
from robot_harness.tools.middleware.timeout import TimeoutMiddleware
from robot_harness.tools.middleware.trace import TraceMiddleware
from robot_harness.tools.schema import ToolBackend, ToolSchema

# ---------------------------------------------------------------------------
# Minimal Tool fixture
# ---------------------------------------------------------------------------


class _EchoTool:
    """Tool that echoes its args back — always succeeds."""

    name = "echo"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="echo",
        description="Echo args back",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    )

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return True

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"echo": args.get("message", "")},
        )

    async def cancel(self, ctx: ToolContext) -> None:
        ctx.cancel()


class _SlowTool:
    """Tool that hangs until cancelled or times out."""

    name = "slow"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="slow",
        description="Always hangs",
        input_schema={"type": "object", "properties": {}},
    )

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return True

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        import asyncio

        await asyncio.sleep(100)
        return ToolResult(tool_name=self.name, trace_id=ctx.trace_id, success=True)

    async def cancel(self, ctx: ToolContext) -> None:
        ctx.cancel()


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_register_and_get() -> None:
    reg = ToolRegistry()
    reg.register(_EchoTool())
    tool = reg.get("echo")
    assert tool.name == "echo"


def test_get_missing_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        reg.get("ghost")


def test_list_returns_schemas() -> None:
    reg = ToolRegistry()
    reg.register(_EchoTool())
    schemas = reg.list_schemas()
    assert len(schemas) == 1
    assert schemas[0].name == "echo"


def test_list_filter_by_backend() -> None:
    from robot_harness.tools.base import ToolFilter

    reg = ToolRegistry()
    reg.register(_EchoTool())
    schemas = reg.list_schemas(ToolFilter(backend="mcp"))
    assert schemas == []
    schemas = reg.list_schemas(ToolFilter(backend="native"))
    assert len(schemas) == 1


def test_export_for_openai_profile() -> None:
    reg = ToolRegistry()
    reg.register(_EchoTool())
    specs = reg.export_for_brain(BrainProfile(name="openai"))
    assert len(specs) == 1
    assert specs[0]["type"] == "function"
    assert specs[0]["function"]["name"] == "echo"


def test_export_for_anthropic_profile() -> None:
    reg = ToolRegistry()
    reg.register(_EchoTool())
    specs = reg.export_for_brain(BrainProfile(name="anthropic"))
    assert specs[0]["name"] == "echo"
    assert "input_schema" in specs[0]


def test_validate_args_missing_required() -> None:
    reg = ToolRegistry()
    reg.register(_EchoTool())
    with pytest.raises(ToolSchemaViolationError) as exc_info:
        reg.validate_args("echo", {})
    assert "message" in str(exc_info.value.violations)


def test_validate_args_ok() -> None:
    reg = ToolRegistry()
    reg.register(_EchoTool())
    reg.validate_args("echo", {"message": "hello"})  # no exception


def test_contains_operator() -> None:
    reg = ToolRegistry()
    reg.register(_EchoTool())
    assert "echo" in reg
    assert "ghost" not in reg


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_schema_to_openai_function() -> None:
    s = ToolSchema(
        name="foo",
        description="desc",
        input_schema={"type": "object", "properties": {}},
    )
    spec = s.to_openai_function()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "foo"


def test_schema_to_mcp_tool() -> None:
    s = ToolSchema(
        name="foo",
        description="desc",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
    )
    spec = s.to_mcp_tool()
    assert spec["name"] == "foo"
    assert "inputSchema" in spec
    assert "outputSchema" in spec


def test_schema_invalid_name_raises() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ToolSchema(
            name="   ",
            description="x",
            input_schema={"type": "object", "properties": {}},
        )


def test_schema_invalid_input_schema_raises() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ToolSchema(name="x", description="x", input_schema={"properties": {}})


# ---------------------------------------------------------------------------
# Middleware tests
# ---------------------------------------------------------------------------


async def _make_ctx(timeout_s: float = 30.0) -> ToolContext:
    return ToolContext(trace_id="t", robot_id="r", timeout_s=timeout_s)


@pytest.mark.asyncio
async def test_trace_middleware_passes_through() -> None:
    tool = build_chain(_EchoTool(), [TraceMiddleware])
    ctx = await _make_ctx()
    result = await tool.invoke({"message": "hi"}, ctx)
    assert result.success
    assert result.output == {"echo": "hi"}


@pytest.mark.asyncio
async def test_timeout_middleware_raises_on_slow_tool() -> None:
    tool = build_chain(_SlowTool(), [TimeoutMiddleware])
    ctx = await _make_ctx(timeout_s=0.05)
    with pytest.raises(ToolTimeoutError):
        await tool.invoke({}, ctx)


@pytest.mark.asyncio
async def test_cancel_middleware_raises_when_cancelled() -> None:
    tool = build_chain(_EchoTool(), [CancelMiddleware])
    ctx = await _make_ctx()
    ctx.cancel()
    with pytest.raises(ToolCancelledError):
        await tool.invoke({"message": "hi"}, ctx)


@pytest.mark.asyncio
async def test_cancel_middleware_passes_when_not_cancelled() -> None:
    tool = build_chain(_EchoTool(), [CancelMiddleware])
    ctx = await _make_ctx()
    result = await tool.invoke({"message": "ok"}, ctx)
    assert result.success
