"""Tests for HarnessMCPServer — the harness exposed as an MCP server.

Runs a REAL MCP handshake over in-memory streams (official SDK test helper) and
a REAL SafetyEnvelope (per CLAUDE.md: no mock SafetyEnvelope, ever).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from robot_harness.config.schema import SafetyConfig
from robot_harness.embodiment.base import EmbodimentCommand
from robot_harness.safety.audit_log import SafetyAuditLog
from robot_harness.safety.envelope import SafetyEnvelope
from robot_harness.tools.base import ToolContext, ToolRegistry, ToolResult
from robot_harness.tools.mcp.server import HarnessMCPServer
from robot_harness.tools.schema import ToolBackend, ToolSchema

# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


class EchoTool:
    name = "test.echo"
    backend: ToolBackend = "inproc"
    is_idempotent = True
    is_cancellable = False
    schema = ToolSchema(
        name="test.echo",
        description="Echo a message back",
        input_schema={
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
        },
        output_schema={
            "type": "object",
            "properties": {"echoed": {"type": "string"}},
            "required": ["echoed"],
        },
    )

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"echoed": args["msg"]},
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass


class FailingTool:
    name = "test.fail"
    backend: ToolBackend = "inproc"
    is_idempotent = True
    is_cancellable = False
    schema = ToolSchema(
        name="test.fail",
        description="Always reports failure",
        input_schema={"type": "object", "properties": {}},
        output_schema={
            "type": "object",
            "properties": {"never": {"type": "string"}},
            "required": ["never"],
        },
    )

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=False,
            error="backend exploded",
            error_type="ToolBackendUnreachableError",
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass


class HiddenTool(EchoTool):
    name = "test.hidden"
    brain_visible = False
    schema = ToolSchema(
        name="test.hidden",
        description="Verb-internal sub-capability, not for external planners",
        input_schema={"type": "object", "properties": {}},
    )


class FakeArmTool:
    """Hardware-bound tool: records dispatches so tests can prove refusal."""

    name = "test.move_arm"
    backend: ToolBackend = "inproc"
    is_idempotent = False
    is_cancellable = True
    hardware_bound = True
    schema = ToolSchema(
        name="test.move_arm",
        description="Move end effector to a cartesian pose",
        input_schema={
            "type": "object",
            "properties": {
                "robot_id": {"type": "string"},
                "pose": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["robot_id", "pose"],
        },
        output_schema={
            "type": "object",
            "properties": {"reached": {"type": "array", "items": {"type": "number"}}},
            "required": ["reached"],
        },
    )

    def __init__(self) -> None:
        self.dispatched: list[dict[str, Any]] = []

    def to_safety_command(self, args: dict[str, Any], ctx: ToolContext) -> EmbodimentCommand | None:
        return EmbodimentCommand(
            robot_id=args.get("robot_id", ctx.robot_id),
            command_type="cartesian",
            values=list(args["pose"]),
        )

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.dispatched.append(args)
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"reached": list(args["pose"])},
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def audit_log(tmp_path: Path) -> SafetyAuditLog:
    return SafetyAuditLog(path=tmp_path / "audit.jsonl")


@pytest.fixture()
def arm_tool() -> FakeArmTool:
    return FakeArmTool()


@pytest.fixture()
def harness_server(audit_log: SafetyAuditLog, arm_tool: FakeArmTool) -> HarnessMCPServer:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(FailingTool())
    registry.register(HiddenTool())
    registry.register(arm_tool)
    envelope = SafetyEnvelope(SafetyConfig(), audit_log=audit_log)
    return HarnessMCPServer(registry, envelope)


# ---------------------------------------------------------------------------
# tools/list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tools_exposes_visible_schemas(harness_server: HarnessMCPServer) -> None:
    async with create_connected_server_and_client_session(harness_server.server) as session:
        listed = await session.list_tools()
    by_name = {t.name: t for t in listed.tools}
    assert "test.echo" in by_name
    assert "test.move_arm" in by_name
    assert by_name["test.echo"].inputSchema == EchoTool.schema.input_schema
    assert by_name["test.echo"].outputSchema == EchoTool.schema.output_schema


@pytest.mark.asyncio
async def test_brain_invisible_tool_is_hidden(harness_server: HarnessMCPServer) -> None:
    async with create_connected_server_and_client_session(harness_server.server) as session:
        listed = await session.list_tools()
    assert "test.hidden" not in {t.name for t in listed.tools}


# ---------------------------------------------------------------------------
# tools/call — plain tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_roundtrip_structured_output(harness_server: HarnessMCPServer) -> None:
    async with create_connected_server_and_client_session(harness_server.server) as session:
        result = await session.call_tool("test.echo", {"msg": "hello"})
    assert result.isError is False
    assert result.structuredContent == {"echoed": "hello"}


@pytest.mark.asyncio
async def test_call_unknown_tool_is_error(harness_server: HarnessMCPServer) -> None:
    async with create_connected_server_and_client_session(harness_server.server) as session:
        result = await session.call_tool("test.nonexistent", {})
    assert result.isError is True


@pytest.mark.asyncio
async def test_input_schema_violation_is_error(harness_server: HarnessMCPServer) -> None:
    # "msg" is required — the SDK validates against the published inputSchema.
    async with create_connected_server_and_client_session(harness_server.server) as session:
        result = await session.call_tool("test.echo", {})
    assert result.isError is True


@pytest.mark.asyncio
async def test_tool_failure_maps_to_is_error(harness_server: HarnessMCPServer) -> None:
    async with create_connected_server_and_client_session(harness_server.server) as session:
        result = await session.call_tool("test.fail", {})
    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["error_type"] == "ToolBackendUnreachableError"
    assert "backend exploded" in result.structuredContent["error"]


# ---------------------------------------------------------------------------
# tools/call — hardware-bound tools gated by the REAL SafetyEnvelope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hardware_tool_in_bounds_dispatches(
    harness_server: HarnessMCPServer, arm_tool: FakeArmTool, audit_log: SafetyAuditLog
) -> None:
    async with create_connected_server_and_client_session(harness_server.server) as session:
        result = await session.call_tool(
            "test.move_arm", {"robot_id": "r0", "pose": [0.5, 0.5, 1.0]}
        )
    assert result.isError is False
    assert result.structuredContent == {"reached": [0.5, 0.5, 1.0]}
    assert len(arm_tool.dispatched) == 1
    assert any(e.outcome == "passed" for e in audit_log.tail(5))


@pytest.mark.asyncio
async def test_hardware_tool_violation_refused_not_dispatched(
    harness_server: HarnessMCPServer, arm_tool: FakeArmTool, audit_log: SafetyAuditLog
) -> None:
    """An out-of-workspace pose from an external client must be refused BEFORE
    the tool dispatches, with the violation audited — the reverse channel is not
    a backdoor around the envelope."""
    async with create_connected_server_and_client_session(harness_server.server) as session:
        result = await session.call_tool(
            "test.move_arm", {"robot_id": "r0", "pose": [50.0, 0.0, 1.0]}
        )
    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["error_type"] == "SafetyEnvelopeViolation"
    assert arm_tool.dispatched == []
    assert any(e.outcome == "violated" for e in audit_log.tail(5))


# ---------------------------------------------------------------------------
# Exposure surface — no host-execution tools on the reverse channel (ISS-033)
# ---------------------------------------------------------------------------


def test_default_context_never_exposes_host_execution_tools(tmp_path: Path) -> None:
    """Regression for ISS-033: ``robot-loom mcp serve`` exposes every
    brain-visible tool in the registry to arbitrary external clients, so the
    context it serves must never carry shell / filesystem tools."""
    from robot_harness.config.schema import HarnessConfig
    from robot_harness.runtime.harness_context import HarnessContext

    ctx = HarnessContext.build(config=HarnessConfig())
    server = HarnessMCPServer(
        ctx.tool_registry,
        ctx.safety_envelope,
    )
    exposed = {t["name"] for t in server.list_tools()}
    assert "shell_run" not in exposed
    assert "fs_read" not in exposed
    assert "fs_write" not in exposed
