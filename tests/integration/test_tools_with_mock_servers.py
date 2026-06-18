"""Integration tests: all tool adapters end-to-end against mock backends.

Perception tools are MCP-backed; the MCP transport is replaced with a
:class:`FakeMCPClientSession` so no real server is required.  Other tools
(grasp, critic, VLA, robot_sdk, memory, etc.) still ship in-process mock
adapters that don't need a server.

Each tool here is exercised exactly as the Brain would invoke it through the
registry, verifying schema-correct round-trip without hardware or GPU.
"""

from __future__ import annotations

from typing import Any

import pytest

from robot_harness.memory.base import NullMemory
from robot_harness.tools.base import ToolContext, ToolRegistry
from robot_harness.tools.critic.vlac_adapter import VlacCriticTool
from robot_harness.tools.generic.fs import ReadFileTool, WriteFileTool
from robot_harness.tools.generic.message import SendMessageTool
from robot_harness.tools.generic.shell import ShellTool
from robot_harness.tools.grasp.anygrasp_adapter import AnyGraspTool
from robot_harness.tools.mcp.client import MCPTool
from robot_harness.tools.memory.ingest_tool import MemoryIngestTool
from robot_harness.tools.memory.query_tool import MemoryQueryTool
from robot_harness.tools.memory.upsert_tool import MemoryUpsertTool
from robot_harness.tools.perception.mcp_bundle import (
    PERCEPTION_DETECT_OBJECTS,
    PERCEPTION_ESTIMATE_DEPTH,
    PERCEPTION_GROUND_PHRASE,
    build_perception_tools,
)
from robot_harness.tools.robot_sdk.http_adapter import RobotSdkTool
from robot_harness.tools.vla.serving_adapter import VlaServingTool
from tests._helpers.mcp import FakeMCPClientSession, make_call_result

_TINY_IMAGE_B64 = "Zm9v"  # 'foo' — opaque to fake server


def _ctx(robot_id: str = "robot-0") -> ToolContext:
    return ToolContext.create(robot_id)


def _build_perception_with_fake() -> tuple[list[MCPTool], FakeMCPClientSession]:
    """Build the perception MCP bundle and attach a single FakeMCPClientSession.

    All perception tools share the same fake so call records and scripted
    responses are centralized.
    """
    fake = FakeMCPClientSession(
        responses={
            PERCEPTION_DETECT_OBJECTS: make_call_result(
                structured={
                    "detections": [
                        {
                            "label": "cup",
                            "confidence": 0.92,
                            "bbox": [0.30, 0.20, 0.55, 0.45],
                            "matched_prompt": "cup",
                        }
                    ],
                    "latency_ms": 41.0,
                    "model_version": "fake-detector@0.0.1",
                }
            ),
            PERCEPTION_ESTIMATE_DEPTH: make_call_result(
                structured={
                    "depth_b64_png16": "ZmFrZQ==",
                    "shape": [240, 320],
                    "min": 0.0,
                    "max": 1.0,
                    "scale_unit": "relative",
                    "latency_ms": 27.0,
                    "model_version": "fake-depth@0.0.1",
                }
            ),
            PERCEPTION_GROUND_PHRASE: make_call_result(
                structured={
                    "detection": {
                        "label": "cup",
                        "confidence": 0.81,
                        "bbox": [0.30, 0.20, 0.55, 0.45],
                    },
                    "latency_ms": 18.0,
                }
            ),
        },
    )
    tools = build_perception_tools("http://fake:9000")
    for tool in tools:
        tool._session_factory = lambda _u, _t, _f=fake: _f  # type: ignore[method-assign,assignment]
    return tools, fake


def _registry_with_all_tools() -> ToolRegistry:
    memory = NullMemory()
    r = ToolRegistry()
    perception_tools, _fake = _build_perception_with_fake()
    for tool in perception_tools:
        r.register(tool)
    r.register(AnyGraspTool())
    r.register(VlacCriticTool())
    r.register(VlaServingTool())
    r.register(RobotSdkTool())
    r.register(ShellTool())
    r.register(ReadFileTool())
    r.register(WriteFileTool())
    r.register(SendMessageTool())
    r.register(MemoryQueryTool(memory))
    r.register(MemoryUpsertTool(memory))
    r.register(MemoryIngestTool(memory))
    return r


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


def test_all_tools_registered() -> None:
    r = _registry_with_all_tools()
    expected = {
        PERCEPTION_DETECT_OBJECTS,
        PERCEPTION_ESTIMATE_DEPTH,
        PERCEPTION_GROUND_PHRASE,
        "grasp.estimate_pose",
        "critic.judge_progress",
        "vla.infer_action",
        "robot_sdk.execute_action",
        "shell_run",
        "fs_read",
        "fs_write",
        "generic.send_message",
        "memory.query",
        "memory.upsert",
        "memory.ingest",
    }
    for name in expected:
        assert name in r, f"Tool '{name}' not registered"


# ---------------------------------------------------------------------------
# Perception (MCP-backed via FakeMCPClientSession)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_objects_via_fake_mcp() -> None:
    tools, fake = _build_perception_with_fake()
    tool = next(t for t in tools if t.name == PERCEPTION_DETECT_OBJECTS)

    result = await tool.invoke(
        {"image_b64": _TINY_IMAGE_B64, "prompts": ["cup"]},
        _ctx(),
    )
    assert result.success
    output = result.output or {}
    assert output["detections"][0]["label"] == "cup"
    assert fake.calls[-1].name == PERCEPTION_DETECT_OBJECTS


@pytest.mark.asyncio
async def test_estimate_depth_via_fake_mcp() -> None:
    tools, _fake = _build_perception_with_fake()
    tool = next(t for t in tools if t.name == PERCEPTION_ESTIMATE_DEPTH)

    result = await tool.invoke({"image_b64": _TINY_IMAGE_B64}, _ctx())
    assert result.success
    output = result.output or {}
    assert output["scale_unit"] == "relative"
    assert output["shape"] == [240, 320]


@pytest.mark.asyncio
async def test_ground_phrase_via_fake_mcp() -> None:
    tools, _fake = _build_perception_with_fake()
    tool = next(t for t in tools if t.name == PERCEPTION_GROUND_PHRASE)

    result = await tool.invoke(
        {"image_b64": _TINY_IMAGE_B64, "phrase": "the cup"},
        _ctx(),
    )
    assert result.success
    assert (result.output or {})["detection"]["label"] == "cup"


# ---------------------------------------------------------------------------
# Other tool mock invocations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anygrasp_mock() -> None:
    tool = AnyGraspTool()
    result = await tool.invoke({"detections": {"count": 1}}, _ctx())
    assert result.success
    pose = (result.output or {}).get("pose", [])
    assert len(pose) == 6


@pytest.mark.asyncio
async def test_vlac_critic_mock() -> None:
    tool = VlacCriticTool()
    result = await tool.invoke({"task_description": "pick the cup"}, _ctx())
    assert result.success
    assert (result.output or {}).get("state") in ("progress", "completion", "failure", "unchanged")


@pytest.mark.asyncio
async def test_vla_serving_mock() -> None:
    tool = VlaServingTool()
    result = await tool.invoke(
        {"image_source": "head_camera", "instruction": "pick the cup", "robot_id": "robot-0"},
        _ctx(),
    )
    assert result.success
    actions = (result.output or {}).get("actions", [])
    assert len(actions) == 1
    assert len(actions[0]) == 7


@pytest.mark.asyncio
async def test_robot_sdk_execute_mock() -> None:
    tool = RobotSdkTool()
    result = await tool.invoke(
        {
            "robot_id": "robot-0",
            "command_type": "cartesian",
            "values": [0.5, 0.0, 0.3, 0.0, 0.0, 0.0],
        },
        _ctx(),
    )
    assert result.success
    assert "action_id" in (result.output or {})


@pytest.mark.asyncio
async def test_send_message_mock() -> None:
    tool = SendMessageTool()
    result = await tool.invoke({"text": "task complete", "level": "info"}, _ctx())
    assert result.success


# ---------------------------------------------------------------------------
# Memory tools with SpatialHubMemory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_upsert_and_query_via_tools() -> None:
    from robot_harness.tools.memory.spatial_hub_adapter import SpatialHubMemory

    hub = SpatialHubMemory()
    upsert = MemoryUpsertTool(hub)
    query = MemoryQueryTool(hub)

    ctx = _ctx()
    upsert_result = await upsert.invoke(
        {"memory_type": "object", "content": {"name": "red_cup"}, "tags": ["cup"]},
        ctx,
    )
    assert upsert_result.success

    query_result = await query.invoke(
        {"memory_type": "object", "text": "red_cup", "top_k": 5},
        ctx,
    )
    assert query_result.success
    hits = (query_result.output or {}).get("hits", [])
    assert len(hits) >= 1


@pytest.mark.asyncio
async def test_memory_ingest_via_tool() -> None:
    from robot_harness.tools.memory.spatial_hub_adapter import SpatialHubMemory

    hub = SpatialHubMemory()
    ingest = MemoryIngestTool(hub)
    result = await ingest.invoke(
        {
            "entries": [
                {"memory_type": "object", "content": {"name": "box"}},
                {"memory_type": "place", "content": {"name": "table"}},
            ]
        },
        _ctx(),
    )
    assert result.success
    assert (result.output or {}).get("count") == 2


# ---------------------------------------------------------------------------
# Schema validation (tool_name in result matches registered name)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_mock_tools_tool_name_in_result() -> None:
    perception_tools, _fake = _build_perception_with_fake()
    tools_and_args: list[tuple[Any, dict[str, Any]]] = [
        (perception_tools[0], {"image_b64": _TINY_IMAGE_B64, "prompts": ["cup"]}),
        (perception_tools[1], {"image_b64": _TINY_IMAGE_B64}),
        (perception_tools[2], {"image_b64": _TINY_IMAGE_B64, "phrase": "cup"}),
        (AnyGraspTool(), {"detections": {}}),
        (VlacCriticTool(), {"task_description": "pick"}),
        (
            VlaServingTool(),
            {"image_source": "cam", "instruction": "pick", "robot_id": "robot-0"},
        ),
        (
            RobotSdkTool(),
            {"robot_id": "robot-0", "command_type": "joint", "values": [0.0] * 6},
        ),
    ]
    for tool, args in tools_and_args:
        ctx = _ctx()
        result = await tool.invoke(args, ctx)
        assert result.tool_name == tool.name
